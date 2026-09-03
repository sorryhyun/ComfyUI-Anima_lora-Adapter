"""CJK vocab-pack loading and application for the Anima vocab-pack node.

A vocab pack is NOT a LoRA: it is a table of extra text-embedding rows
(``ext_embed [rows, 1024]``, ids ``>= 32128``) plus a JSON sidecar carrying
the segmentation/row maps, trained so Japanese prompt spans land on
dedicated rows instead of degrading to ``<unk>`` on the T5 side. Two patch
surfaces, both reverted by ComfyUI's normal unpatch machinery:

- **CLIP side**: the ``AnimaTokenizer`` is wrapped so that a prompt
  containing a *routed* character gets its ``t5xxl`` id stream re-encoded by
  ``HybridT5Encoder`` (routed spans -> ext ids via the Qwen tokenizer; the
  rest runs through the ordinary T5 tokenizer). Which characters route is
  the pack's own ``route`` rule: the CJK ranges, plus — in packs built after
  2026-09-03 — the symbol tail T5 spiece cannot spell (``^^^`` ``:<`` ``~``
  ``·`` ``×`` ``☆``, emoji), which the stock path folds into one ``<unk>``.
  Prompts with no routed character return the inner tokenizer's stream
  untouched — English-only prompts are bit-identical with or without the
  pack. A pack without ``route`` routes CJK only, exactly as before.
- **MODEL side**: ComfyUI core hardcodes the 32128-row
  ``llm_adapter.embed`` table, and an id ``>= 32128`` would hard-crash the
  embedding lookup. A ``forward_pre_hook`` on ``llm_adapter`` clamps ext
  ids to ``<unk>`` before ``embed`` runs and stashes the originals; a
  ``forward_hook`` on ``llm_adapter.embed`` then overwrites those positions
  with the pack's rows. Hooks are installed via
  ``ModelPatcher.add_object_patch`` on the ``_forward_(pre_)hooks``
  OrderedDicts — same hook-not-override invariant as the other nodes
  (overriding ``forward`` strands weights on CPU under the cast-weights
  path).

The sidecar JSON must sit next to the safetensors with the same stem
(``foo.safetensors`` + ``foo.json``) — both files ship together.
"""

import importlib
import json
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple

import torch

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_VENDOR = _HERE / "_vendor"


def _resolve_ext_vocab():
    # Same vendor-first resolution story as adapter.py's router-compute
    # resolver (see the long comment there): the live anima_lora tree wins
    # when importable, otherwise force this node's own ``_vendor`` tree,
    # evicting any incomplete ``library`` namespace a sibling anima node
    # cached first. The trained ext rows are keyed to the exact segmentation
    # in ``ext_vocab.py`` — drift silently mis-routes prompts to wrong rows.
    try:
        return importlib.import_module("library.anima.ext_vocab")
    except ImportError:
        pass
    if _VENDOR.exists():
        if str(_VENDOR) not in sys.path:
            sys.path.insert(0, str(_VENDOR))
        for _name in [
            k
            for k in list(sys.modules)
            if k in ("library", "networks") or k.startswith(("library.", "networks."))
        ]:
            del sys.modules[_name]
        return importlib.import_module("library.anima.ext_vocab")
    return importlib.import_module("library.anima.ext_vocab")


_ev = _resolve_ext_vocab()
T5_TABLE_SIZE = _ev.T5_TABLE_SIZE
T5_EOS_ID = _ev.T5_EOS_ID
T5_UNK_ID = _ev.T5_UNK_ID

# Cache: path -> (ext table fp32 cpu, mapping dict).
_pack_cache: Dict[str, Tuple[torch.Tensor, dict]] = {}


def load_vocab_pack(path: str) -> Tuple[torch.Tensor, dict]:
    """Load ``ext_embed`` + the same-stem JSON sidecar, with caching."""
    if path in _pack_cache:
        return _pack_cache[path]
    from safetensors.torch import load_file

    tensors = load_file(path)
    if "ext_embed" not in tensors:
        raise ValueError(
            f"{os.path.basename(path)} has no 'ext_embed' tensor — not a vocab "
            "pack. (A LoRA goes in AnimaAdapterLoader, not this node.)"
        )
    table = tensors["ext_embed"].float().cpu()

    sidecar = Path(path).with_suffix(".json")
    if not sidecar.exists():
        raise FileNotFoundError(
            f"vocab pack sidecar not found: {sidecar} — the pack ships as a "
            ".safetensors + .json pair with the same stem; copy both files."
        )
    mapping = json.loads(sidecar.read_text(encoding="utf-8"))
    if int(mapping.get("rows", table.shape[0])) != table.shape[0]:
        raise ValueError(
            f"vocab pack mismatch: sidecar says {mapping['rows']} rows, "
            f"tensor has {table.shape[0]} — the .json is from a different pack."
        )
    _pack_cache[path] = (table, mapping)
    return table, mapping


def build_encoder(clip, mapping: dict):
    """HybridT5Encoder from the CLIP's own T5 tokenizer + comfy's bundled Qwen.

    The T5 side reuses ``clip.tokenizer.t5xxl.tokenizer`` (the exact
    tokenizer the native path runs, so non-CJK runs can't drift). The Qwen
    side needs offset mappings, which the slow ``Qwen2Tokenizer`` comfy
    instantiates doesn't provide — so load a fast tokenizer from the same
    bundled ``qwen25_tokenizer`` files (vocab-identical to Qwen3-0.6B's,
    verified id-for-id against the training-side encoder).
    """
    import comfy.text_encoders.anima as _cta
    from transformers import AutoTokenizer

    t5_tok = clip.tokenizer.t5xxl.tokenizer
    qwen_dir = os.path.join(
        os.path.dirname(os.path.realpath(_cta.__file__)), "qwen25_tokenizer"
    )
    qwen_tok = AutoTokenizer.from_pretrained(qwen_dir)
    if not getattr(qwen_tok, "is_fast", False):
        raise RuntimeError(
            f"fast Qwen tokenizer unavailable from {qwen_dir} — the vocab pack "
            "needs offset mappings (pip install tokenizers)."
        )
    return _ev.HybridT5Encoder.from_mapping(t5_tok, qwen_tok, mapping)


class VocabPackTokenizer:
    """Wraps ``AnimaTokenizer``; rewrites the t5xxl stream for CJK prompts.

    The qwen3_06b stream (and everything else — untokenize, decode,
    state_dict) delegates to the inner tokenizer unchanged. Only when the
    prompt contains at least one CJK character is the t5xxl stream replaced
    with the hybrid encoding, so pure-English prompts stay bit-identical.

    Prompt weighting ``(tag:1.2)`` is honored via comfy's own parser; the
    ``embedding:name`` syntax is not resolved on the rewritten t5 stream
    (t5-side textual embeddings are all but unused for Anima).
    """

    def __init__(self, inner, encoder):
        self._vp_inner = inner
        self._vp_encoder = encoder

    def tokenize_with_weights(self, text: str, return_word_ids=False, **kwargs):
        out = self._vp_inner.tokenize_with_weights(text, return_word_ids, **kwargs)
        if self._vp_routes(text):
            out["t5xxl"] = self._vp_t5_stream(text, return_word_ids)
        return out

    def _vp_routes(self, text: str) -> bool:
        # The routing rule lives in the pack json (``route``: CJK ranges plus
        # the symbol tail T5 cannot spell — ^ < ~ · × ☆, emoji). An ext_vocab
        # older than that field has no ``routes``; fall back to the legacy
        # CJK-only predicate so an old vendor tree keeps working unchanged.
        enc = self._vp_encoder
        if hasattr(enc, "routes"):
            return enc.routes(text)
        return any(_ev.is_cjk_char(c) for c in text)

    def _vp_t5_stream(self, text: str, return_word_ids: bool):
        from comfy.sd1_clip import escape_important, token_weights, unescape_important

        enc = self._vp_encoder
        pairs = []
        word_idx = 0
        for segment, weight in token_weights(escape_important(text), 1.0):
            segment = unescape_important(segment)
            for kind, span in _ev.segment_runs(
                segment, getattr(enc, "route", None)
            ):
                if kind == "cjk":
                    ids, _offs = enc._encode_cjk_words(span)
                else:
                    ids = enc.t5_tok(span, add_special_tokens=False)["input_ids"]
                for i in ids:
                    i = int(i)
                    pairs.append(
                        (i, weight, word_idx) if return_word_ids else (i, weight)
                    )
                word_idx += 1
        pairs.append(
            (T5_EOS_ID, 1.0, word_idx) if return_word_ids else (T5_EOS_ID, 1.0)
        )
        return [pairs]

    def __getattr__(self, name):
        return getattr(self._vp_inner, name)


def apply_vocab_pack(model, table: torch.Tensor) -> None:
    """Install the ext-row hooks on an already-cloned ModelPatcher.

    The table stays on CPU in fp32; only the rows a prompt actually uses are
    gathered and moved to the embed output's device/dtype (a handful of KB
    per encode — no resident VRAM cost).
    """
    diffusion_model = model.get_model_object("diffusion_model")
    if not hasattr(diffusion_model, "llm_adapter"):
        raise RuntimeError(
            "vocab pack requires diffusion_model.llm_adapter (Anima DiT) — "
            "the loaded model has no llm_adapter attribute."
        )
    adapter = diffusion_model.llm_adapter
    state: dict = {}

    def _clamp_pre_hook(module, args):
        if len(args) < 2 or not torch.is_tensor(args[1]):
            state.pop("ids", None)
            return None
        ids = args[1]
        mask = ids >= T5_TABLE_SIZE
        if not bool(mask.any()):
            state.pop("ids", None)
            return None
        state["ids"] = ids
        return (args[0], ids.masked_fill(mask, T5_UNK_ID)) + tuple(args[2:])

    def _embed_hook(module, args, output):
        ids = state.pop("ids", None)
        if ids is None:
            return None
        mask = ids >= T5_TABLE_SIZE
        rows = table[(ids[mask] - T5_TABLE_SIZE).to("cpu", torch.long)]
        out = output.clone()
        out[mask] = rows.to(device=output.device, dtype=output.dtype)
        return out

    new_pre_hooks = OrderedDict(adapter._forward_pre_hooks)
    new_pre_hooks[id(_clamp_pre_hook)] = _clamp_pre_hook
    model.add_object_patch(
        "diffusion_model.llm_adapter._forward_pre_hooks", new_pre_hooks
    )
    new_hooks = OrderedDict(adapter.embed._forward_hooks)
    new_hooks[id(_embed_hook)] = _embed_hook
    model.add_object_patch(
        "diffusion_model.llm_adapter.embed._forward_hooks", new_hooks
    )
    logger.info("vocab pack: %d ext rows hooked onto llm_adapter.embed", table.shape[0])
