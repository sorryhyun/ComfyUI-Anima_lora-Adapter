"""Per-step-expert turbo loader for ComfyUI.

Drives a turbo DP-DMD student trained with ``per_step_expert=true`` — a shared
``lora_down`` + K up-heads per Linear, head ``k`` serving denoise step ``k``
(``networks/lora_modules/step_expert.py``). There is NO router: selection is by
the denoise-step counter, which a stock LoRA loader can't supply, so this needs
a dedicated node (mirrors why Hydra/ReFT need ``AnimaAdapterLoader``).

Two pieces of per-load shared state:

  * ``step_state["idx"]`` — the active head index, advanced once per
    ``diffusion_model`` forward by a ``_forward_pre_hooks`` pre-hook. With the
    turbo contract (cfg=1.0 → one forward per denoise step) a forward-count
    modulo K maps forward i → head ``i % K`` exactly, and a fresh K-step
    denoise wraps cleanly back to head 0. **Assumption** (logged): one model
    forward per step. CFG>1 (two forwards/step) would desync — turbo runs
    cfg=1.0, and ``--cfg 1.0`` is the documented inference contract.
  * per-Linear ``forward_hook`` adds ``lora_ups[idx](lora_down(x)) * scale``.

The on-disk file keeps the training-runtime FUSED qkv/kv key layout; ComfyUI's
DiT uses SPLIT q/k/v, so the parser splits each fused group into its components
(down cloned, each head's up chunked along the output rows) before mapping to
ComfyUI's module keys — same split contract Hydra/save uses, inlined here so
the node has no cross-package dependency.

Live-hook invariant ([[project_blockcompile_rebuilds_dit_strands_hooks]]): hooks
bind to ``model.get_model_object`` / ``add_object_patch`` at apply time, so an
AnimaBlockCompile clone that rebuilds the DiT re-installs them on the live tree.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Fused-attention split spec, inlined from networks/attn_fuse.py (tiny + stable)
# so the standalone-installed node needs no cross-package import. The DiT
# runtime fuses self-attn q/k/v → qkv_proj and cross-attn k/v → kv_proj; the
# component order is load-bearing (the up rows are concatenated in this order).
_ATTN_FUSE_SPECS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("self_attn", "qkv", ("q", "k", "v")),
    ("cross_attn", "kv", ("k", "v")),
)


def _match_fused_spec(prefix: str):
    for attn_type, fused_letters, components in _ATTN_FUSE_SPECS:
        if prefix.endswith(f"{attn_type}_{fused_letters}_proj"):
            return attn_type, fused_letters, components
    return None


def _split_fused_modules(modules: Dict[str, dict]) -> Dict[str, dict]:
    """Split fused qkv/kv step-expert modules into per-component prefixes.

    ``lora_down`` is cloned per component; each head's ``lora_up`` (rows =
    concatenated component outputs) is chunked along dim 0 in component order.
    Non-fused prefixes pass through untouched.
    """
    out: Dict[str, dict] = {}
    for prefix, mod in modules.items():
        spec = _match_fused_spec(prefix)
        if spec is None:
            out[prefix] = mod
            continue
        attn_type, fused_letters, components = spec
        n = len(components)
        base = prefix[: -len(f"{attn_type}_{fused_letters}_proj")]
        down = mod["lora_down"]
        ups = mod["lora_ups"]  # {k: (out_total, r)}
        alpha = mod.get("alpha")
        inv_scale = mod.get("inv_scale")
        # Per-head up chunks: dict[k] -> list over components.
        up_chunks = {k: torch.chunk(v, n, dim=0) for k, v in ups.items()}
        for ci, letter in enumerate(components):
            comp_prefix = f"{base}{attn_type}_{letter}_proj"
            out[comp_prefix] = {
                "lora_down": down.clone(),
                "lora_ups": {k: up_chunks[k][ci].contiguous() for k in ups},
            }
            if alpha is not None:
                out[comp_prefix]["alpha"] = alpha.clone()
            if inv_scale is not None:
                out[comp_prefix]["inv_scale"] = inv_scale.clone()
    return out


def parse_step_expert(
    weights_sd: Dict[str, torch.Tensor], metadata: Dict[str, str]
) -> Optional[dict]:
    """Group per-step-expert keys. Returns None if this isn't a step-expert file.

    Discriminated by the ``ss_turbo_per_step_expert`` metadata stamp (the
    ``.lora_ups.{k}.weight`` key shape alone is ambiguous with Hydra-MoE).
    """
    if str(metadata.get("ss_turbo_per_step_expert", "")).strip() not in (
        "1",
        "true",
        "True",
    ):
        return None

    modules: Dict[str, dict] = {}
    for key, value in weights_sd.items():
        if not key.startswith("lora_unet_"):
            continue
        parts = key.split(".")
        prefix = parts[0]
        rest = ".".join(parts[1:])
        mod = modules.setdefault(prefix, {})
        if rest == "lora_down.weight":
            mod["lora_down"] = value
        elif rest.startswith("lora_ups.") and rest.endswith(".weight"):
            idx = int(rest.split(".")[1])
            mod.setdefault("lora_ups", {})[idx] = value
        elif rest == "alpha":
            mod["alpha"] = value
        elif rest == "inv_scale":
            mod["inv_scale"] = value

    modules = {p: m for p, m in modules.items() if "lora_down" in m and "lora_ups" in m}
    if not modules:
        return None

    K = int(metadata.get("ss_turbo_step_expert_K", "0") or "0")
    if K <= 1:
        K = max(max(m["lora_ups"].keys()) + 1 for m in modules.values())
    modules = _split_fused_modules(modules)
    return {"K": K, "modules": modules}


def _make_step_pre_hook(step_state: dict, K: int):
    """Pre-hook on diffusion_model: advance the per-step head index.

    Forward-count modulo K (see module docstring for the one-forward-per-step
    assumption). Dynamo-disabled — it only touches a Python dict.
    """

    @torch._dynamo.disable
    def step_pre_hook(module, args):
        idx = step_state.get("counter", 0)
        step_state["idx"] = idx % K
        step_state["counter"] = idx + 1

    return step_pre_hook


def _make_step_hook(params: dict, strength: float, step_state: dict, K: int):
    """Per-Linear forward hook: add the active step-head's LoRA delta."""
    state = {
        "lora_down": params["lora_down"],  # (r, in)
        "lora_ups": params["lora_ups"],  # {k: (out, r)}
        "inv_scale": params.get("inv_scale"),
        "scale": float(params["scale"]),
        "device": None,
    }

    def _ensure_on_device(x: torch.Tensor) -> None:
        if state["device"] == x.device:
            return
        state["lora_down"] = state["lora_down"].to(device=x.device, dtype=torch.float32)
        state["lora_ups"] = {
            k: v.to(device=x.device, dtype=torch.float32)
            for k, v in state["lora_ups"].items()
        }
        if state["inv_scale"] is not None:
            state["inv_scale"] = state["inv_scale"].to(
                device=x.device, dtype=torch.float32
            )
        state["device"] = x.device

    def step_hook(module, inputs, output):
        x = inputs[0]
        _ensure_on_device(x)
        idx = min(int(step_state.get("idx", 0)), K - 1)
        up = state["lora_ups"].get(idx)
        if up is None:
            return output
        x_lora = x.float()
        if state["inv_scale"] is not None:
            x_lora = x_lora * state["inv_scale"]
        lx = torch.nn.functional.linear(x_lora, state["lora_down"])
        delta = torch.nn.functional.linear(lx, up) * state["scale"] * strength
        return output + delta.to(output.dtype)

    return step_hook


def apply_step_expert(model, data: dict, strength: float) -> int:
    """Install the step pre-hook + per-Linear forward hooks. Returns hook count."""
    import comfy.lora

    if strength == 0:
        return 0

    from .adapter import _resolve_module  # live-or-vendor module resolver

    K = int(data["K"])
    step_state: dict = {"counter": 0, "idx": 0}

    diffusion_model = model.get_model_object("diffusion_model")
    pre_hook = _make_step_pre_hook(step_state, K)
    new_pre_hooks = OrderedDict(diffusion_model._forward_pre_hooks)
    new_pre_hooks[id(pre_hook)] = pre_hook
    model.add_object_patch("diffusion_model._forward_pre_hooks", new_pre_hooks)

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    patched = 0
    skipped: list[str] = []
    for prefix, mod in data["modules"].items():
        comfy_sd_key = key_map.get(prefix)
        if comfy_sd_key is None:
            skipped.append(f"{prefix}: not in ComfyUI key_map")
            continue
        module_path = (
            comfy_sd_key[: -len(".weight")]
            if comfy_sd_key.endswith(".weight")
            else comfy_sd_key
        )
        try:
            linear = _resolve_module(model, module_path)
        except (AttributeError, IndexError, ValueError) as e:
            skipped.append(f"{prefix}: resolve {module_path} failed ({e})")
            continue

        rank = mod["lora_down"].shape[0]
        alpha_t = mod.get("alpha")
        alpha = (
            float(alpha_t.item() if hasattr(alpha_t, "item") else alpha_t)
            if alpha_t is not None
            else float(rank)
        )
        params = {
            "lora_down": mod["lora_down"],
            "lora_ups": mod["lora_ups"],
            "inv_scale": mod.get("inv_scale"),
            "scale": alpha / rank,
        }
        hook = _make_step_hook(params, strength, step_state, K)
        new_hooks = OrderedDict(linear._forward_hooks)
        new_hooks[id(hook)] = hook
        model.add_object_patch(f"{module_path}._forward_hooks", new_hooks)
        patched += 1

    if skipped:
        logger.warning(
            f"step-expert turbo skipped {len(skipped)} prefix(es); "
            f"first few: {skipped[:5]}"
        )
    logger.info(
        f"step-expert turbo: {patched} Linears hooked, K={K} step-heads. "
        "Assuming one model forward per denoise step (cfg=1.0 turbo contract); "
        "drive with the matching --infer_steps and cfg=1.0."
    )
    return patched
