#!/usr/bin/env python3
"""
facelivt-to-coreml.py — convert FaceLiVT variants straight from PyTorch to CoreML (.mlpackage).

onnx2torch chokes on FaceLiVT's RepConv stem, so for this family we skip ONNX entirely and
convert from the official PyTorch weights (the same source export_facelivt.py uses). coremltools'
torch front-end handles the linear-attention / einsum ops that the ONNX path couldn't.

    source ~/.venvs/coreml/bin/activate
    pip install coremltools torch timm einops huggingface_hub
    python scripts/facelivt-to-coreml.py v2_s        # one variant
    python scripts/facelivt-to-coreml.py --all       # all six

Output: assets/coreml/facelivt_<variant>.mlpackage  (input "data" 1×3×112×112 -> "embedding" 512).
After converting, add the .mlpackage to the Xcode target (Copy Bundle Resources) — see ANE_SETUP.md.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "assets", "coreml")
SRC = os.path.join(HERE, ".facelivt_src")
CKPT_REPO = "novendrastywn/FaceLiVT"
REPO_GIT = "https://github.com/novendrastywn/FaceLiVT"

# key -> (arch name in backbones.get_model, HF checkpoint filename, output .mlpackage name)
VARIANTS = {
    "v1_s":  ("facelivt_s",    "facelivt-s.pt",    "facelivt_v1_s.mlpackage"),
    "v1_m":  ("facelivt_m",    "facelivt-m.pt",    "facelivt_v1_m.mlpackage"),
    "v2_xs": ("facelivtv2_xs", "facelivtv2-xs.pt", "facelivt_v2_xs.mlpackage"),
    "v2_s":  ("facelivtv2_s",  "facelivtv2-s.pt",  "facelivt_v2_s.mlpackage"),
    "v2_m":  ("facelivtv2_m",  "facelivtv2-m.pt",  "facelivt_v2_m.mlpackage"),
    "v2_l":  ("facelivtv2_l",  "facelivtv2-l.pt",  "facelivt_v2_l.mlpackage"),
}


def register_custom_ops():
    """Decompose torch ops the CoreML converter doesn't ship, into primitive MIL ops."""
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    from coremltools.converters.mil.frontend.torch.torch_op_registry import (
        register_torch_op,
        _TORCH_OPS_REGISTRY,
    )
    from coremltools.converters.mil import Builder as mb

    def _maybe(name):
        # Skip if already registered (newer coremltools) to avoid a double-registration error.
        return name not in _TORCH_OPS_REGISTRY.name_to_func_mapping

    if _maybe("addcmul"):
        @register_torch_op
        def addcmul(context, node):
            # addcmul(input, t1, t2, *, value=1) = input + value * t1 * t2
            ins = _get_inputs(context, node, expected=[3, 4])
            prod = mb.mul(x=ins[1], y=ins[2])
            if len(ins) > 3:
                v = ins[3]
                val = getattr(v, "val", None)
                # value is usually the const 1 (and comes through as int32) — skip the multiply
                # then; otherwise cast the scalar to fp32 so dtypes match prod.
                if val is None or float(val) != 1.0:
                    prod = mb.mul(x=prod, y=mb.cast(x=v, dtype="fp32"))
            context.add(mb.add(x=ins[0], y=prod, name=node.name))


# Ops whose FP16 activations overflow the ~65504 half-precision ceiling in the FaceLiVT
# transformer and produce NaN/Inf embeddings on the ANE. FaceLiVT v2 uses LINEAR attention
# (no softmax), so the blowup comes from the normalization layers (batch_norm divides by
# sqrt(var+eps)) and GELU, plus the reduction/rsqrt primitives that feed them. We keep just
# these in FP32; convs and matmuls stay FP16 and continue to run on the Neural Engine, so we
# lose almost none of the ANE speedup while killing the overflow.
_FP32_KEEP_OPS = frozenset({
    "batch_norm", "layer_norm", "instance_norm", "l2_norm",
    "gelu", "erf",
    "reduce_mean", "reduce_sum", "reduce_max", "reduce_l2_norm",
    "rsqrt", "sqrt", "exp", "softmax",
})


def _mixed_fp16_precision(ct):
    """FP16 everywhere EXCEPT the overflow-prone norm/activation ops (see _FP32_KEEP_OPS).

    op_selector returns True for ops to convert to FP16; returning False leaves the op in
    FP32. If a variant still NaNs after this, add "matmul" (linear-attention accumulation)
    to _FP32_KEEP_OPS — that's the next-most-likely overflow site — and re-validate on device.
    """
    return ct.transform.FP16ComputePrecision(
        op_selector=lambda op: op.op_type not in _FP32_KEEP_OPS
    )


def convert(key, torch, ct, get_model, hf_hub_download):
    arch, ckpt_file, out_name = VARIANTS[key]
    print(f"== {key} ({arch}) ==")
    ckpt = hf_hub_download(repo_id=CKPT_REPO, filename=ckpt_file)
    net = get_model(arch, fp16=False)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    net.load_state_dict(state)
    net.eval()

    dummy = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        traced = torch.jit.trace(net, dummy)

    print(f"  PyTorch -> CoreML …")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="data", shape=(1, 3, 112, 112), dtype=float)],
        outputs=[ct.TensorType(name="embedding", dtype=float)],
        convert_to="mlprogram",
        # FULL FP32. FaceLiVT v2-S overflows FP16 on the Neural Engine and returns all-NaN
        # embeddings (verified on device: even mixed precision keeping batch_norm/gelu/matmul
        # in FP32 still NaN'd — a conv left in FP16 on the ANE blew past the ~65504 ceiling,
        # sample value -63936). FP32 has no FP16 ops, so CoreML runs it on GPU/CPU (never the
        # ANE), where overflow is impossible. Slower/warmer than ANE but CORRECT. If ANE speed
        # is needed later, _mixed_fp16_precision (below) is the starting point — but it must be
        # re-validated ON DEVICE, not just on Mac (Mac CPU is finite even when the ANE is not).
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS16,
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, out_name)
    mlmodel.save(out_path)
    print(f"  ✓ saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*", help="variant keys (e.g. v2_s); omit with --all")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    keys = list(VARIANTS) if args.all else args.variants
    if not keys:
        print("Usage: python scripts/facelivt-to-coreml.py <variant> [...] | --all")
        print("Variants:", ", ".join(VARIANTS))
        return 1

    import torch
    import coremltools as ct
    from huggingface_hub import hf_hub_download

    register_custom_ops()

    if not os.path.isdir(SRC):
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_GIT, SRC])
    sys.path.insert(0, SRC)
    from backbones import get_model  # type: ignore

    for k in keys:
        if k not in VARIANTS:
            print(f"  ! unknown variant '{k}' — known: {', '.join(VARIANTS)}")
            continue
        try:
            convert(k, torch, ct, get_model, hf_hub_download)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ FAILED {k}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
