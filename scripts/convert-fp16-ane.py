#!/usr/bin/env python3
"""
convert-fp16-ane.py — convert face models to FP16 CoreML for ANE-only use.

Converts FaceLiVT v2-S and EdgeFace-S from official .pt checkpoints (no ONNX intermediary),
and MobileFaceNet from ONNX (simple CNN, converts cleanly).

    source ~/.venvs/coreml/bin/activate
    pip install coremltools torch timm einops huggingface_hub onnx2torch

    python scripts/convert-fp16-ane.py                    # convert all
    python scripts/convert-fp16-ane.py facelivt            # FaceLiVT only
    python scripts/convert-fp16-ane.py edgeface            # EdgeFace only
    python scripts/convert-fp16-ane.py mobilefacenet       # MobileFaceNet only

Output: assets/coreml/<name>_fp16.mlpackage
"""
import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "assets", "coreml")

# ── FaceLiVT paths ───────────────────────────────────────────────────────────
FACELIVT_SRC = os.path.join(REPO, "scripts", ".facelivt_src")
FACELIVT_REPO = "https://github.com/novendrastywn/FaceLiVT"
FACELIVT_CKPT_REPO = "novendrastywn/FaceLiVT"

# ── EdgeFace paths ───────────────────────────────────────────────────────────
EDGEFACE_SRC = os.path.join(REPO, "scripts", ".edgeface_src")
EDGEFACE_REPO = "https://github.com/otroshi/edgeface"
EDGEFACE_CKPT_REPO = "Idiap/EdgeFace-S-GAMMA"


def _register_facelivt_ops():
    """Register custom MIL ops the CoreML converter doesn't ship (addcmul)."""
    try:
        from coremltools.converters.mil.frontend.torch.ops import _get_inputs
        from coremltools.converters.mil.frontend.torch.torch_op_registry import (
            register_torch_op,
            _TORCH_OPS_REGISTRY,
        )
        from coremltools.converters.mil import Builder as mb

        def _maybe(name):
            return name not in _TORCH_OPS_REGISTRY.name_to_func_mapping

        if _maybe("addcmul"):
            @register_torch_op
            def addcmul(context, node):
                ins = _get_inputs(context, node, expected=[3, 4])
                prod = mb.mul(x=ins[1], y=ins[2])
                if len(ins) > 3:
                    v = ins[3]
                    val = getattr(v, "val", None)
                    if val is None or float(val) != 1.0:
                        prod = mb.mul(x=prod, y=mb.cast(x=v, dtype="fp32"))
                context.add(mb.add(x=ins[0], y=prod, name=node.name))
    except ImportError:
        pass


def convert_facelivt_v2_s():
    import torch
    import coremltools as ct
    from huggingface_hub import hf_hub_download

    # Ensure the FaceLiVT source is cloned for the model definition.
    if not os.path.isdir(FACELIVT_SRC):
        subprocess.check_call(["git", "clone", "--depth", "1", FACELIVT_REPO, FACELIVT_SRC])
    sys.path.insert(0, FACELIVT_SRC)
    from backbones import get_model  # type: ignore

    _register_facelivt_ops()

    print("[facelivt_v2_s_fp16] Loading from HuggingFace .pt checkpoint...")
    ckpt_file = "facelivtv2-s.pt"
    ckpt = hf_hub_download(repo_id=FACELIVT_CKPT_REPO, filename=ckpt_file)
    net = get_model("facelivtv2_s", fp16=False)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    net.load_state_dict(state)
    net.eval()

    dummy = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        traced = torch.jit.trace(net, dummy)

    print("[facelivt_v2_s_fp16] Converting to CoreML FP16...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="data", shape=(1, 3, 112, 112), dtype=float)],
        outputs=[ct.TensorType(name="embedding", dtype=float)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS16,
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "facelivt_v2_s_fp16.mlpackage")
    mlmodel.save(out_path)
    print(f"[facelivt_v2_s_fp16] Saved {out_path}")


def convert_edgeface_s():
    import torch
    import coremltools as ct
    import numpy as np
    from huggingface_hub import hf_hub_download

    # Ensure the EdgeFace source is cloned for the model definition.
    if not os.path.isdir(EDGEFACE_SRC):
        subprocess.check_call(["git", "clone", "--depth", "1", EDGEFACE_REPO, EDGEFACE_SRC])
    sys.path.insert(0, EDGEFACE_SRC)
    from backbones import get_model  # type: ignore

    print("[edgeface_s_fp16] Loading from HuggingFace .pt checkpoint...")
    ckpt_file = "edgeface_s_gamma_05.pt"
    ckpt = hf_hub_download(repo_id=EDGEFACE_CKPT_REPO, filename=ckpt_file)
    model = get_model("edgeface_s_gamma_05")
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    # EdgeFace uses dynamic Clip ops that onnx2torch can't handle directly.
    # Workaround: export to ONNX, fix Clip inputs, then convert via onnx2torch.
    import onnx
    from onnx import helper, numpy_helper
    from onnx2torch import convert as onnx2torch_convert

    tmp_onnx = os.path.join(OUT_DIR, "_edgeface_tmp.onnx")
    dummy = torch.randn(1, 3, 112, 112)
    print("[edgeface_s_fp16] Exporting to ONNX (opset 13)...")
    torch.onnx.export(
        model, dummy, tmp_onnx,
        input_names=["input"], output_names=["embedding"],
        opset_version=13, dynamo=False,
    )

    # Fix Clip ops with empty min/max inputs by adding initializer constants.
    onnx_model = onnx.load(tmp_onnx)
    fixed = 0
    for node in onnx_model.graph.node:
        if node.op_type == "Clip":
            while len(node.input) < 3:
                node.input.append("")
            for idx, label in [(1, "min"), (2, "max")]:
                if node.input[idx] == "":
                    init_name = node.name + "_" + label
                    val = -65504.0 if label == "min" else 65504.0
                    init = numpy_helper.from_array(
                        np.array([val], dtype=np.float32), name=init_name
                    )
                    onnx_model.graph.initializer.append(init)
                    node.input[idx] = init_name
                    fixed += 1
    if fixed:
        onnx.save(onnx_model, tmp_onnx)
        print(f"[edgeface_s_fp16] Fixed {fixed} Clip inputs via initializers")

    print("[edgeface_s_fp16] ONNX -> PyTorch (onnx2torch)...")
    torch_model = onnx2torch_convert(tmp_onnx).eval()

    # Patch coremltools' int op handler (EdgeFace produces dynamic int values).
    import coremltools.converters.mil.frontend.torch.ops as torch_ops
    from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY

    def _patched_int(context, node):
        ins = torch_ops._get_inputs(context, node, expected=[1])
        x = ins[0]
        val = getattr(x, "val", None)
        if val is not None:
            try:
                context.add(torch_ops.mb.const(
                    val=np.array(int(val), dtype=np.int32), name=node.name
                ))
                return
            except (TypeError, ValueError):
                pass
        context.add(torch_ops.mb.cast(x=x, dtype="int32", name=node.name))

    _TORCH_OPS_REGISTRY.name_to_func_mapping["int"] = _patched_int

    print("[edgeface_s_fp16] PyTorch -> CoreML FP16...")
    traced = torch.jit.trace(torch_model, dummy)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, 3, 112, 112), dtype=float)],
        outputs=[ct.TensorType(name="embedding", dtype=float)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS16,
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "edgeface_s_fp16.mlpackage")
    mlmodel.save(out_path)

    # Clean up temp ONNX.
    try:
        os.remove(tmp_onnx)
    except OSError:
        pass

    print(f"[edgeface_s_fp16] Saved {out_path}")


def convert_mobilefacenet():
    import torch
    import coremltools as ct
    from onnx2torch import convert as onnx2torch_convert

    onnx_path = os.path.join(REPO, "assets", "models", "mobilefacenet.onnx")
    if not os.path.exists(onnx_path):
        print(f"[mobilefacenet_fp16] {onnx_path} not found — skipping")
        return

    print("[mobilefacenet_fp16] ONNX -> PyTorch...")
    torch_model = onnx2torch_convert(onnx_path).eval()
    dummy = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        traced = torch.jit.trace(torch_model, dummy)

    print("[mobilefacenet_fp16] PyTorch -> CoreML FP16...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input.1", shape=(1, 3, 112, 112), dtype=float)],
        outputs=[ct.TensorType(name="516", dtype=float)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS16,
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "mobilefacenet_fp16.mlpackage")
    mlmodel.save(out_path)
    print(f"[mobilefacenet_fp16] Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?",
                    choices=["facelivt", "edgeface", "mobilefacenet", "all"],
                    default="all",
                    help="which model to convert (default: all)")
    args = ap.parse_args()

    if args.model in ("facelivt", "all"):
        try:
            convert_facelivt_v2_s()
        except Exception as e:
            print(f"[facelivt_v2_s_fp16] FAILED: {e}")

    if args.model in ("edgeface", "all"):
        try:
            convert_edgeface_s()
        except Exception as e:
            print(f"[edgeface_s_fp16] FAILED: {e}")

    if args.model in ("mobilefacenet", "all"):
        try:
            convert_mobilefacenet()
        except Exception as e:
            print(f"[mobilefacenet_fp16] FAILED: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
