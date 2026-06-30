#!/usr/bin/env python3
"""
onnx-to-coreml.py — convert a bundled ONNX face model to a CoreML .mlpackage for the ANE.

coremltools dropped its ONNX front-end, so we go ONNX -> PyTorch (onnx2torch) -> CoreML.
The input is kept as a plain float tensor with the SAME name/shape the app feeds, so the
embeddings match the ORT-CPU path (the hybrid pipeline mixes both, so they must agree).

  ⚠️ coremltools needs Python 3.10–3.11 (NOT 3.14). Use an isolated venv:

      /usr/bin/python3 -m venv ~/.venvs/coreml        # or any 3.10/3.11 interpreter
      source ~/.venvs/coreml/bin/activate
      pip install --upgrade pip
      pip install coremltools torch onnx onnx2torch

  Then, from the repo root:

      python scripts/onnx-to-coreml.py mobilefacenet      # start with this one (ANE-friendly)
      python scripts/onnx-to-coreml.py facelivt_v2_s
      python scripts/onnx-to-coreml.py --all

Output: assets/coreml/<name>.mlpackage  (then add it to the Xcode target — see ANE_SETUP.md).

  ⚠️ Not every op maps to the Neural Engine. After bundling, open the .mlpackage in Xcode and
  run the Core ML "Performance" report on a device: it shows which layers run on ANE vs CPU/GPU.
  Plain-conv ArcFace models (mobilefacenet) map cleanly; the linear-attention FaceLiVT/ViT
  variants may partially fall back — convert and measure before trusting them.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_DIR = os.path.join(REPO, "assets", "models")
OUT_DIR = os.path.join(REPO, "assets", "coreml")

# (onnx file, input feature name, output feature name, input shape) — mirrors src/models/registry.ts.
MODELS = {
    "mobilefacenet": ("mobilefacenet.onnx", "input.1", "516", (1, 3, 112, 112)),
    "facelivt_v1_s": ("facelivt_v1_s.onnx", "data", "embedding", (1, 3, 112, 112)),
    "facelivt_v1_m": ("facelivt_v1_m.onnx", "data", "embedding", (1, 3, 112, 112)),
    "facelivt_v2_xs": ("facelivt_v2_xs.onnx", "data", "embedding", (1, 3, 112, 112)),
    "facelivt_v2_s": ("facelivt_v2_s.onnx", "data", "embedding", (1, 3, 112, 112)),
    "facelivt_v2_m": ("facelivt_v2_m.onnx", "data", "embedding", (1, 3, 112, 112)),
    "facelivt_v2_l": ("facelivt_v2_l.onnx", "data", "embedding", (1, 3, 112, 112)),
    "edgeface_s": ("edgeface_s_gamma_05.onnx", "input", "embedding", (1, 3, 112, 112)),
    "ghostfacenet_v2": ("ghostfacenet_v2.onnx", "input", "embedding", (1, 3, 112, 112)),
    "sface": ("sface_2021dec.onnx", "data", "fc1", (1, 3, 112, 112)),
    "facenet": ("facenet.onnx", "image_input", "Bottleneck_BatchNorm", (1, 160, 160, 3)),
}


def convert(key: str) -> None:
    import torch
    import coremltools as ct
    from onnx2torch import convert as onnx2torch_convert

    onnx_name, in_name, out_name, shape = MODELS[key]
    onnx_path = os.path.join(ONNX_DIR, onnx_name)
    if not os.path.exists(onnx_path):
        print(f"  ! {onnx_path} not found — skipping {key}")
        return

    print(f"[{key}] ONNX -> PyTorch …")
    torch_model = onnx2torch_convert(onnx_path).eval()
    dummy = torch.rand(*shape, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(torch_model, dummy)

    print(f"[{key}] PyTorch -> CoreML (.mlpackage) …")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=in_name, shape=shape, dtype=float)],
        outputs=[ct.TensorType(name=out_name, dtype=float)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,  # ANE runs fp16
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS16,
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{key}.mlpackage")
    mlmodel.save(out_path)
    print(f"[{key}] ✓ saved {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="model keys (see MODELS); omit with --all")
    ap.add_argument("--all", action="store_true", help="convert every known model")
    args = ap.parse_args()

    keys = list(MODELS) if args.all else args.models
    if not keys:
        print("Usage: python scripts/onnx-to-coreml.py <model_key> [...]  |  --all")
        print("Known:", ", ".join(MODELS))
        return 1
    for k in keys:
        if k not in MODELS:
            print(f"  ! unknown model '{k}' — known: {', '.join(MODELS)}")
            continue
        convert(k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
