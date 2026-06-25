#!/usr/bin/env python3
"""
export_facelivt.py — produce facelivt_v2_s.onnx from the official FaceLiVTv2-S weights.

The community ONNX (rifatrahman378/facelivt-onnx) captures an intermediate tensor
(1284-D `view_24`) instead of the trained 512-D embedding, so we export from the .pt
ourselves — single self-contained file (no external .data sidecar).

    pip install torch timm onnx huggingface_hub einops
    python scripts/export_facelivt.py
    npm run push-models    # or adb-push the resulting .onnx

Spec (from the repo's inference.py): 112×112, RGB, (x-127.5)/127.5, NCHW, input "data".
"""
import glob
import os
import subprocess
import sys

ARCH = "facelivtv2_s"            # get_model() name (underscore)
CKPT_REPO = "novendrastywn/FaceLiVT"
CKPT_FILE = "facelivtv2-s.pt"    # HF filename (hyphen)
REPO_GIT = "https://github.com/novendrastywn/FaceLiVT"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "models", "facelivt_v2_s.onnx")


def main() -> None:
    import torch
    from huggingface_hub import hf_hub_download

    src = os.path.join(HERE, ".facelivt_src")
    if not os.path.isdir(src):
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_GIT, src])
    sys.path.insert(0, src)
    from backbones import get_model  # type: ignore

    ckpt = hf_hub_download(repo_id=CKPT_REPO, filename=CKPT_FILE)
    net = get_model(ARCH, fp16=False)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    net.load_state_dict(state)
    net.eval()

    dummy = torch.randn(1, 3, 112, 112)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    def export(dynamo: bool) -> None:
        torch.onnx.export(
            net, dummy, OUT,
            input_names=["data"], output_names=["embedding"],
            dynamic_axes={"data": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=13, dynamo=dynamo,
        )

    try:
        export(dynamo=False)  # legacy exporter → weights inline
    except Exception as e:  # noqa: BLE001
        print(f"legacy export failed ({e}); retrying with dynamo=True…")
        export(dynamo=True)

    # Guarantee a single self-contained file (ORT can't find an external .data on device).
    import onnx
    onnx.save_model(onnx.load(OUT), OUT, save_as_external_data=False)
    for sc in glob.glob(OUT + "*"):
        if sc != OUT and (sc.endswith(".data") or sc.endswith(".onnx_data")):
            os.remove(sc)

    m = onnx.load(OUT, load_external_data=False)
    dim = [d.dim_value or d.dim_param for d in m.graph.output[0].type.tensor_type.shape.dim]
    kb = os.path.getsize(OUT) // 1024
    flag = "⚠️ check (looks small/external)" if kb < 3000 else "single self-contained file"
    print(f"✅ wrote {OUT}  ({kb} KB, {flag})")
    print(f"   output '{m.graph.output[0].name}' dims={dim}  (expect 512-D)")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Missing dependency: {e}\nRun: pip install torch timm onnx huggingface_hub einops")
        sys.exit(1)
