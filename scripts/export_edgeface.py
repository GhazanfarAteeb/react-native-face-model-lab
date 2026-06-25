#!/usr/bin/env python3
"""
export_edgeface.py — produce edgeface_s_gamma_05.onnx from the official PyTorch weights.

EdgeFace publishes only PyTorch checkpoints (no ONNX anywhere), so this one model needs
a one-time export. Run it on any machine with PyTorch (Python 3.9–3.12 recommended; torch
may not have wheels for very new Python yet).

    pip install torch timm onnx huggingface_hub
    python scripts/export_edgeface.py
    npm run push-models        # then push the resulting .onnx to the device

Output: assets/models/edgeface_s_gamma_05.onnx  (input [1,3,112,112] → 512-D embedding)

Other variants: change VARIANT to 'edgeface_xs_gamma_06' or 'edgeface_base' and update the
matching registry entry's assetName.
"""
import glob
import os
import subprocess
import sys

VARIANT = "edgeface_s_gamma_05"
REPO = "Idiap/EdgeFace-S-GAMMA"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "models", f"{VARIANT}.onnx")


def main() -> None:
    import torch
    from huggingface_hub import hf_hub_download

    # EdgeFace's model definition lives in the otroshi/edgeface repo. Clone it for the
    # backbone factory (get_model), which builds the timm-based architecture.
    edgeface_dir = os.path.join(HERE, ".edgeface_src")
    if not os.path.isdir(edgeface_dir):
        subprocess.check_call(
            ["git", "clone", "--depth", "1", "https://github.com/otroshi/edgeface", edgeface_dir]
        )
    sys.path.insert(0, edgeface_dir)
    from backbones import get_model  # type: ignore

    ckpt = hf_hub_download(repo_id=REPO, filename=f"{VARIANT}.pt")
    model = get_model(VARIANT)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, 112, 112)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        OUT,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=13,
        dynamo=False,  # legacy exporter → weights inline (no external .data sidecar)
    )

    # Belt-and-suspenders: re-save FULLY EMBEDDED. ORT cannot find an external .data
    # sidecar once the model is copied to the device, so the file MUST be self-contained.
    import onnx

    onnx.save_model(onnx.load(OUT), OUT, save_as_external_data=False)
    for sidecar in glob.glob(OUT + "*"):
        if sidecar != OUT and (sidecar.endswith(".data") or sidecar.endswith(".onnx_data")):
            os.remove(sidecar)

    size_kb = os.path.getsize(OUT) // 1024
    if size_kb < 5000:
        print(f"⚠️  {OUT} is only {size_kb} KB — weights may still be external. Check for a .data file.")
    else:
        print(f"✅ wrote {OUT}  ({size_kb} KB, single self-contained file)")
    print("Registry expects: input 'input', output 'embedding', 112², RGB, (x-0.5)/0.5.")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Missing dependency: {e}\nRun: pip install torch timm onnx huggingface_hub")
        sys.exit(1)
