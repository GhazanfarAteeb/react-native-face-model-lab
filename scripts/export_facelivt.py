#!/usr/bin/env python3
"""
export_facelivt.py — export every FaceLiVT variant to a single-file ONNX.

The community ONNX (rifatrahman378/facelivt-onnx) captures a 1284-D intermediate
(`view_24`) instead of the trained 512-D embedding, so we export from the official
.pt checkpoints (novendrastywn/FaceLiVT) ourselves — each a single self-contained
file (no external .data sidecar).

    pip install torch timm onnx huggingface_hub einops
    python scripts/export_facelivt.py            # all variants
    python scripts/export_facelivt.py v2_s v2_l  # just some (by key)
    npm run push-models                          # or adb-push the .onnx files

Spec (repo inference.py): 112×112, RGB, (x-127.5)/127.5, NCHW, input "data" -> "embedding".
"""
import glob
import os
import subprocess
import sys

# key -> (get_model arch name, HF checkpoint filename, output onnx name)
VARIANTS = {
    "v1_s":  ("facelivt_s",    "facelivt-s.pt",    "facelivt_v1_s.onnx"),
    "v1_m":  ("facelivt_m",    "facelivt-m.pt",    "facelivt_v1_m.onnx"),
    "v2_xs": ("facelivtv2_xs", "facelivtv2-xs.pt", "facelivt_v2_xs.onnx"),
    "v2_s":  ("facelivtv2_s",  "facelivtv2-s.pt",  "facelivt_v2_s.onnx"),
    "v2_m":  ("facelivtv2_m",  "facelivtv2-m.pt",  "facelivt_v2_m.onnx"),
    "v2_l":  ("facelivtv2_l",  "facelivtv2-l.pt",  "facelivt_v2_l.onnx"),
}
CKPT_REPO = "novendrastywn/FaceLiVT"
REPO_GIT = "https://github.com/novendrastywn/FaceLiVT"
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "assets", "models")


def export_one(get_model, torch, onnx, hf_hub_download, arch, ckpt_file, out_name):
    out = os.path.join(MODELS, out_name)
    ckpt = hf_hub_download(repo_id=CKPT_REPO, filename=ckpt_file)
    net = get_model(arch, fp16=False)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    net.load_state_dict(state)
    net.eval()
    dummy = torch.randn(1, 3, 112, 112)

    def do(dynamo):
        torch.onnx.export(
            net, dummy, out,
            input_names=["data"], output_names=["embedding"],
            dynamic_axes={"data": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=13, dynamo=dynamo,
        )

    try:
        do(False)
    except Exception as e:  # noqa: BLE001
        print(f"  legacy export failed ({e}); retrying dynamo…")
        do(True)

    onnx.save_model(onnx.load(out), out, save_as_external_data=False)
    for sc in glob.glob(out + "*"):
        if sc != out and (sc.endswith(".data") or sc.endswith(".onnx_data")):
            os.remove(sc)
    m = onnx.load(out, load_external_data=False)
    dim = [d.dim_value or d.dim_param for d in m.graph.output[0].type.tensor_type.shape.dim]
    kb = os.path.getsize(out) // 1024
    ok = "✅" if (kb >= 3000 and 512 in [d for d in dim if isinstance(d, int)]) else "⚠️"
    print(f"  {ok} {out_name}  ({kb} KB)  out '{m.graph.output[0].name}' dims={dim}")


def main():
    import torch
    import onnx
    from huggingface_hub import hf_hub_download

    keys = sys.argv[1:] or list(VARIANTS.keys())
    src = os.path.join(HERE, ".facelivt_src")
    if not os.path.isdir(src):
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_GIT, src])
    sys.path.insert(0, src)
    from backbones import get_model  # type: ignore

    os.makedirs(MODELS, exist_ok=True)
    for key in keys:
        if key not in VARIANTS:
            print(f"skip unknown variant '{key}' (have: {', '.join(VARIANTS)})")
            continue
        arch, ckpt_file, out_name = VARIANTS[key]
        print(f"== {key} ({arch}) ==")
        try:
            export_one(get_model, torch, onnx, hf_hub_download, arch, ckpt_file, out_name)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ FAILED {key}: {e}")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Missing dependency: {e}\nRun: pip install torch timm onnx huggingface_hub einops")
        sys.exit(1)
