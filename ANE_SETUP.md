# ANE hybrid setup (iOS) — finishing the native CoreML path

The JS side is done and ships safe: selecting **CPU + CoreML** on the Scan screen requests the
ANE hybrid, and if the native pieces below aren't present yet it silently falls back to CPU.
These are the steps that need your Mac/Xcode/device — do them in order, validating with
**mobilefacenet** first (it maps cleanly to the Neural Engine).

## 1. Add the native module to the Xcode target  (Xcode is already open)

1. In the Project navigator, right-click the **FaceModelLab** group (the folder with
   `AppDelegate.swift`) → **Add Files to "FaceModelLab"…**
2. Select `ios/FaceModelLab/FaceCoreML.mm` → make sure **"FaceModelLab" target is checked** → Add.
3. Xcode may ask to **create a bridging header** — click **Don't Create**. (It's an RN module
   consumed by JS, not by Swift; no bridging header is needed.)

That's the whole native-code step — it registers as `NativeFaceCoreML` automatically.

## 2. Convert a model to CoreML

`coremltools` needs **Python 3.10–3.11** (your system 3.14 won't work). One-time:

```bash
/usr/bin/python3 -m venv ~/.venvs/coreml      # or any 3.10/3.11 python
source ~/.venvs/coreml/bin/activate
pip install --upgrade pip
pip install coremltools torch onnx onnx2torch
```

Then from the repo root:

```bash
python scripts/onnx-to-coreml.py mobilefacenet
```

→ produces `assets/coreml/mobilefacenet.mlpackage`.

## 3. Bundle the .mlpackage

1. Drag `assets/coreml/mobilefacenet.mlpackage` into the **FaceModelLab** group in Xcode.
2. Check the **FaceModelLab target**, and confirm it lands in **Build Phases → Copy Bundle
   Resources**. Xcode compiles `.mlpackage` → `mobilefacenet.mlmodelc` into the app bundle,
   which is what the native module loads.

The registry already points `mobilefacenet.coremlAsset → 'mobilefacenet.mlpackage'`, so nothing
else to wire.

## 4. Build, run, measure

1. `npx react-native run-ios` (or Run in Xcode) on a **real device** (the Simulator has no ANE).
2. Scan tab → **iOS inference backend: select both CPU and CoreML** (header shows "· ANE hybrid").
3. Pick **MobileFaceNet**, run a scan. If `mobilefacenet.mlmodelc` loaded, the embed stage is now
   served by CPU **and** the ANE in parallel; watch the headline /photo and faces/s.

### Did it actually use the ANE?
Open `assets/coreml/mobilefacenet.mlpackage` in Xcode → **Performance** tab → run on the device.
It reports each layer's compute unit (ANE / GPU / CPU). Plain-conv ArcFace maps to the ANE;
the linear-attention **FaceLiVT/ViT** variants may partially fall back — convert and check each
before trusting its hybrid numbers.

## Notes / next optimizations
- Tensors currently cross the bridge as `number[]` (~37k floats per face in). If profiling shows
  that marshaling dominates, the upgrade is an ArrayBuffer/JSI path in `FaceCoreML.mm` +
  `NativeFaceCoreML.ts`.
- Routing is automatic & self-balancing (`HybridEmbedder` → least-loaded backend), so you don't
  set a CPU:ANE ratio — the faster engine just pulls more.
- To force-prefer the ANE over the GPU, change `MLComputeUnitsAll` →
  `MLComputeUnitsCPUAndNeuralEngine` in `FaceCoreML.mm`.
