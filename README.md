<div align="center">

# FaceModelLab 🧪

### Benchmarking Face Recognition Models on Real Mobile Devices

A standalone **React Native 0.85** lab for comparing face-detection and face-embedding models on real photos and real devices — with the same pipeline used by the rnbaby recognition workflow.

<p>
  <a href="https://github.com/GhazanfarAteeb/react-native-face-model-lab/stargazers"><img src="https://img.shields.io/github/stars/GhazanfarAteeb/react-native-face-model-lab?style=for-the-badge&logo=github" alt="GitHub stars" /></a>
  <a href="https://github.com/GhazanfarAteeb/react-native-face-model-lab"><img src="https://img.shields.io/github/languages/top/GhazanfarAteeb/react-native-face-model-lab?style=for-the-badge" alt="Top language" /></a>
  <a href="https://github.com/GhazanfarAteeb/react-native-face-model-lab/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-research-lightgrey?style=for-the-badge" alt="License" /></a>
</p>

</div>

---

## 🎯 Why FaceModelLab?

Choosing a face-recognition model for a mobile application is not just about accuracy on a benchmark dataset.

A model that looks great on paper can still be too slow, too large, or too expensive to run on a phone. FaceModelLab exists to answer a more practical question:

> **Which face-recognition pipeline gives us the best balance of speed, model size, and matching quality on an actual mobile device?**

The lab keeps **face detection, cropping, alignment, preprocessing, embedding, and matching** consistent so that different embedding models can be compared as fairly as possible.

## 🔬 Benchmark Pipeline

```text
┌─────────────────────┐
│   Reference Faces   │
│  Baby / Parent(s)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Face Detection    │
│ ML Kit / YuNet /    │
│ SCRFD / BlazeFace   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Crop + Alignment    │
│ ArcFace 5-point /   │
│ Bounding Box        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Pre-processing    │
│ RGB/BGR · NCHW/NHWC │
│ Model-specific norm │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Embedding Model  │
│ ONNX / Core ML      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ L2 Normalize +      │
│ Cosine Similarity   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Ranked Matches      │
│ + Timing Metrics    │
└─────────────────────┘
```

### Fair-comparison principle

**Detection and cropping are shared across model runs.** The embedding model is the variable being changed, which makes timing and matching comparisons much more meaningful.

---

## 📱 App Flow

### 1. Model

Choose the embedding model, detector, and alignment strategy.

The screen exposes important model metadata including:

- Input resolution
- RGB/BGR channel order
- NCHW/NHWC layout
- Normalization
- Embedding dimension
- Runtime
- License
- Model availability

### 2. Scan

Select reference faces from detected face tiles for:

- 👶 Baby
- 👨 Parent
- 👩 Parent

Then configure scan size and similarity threshold before starting the benchmark.

The scan displays live progress including:

- Photos scanned
- Faces detected
- Matches found
- Scan progress

### 3. Results

Review:

- Ranked matches per reference bucket
- Cosine similarity / confidence
- Total scan duration
- Per-stage timing
- Model load time
- Model footprint
- Faces per second
- Separation between identity buckets

### 4. Compare Runs

Compare multiple runs side-by-side to understand which model/detector combination performs best on the same workload.

---

## 🧠 Embedding Models

| Model | Runtime | Input | Embedding | Status |
| --- | --- | --- | ---: | --- |
| **MobileFaceNet** | ONNX | 112×112 RGB NCHW | 512-D | ✅ Bundled |
| **FaceNet / InceptionResnetV1** | ONNX | 160×160 RGB NHWC | 128-D | ✅ Bundled |
| **SFace** | ONNX | 112×112 BGR NCHW | 128-D | ✅ Bundled |
| **EdgeFace-S** | ONNX | 112×112 RGB NCHW | 512-D | ✅ Bundled |
| **GhostFaceNet V2** | ONNX | 112×112 RGB NCHW | 512-D | ✅ Bundled |
| **FaceLiVT v1-S / v1-M** | ONNX | 112×112 RGB NCHW | 512-D | ✅ Bundled |
| **FaceLiVT v2-XS / v2-S / v2-M / v2-L** | ONNX | 112×112 RGB NCHW | 512-D | ✅ Bundled |
| **SphereFace (BabyArt)** | Core ML | 96×112 BGR | 512-D | ✅ Bundled |
| **FaceLiVT v2-S FP16** | Core ML / ANE | 112×112 RGB | 512-D | ⚠️ Validate |
| **EdgeFace-S FP16** | Core ML / ANE | 112×112 RGB | 512-D | ⚠️ Validate |
| **MobileFaceNet FP16** | Core ML / ANE | 112×112 RGB | 512-D | ⚠️ Validate |

> ⚠️ Model normalization is critical. A wrong mean/std, channel order, or layout can produce poor embeddings and make a good model appear inaccurate. Always verify preprocessing against the model's source/export before trusting benchmark results.

---

## 👁️ Face Detectors

| Detector | Runtime | Input | Landmarks | Status |
| --- | --- | --- | ---: | --- |
| **ML Kit** | Native | Native | 5 | ✅ Recommended |
| **YuNet** | ONNX | 640×640 | 5 | ✅ Working |
| **SCRFD** | ONNX | 640×640 | 5 | ⚠️ Android crash under investigation |
| **BlazeFace** | ONNX | 128×128 | Limited | ✅ Working — BBox fallback |

### Current detector notes

- **ML Kit** is currently the preferred detector for end-to-end mobile benchmarking because detection runs natively and adds very little overhead.
- **YuNet** provides a useful ONNX comparison but introduces additional decode/NMS overhead on Android.
- **SCRFD** currently has an Android `SIGABRT` during ONNX Runtime session creation on some configurations; desktop execution is working.
- **BlazeFace** is available but currently falls back to bounding-box cropping because it does not provide the same 5-point landmark set required for ArcFace alignment.

---

## 🏆 Current Findings

### Fastest pipeline observed

**ML Kit + FaceLiVT v2-XS + ArcFace alignment**

FaceLiVT v2-XS is the smallest FaceLiVT variant currently bundled and produces a **512-dimensional embedding**. ML Kit keeps face detection overhead low because detection is handled natively.

### Other observations

- **ML Kit + FaceLiVT** → strong speed/footprint combination.
- **ML Kit + EdgeFace-S** → good quality with heavier inference than FaceLiVT-S.
- **FaceNet** → useful baseline, but substantially larger at approximately **89 MB** and uses 160×160 NHWC input with prewhitening.
- **ArcFace 5-point alignment** → consistently useful for ArcFace-family models compared with simple bounding-box crops.
- **YuNet** → viable alternative detector with additional ONNX/JS processing overhead.

> These are engineering observations from this lab, not universal model rankings. Results depend on device, gallery size, image distribution, preprocessing, thresholds, and model exports.

---

## 📊 What Is Measured?

| Metric | What it tells you |
| --- | --- |
| **Total scan duration** | Real wall-clock time for the complete scan |
| **ms / photo** | Average processing cost per image |
| **Faces / second** | End-to-end throughput |
| **Detection median** | Time spent detecting faces |
| **Crop / preprocess median** | Image preparation overhead |
| **Inference median** | Embedding-model cost |
| **Model load time** | Startup/runtime loading cost |
| **Model size** | Storage / application footprint |
| **Similarity score** | How closely a scanned face matches a reference |
| **Separation** | Gap between identity scores; higher can indicate less confusion |

---

## 🏗️ Architecture

```text
React Native 0.85
│
├── Camera Roll / Gallery
│   └── Recent photos + face tiles
│
├── Detection Layer
│   ├── ML Kit
│   ├── YuNet
│   ├── SCRFD
│   └── BlazeFace
│
├── Image Pipeline
│   ├── JPEG decode
│   ├── Face crop
│   ├── Resize
│   ├── ArcFace 5-point warp
│   └── Model-specific normalization
│
├── Inference Layer
│   ├── ONNX Runtime React Native
│   └── Core ML / Apple Neural Engine
│
├── Matching
│   ├── L2 normalization
│   └── Cosine similarity
│
└── Benchmarking
    ├── Timing
    ├── Throughput
    ├── Similarity
    └── Run comparison
```

The implementation deliberately keeps the dependency surface focused: **Camera Roll, ML Kit, Image Editor, JPEG decoding, ONNX Runtime, React Native FS, and SVG**.

---

## 🧰 Tech Stack

<div align="center">

<img src="https://skillicons.dev/icons?i=react,ts,android,apple,onnx,docker,git,github" alt="Tech stack" />

<br />

**React Native · TypeScript · ONNX Runtime · Core ML · ML Kit · OpenCV · Face Recognition · Computer Vision**

</div>

---

## 🚀 Getting Started

### Requirements

- Node.js **22.11+**
- React Native **0.85.x** toolchain
- Android Studio / Android SDK for Android development
- Xcode + CocoaPods for iOS development
- A physical device is strongly recommended for meaningful performance measurements

### Install

```bash
git clone https://github.com/GhazanfarAteeb/react-native-face-model-lab.git
cd react-native-face-model-lab
npm install
```

### Start Metro

```bash
npm start
```

### Android

```bash
npm run android
```

Android 13+ requires the photos permission requested by the app (`READ_MEDIA_IMAGES`).

### iOS

```bash
cd ios
bundle install
bundle exec pod install
cd ..
npm run ios
```

For iOS, the bundled `mobilefacenet.onnx` must be included in the **FaceModelLab** Xcode target.

---

## 📦 Model Management

Models can be bundled with the application or loaded at runtime.

### Bundled models

Place weights under:

```text
assets/models/<model>.onnx
```

and register the model in:

```text
src/models/registry.ts
```

### Runtime models

Non-bundled models are loaded from:

```text
<DocumentDirectory>/models/<assetName>
```

You can push models to a debug installation with:

```bash
npm run push-models
```

For Android, `adb` can also be used with the app's internal files directory.

---

## ➕ Adding a New Model

Adding a benchmark candidate is intentionally registry-driven.

1. Add the model weight to `assets/models/` or make it available at runtime.
2. Add one `ModelSpec` entry in `src/models/registry.ts`.
3. Define the exact:
   - Input width/height
   - RGB/BGR channel order
   - NCHW/NHWC layout
   - Mean/std or prewhitening
   - Input/output tensor names
   - Embedding dimension
   - Alignment strategy
   - Runtime and license
4. Run the app and benchmark it against the same reference/gallery set.

Example shape:

```ts
{
  id: 'my-model',
  label: 'My Model',
  family: 'MyModel',
  runtime: 'onnx',
  assetName: 'my_model.onnx',
  bundled: true,
  input: {
    width: 112,
    height: 112,
    layout: 'NCHW',
    channels: 'RGB',
  },
  norm: {
    mean: 127.5,
    std: 127.5,
  },
  output: {
    dim: 512,
    l2normalized: false,
    inputName: 'input',
    outputName: 'embedding',
  },
  align: 'arcface',
  enabled: true,
}
```

**Do not benchmark a model until its preprocessing contract has been verified.**

---

## ⚖️ Licensing & Model Weights

The application code and the individual model weights may have **different licensing terms**.

Some bundled models are research-only, proprietary, or require separate verification for commercial use. Always review the original model repository/license before distributing a model or using it in a commercial product.

In particular, pay attention to the licenses listed in the model registry before shipping any model outside a research/benchmarking environment.

---

## ⚠️ Benchmarking Notes

FaceModelLab is an **engineering benchmark and research tool**, not a clinical or biometric identity-verification product.

For meaningful comparisons:

- Use the same gallery for every run.
- Use the same reference faces.
- Keep detector and alignment settings fixed when comparing embedding models.
- Record the device and OS version.
- Warm up the runtime before comparing steady-state inference.
- Verify model preprocessing and export correctness.
- Do not compare raw similarity thresholds across different embedding models without calibration.
- Treat accuracy results as device/data-dependent rather than universal model rankings.

---

## 🗺️ Roadmap

- [x] Multi-model embedding registry
- [x] ML Kit face detection
- [x] YuNet detector
- [x] BlazeFace detector
- [x] ArcFace and BBox alignment
- [x] Live scan metrics
- [x] Model timing and footprint metrics
- [x] Run comparison
- [x] Core ML / ANE model experiments
- [ ] Resolve SCRFD Android runtime crash
- [ ] Complete BlazeFace benchmark pass
- [ ] Complete MobileFaceNet / SFace / GhostFaceNet comparison matrix
- [ ] Automated benchmark export
- [ ] Reproducible benchmark reports across multiple devices

---

<div align="center">

## Built for real-device experimentation 🧪📱

**FaceModelLab turns face-recognition model selection into a measurable mobile engineering problem.**

<a href="https://github.com/GhazanfarAteeb">Ghazanfar Ateeb</a> ·
<a href="https://github.com/GhazanfarAteeb/react-native-face-model-lab">Repository</a>

</div>
