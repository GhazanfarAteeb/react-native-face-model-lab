# FaceModelLab

A standalone React Native app for answering **"which face-recognition model is best for the rnbaby pipeline?"** — on real photos, on a real device.

It reproduces rnbaby's recognition pipeline exactly:

```
pick references (baby + parents, separately)
      │   detect (ML Kit or YuNet) → crop / ArcFace-align → preprocess → embed → L2-normalize
      ▼
scan the gallery  ──per face──►  cosine-match against each reference bucket
      ▼
ranked best matches per bucket  +  the time the scan took
```

**Detection and cropping are run once and shared**, so swapping the embedding model
(the Model screen) is the *only* thing that changes between scans. That's what makes the
time and match-quality numbers a fair, apples-to-apples comparison — your hint to "use ML
Kit to cut short the image scanning" is baked into the methodology.

## Screens

The flow mirrors rnbaby's real reference-selection + scanning UX (no auth, no signup):

1. **Model** — choose the embedding model, the **detector** (ML Kit / YuNet) and
   **alignment**. Shows each model's input size, channel order, normalization, embedding
   dim, license, and whether its weight file is present.
2. **Scan** — fill **Baby** and **Parent** reference *slots* by tapping faces in a grid of
   faces detected across your recent gallery (rnbaby's face-tile picker — not whole-photo
   picking). Set scan size + threshold, then **Start finding matches** → a live circular
   progress ring with stats (photos scanned, faces found, matches).
3. **Results** — best matches per bucket (ranked by cosine similarity, confidence %), the
   **time the scan took** (headline + per-stage medians: detect / crop / preprocess /
   inference), model load + size, and a **Compare runs** table to line up models/detectors.

## What's measured

| Metric | Meaning |
| --- | --- |
| scan duration / ms-per-photo | wall-clock for the whole scan — the "100s of pics in N s" number |
| per-stage median (detect/crop/preprocess/infer) | where the time actually goes |
| faces/sec | end-to-end throughput |
| model load + file size | startup cost and footprint |
| matches per bucket + similarity | quality — does it actually find the baby? |
| separation | mean gap between a face's baby-score and parent-score (higher = less confusable) |

## Models included

The registry (`src/models/registry.ts`) ships pre-populated. Adding a model = one entry +
its weight file.

| Model | Runtime | Input | Dim | State |
| --- | --- | --- | --- | --- |
| **MobileFaceNet** (InsightFace w600k_mbf) | ONNX | 112² RGB NCHW | 512 | ✅ bundled — rnbaby baseline |
| **SFace** (OpenCV Zoo 2021dec) | ONNX | 112² **BGR** NCHW | 128 | ✅ bundled — Apache-2.0, fetched + verified |
| **GhostFaceNet V2** | ONNX | 112² RGB NCHW | 512 | ✅ bundled — community ONNX, fetched + verified |
| **EdgeFace-S** (γ=0.5) | ONNX | 112² RGB NCHW | 512 | ✅ bundled — exported from official `.pt` + verified |
| **FaceNet** (Keras) | ONNX | 160² RGB **NHWC** | 128 | ✅ bundled (89 MB) — prewhiten, verified |
| FaceLiVT (ICIP 2025) | ONNX | 112² | 512 | placeholder — no public weights exist yet |

Detection can also be switched (Model screen): **ML Kit** (native) or **YuNet** (OpenCV Zoo
ONNX, bundled). The five **bundled** embedders run out of the box (Android auto-bundles from
`android/app/src/main/assets/`; on iOS add the `assets/models/*.onnx` files to the Xcode
target once). SFace and GhostFaceNet were downloaded and their input/output tensor specs
verified directly from the `.onnx` (`data`→`fc1` 128-D; `input`→`embedding` 512-D).

Bundling all of them is **~140 MB** of model assets (FaceNet alone is 89 MB). To slim the
app, set `bundled: false` on any registry entry you don't need and push it on demand with
`npm run push-models` instead.

> ⚠️ **Verify normalization before trusting a comparison.** The fetched models' mean/std are
> set to each model's *published* preprocessing, but a wrong mean/std silently produces
> garbage embeddings (low matches) that look like "this model is bad" when it's really
> mis-fed. Confirm against the source repo and adjust the registry entry.

## Setup

```bash
cd FaceModelLab
npm install
```

### iOS

```bash
cd ios && bundle install && bundle exec pod install && cd ..
npm run ios
```

**One-time:** the bundled `mobilefacenet.onnx` must be in the app bundle. In Xcode, drag
`assets/models/mobilefacenet.onnx` into the **FaceModelLab** target → *Copy items if needed*,
*Add to target*. (Android picks it up automatically from `android/app/src/main/assets/`.)

### Android

```bash
npm run android
```

Grant the photos permission when prompted (the app requests `READ_MEDIA_IMAGES` on
Android 13+).

## Adding / fetching more models

Non-bundled models load from the app's on-device models dir
(`<Documents>/models/<assetName>`). Drop the file there one of three ways:

- **Helper script** (simulator / debug Android): `npm run push-models`
  — copies everything in `assets/models/*.onnx` to the running app.
- **iOS device:** Finder → your iPhone → *Files* → FaceModelLab → drop the file in
  (file sharing is enabled). The app migrates files from the Documents root into `models/`.
- **adb:** `adb push model.onnx /data/local/tmp/ && adb shell run-as com.facemodellab cp /data/local/tmp/model.onnx files/models/`

Where to get the weights:

- **SFace** → OpenCV Zoo: `models/face_recognition_sface/face_recognition_sface_2021dec.onnx`
  (rename to `sface_2021dec.onnx`).
- **EdgeFace** → `otroshi/edgeface` checkpoints → export to ONNX (`torch.onnx.export`, 112²).
- **GhostFaceNet V2** → `HamadYA/GhostFaceNets` Keras weights → `tf2onnx`.
- **FaceNet** → already at `assets/models/facenet.onnx` (from rnbaby2): `npm run push-models`.

Then add a registry entry (copy an existing one and fix `input`, `channels`, `norm`,
`output.dim`).

## Notes / accuracy

- **Threshold** on the Scan screen is the cosine cut-off for counting a "match"; ranking is
  shown regardless, so you can eyeball where the right cut-off is per model. rnbaby uses
  ~0.62–0.75 for its (aligned) MobileFaceNet — other models have different optimal cut-offs.
- **Alignment** defaults to each model's preference (ArcFace-family → 5-point warp). Use the
  *ArcFace / Bbox* override on the Model screen to measure how much alignment is worth.
- **Detector** — ML Kit (native, fast, with classification) or YuNet (OpenCV Zoo ONNX,
  fixed 640², decoded + NMS in JS). Toggle on the Model screen; the Results stage timing names
  which ran. Both emit 5 landmarks into the same crop/align path, so detectors compare fairly.
- The pipeline is pure-JS for decode (jpeg-js) + the ArcFace warp, and native for detection
  (ML Kit), crop/resize (ImageEditor) and inference (ONNX Runtime) — no Skia, so the
  native dependency list stays short.
