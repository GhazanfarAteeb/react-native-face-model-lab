# FaceModelLab — scanning performance plan & how to measure it

## Why iOS ≫ Android (the rnbaby finding)

rnbaby is fast on iOS because it **doesn't use the ONNX/ML Kit/JS pipeline for scanning** —
it has a native `CoreMLFaceEmbedder` that runs Vision detection + CoreML embedding **on the
Apple Neural Engine**, fused into one native call per photo (`scanPhoto`). Its own comment:
ONNX Runtime via RN "rarely engages the Neural Engine even with the CoreML EP listed," so it
abandoned that path; CoreML-on-ANE is "~5-10× the throughput at ~0.5W." Android uses a native
`MLKitFaceEmbedder` (ML Kit + ONNX/NNAPI) — NNAPI is fragmented and usually has no usable NPU
path, so it's effectively CPU. **That asymmetry is the whole story: iOS gets the ANE, Android
doesn't.**

This lab benchmarks arbitrary ONNX models, so it can't fully adopt CoreML-per-model, but the
lessons drive the plan below.

## The phases

| Phase | What | Status |
|---|---|---|
| 1 | **Native fused detect+embed** (Vision+CoreML/ANE on iOS) as the speed ceiling | Native — lift rnbaby's modules; not buildable from here |
| 2 | **Cross-run cache**: reuse detection + aligned crops across model runs | ✅ Shipped (`scanCache.ts`) |
| 3 | **Off-thread crop/preprocess** (worklet → native) | Seam shipped; activation in `WORKLET_OFFLOAD.md` |
| 4 | **Robustness/throughput**: per-photo timeout, thermal-aware concurrency, quality gate | Timeout ✅ shipped; thermal/quality = native/next |

Plus the earlier throughput/UX layer (all shipped): concurrent worker pool, serialized-ORT
lock, device-adaptive Auto parallelism, frame-budget yielding, detection downscale,
checkpoint/resume, live grid + JS-thread health meter.

## Phase 2 in detail (the big benchmark win, shipped)

Detection and the aligned crop are **model-agnostic**:

- **detection** depends only on `(uri, detector, maxImageDim)` → reused by *every* model.
- **crop** depends on `(uri, faceIndex, inputW×H, align, detector, maxImageDim)` → reused by
  every model with the same input size + alignment (all 112²/arcface models share crops).

So when you re-scan the same gallery with another model, only **preprocess (~7ms) + inference
(~30ms)** re-run. The 337ms detect and 468ms crop are served from cache. Detection cache is
unbounded (tiny); crop cache is a bounded LRU (~400 entries).

**Decode is now lazy too.** The working image (`normalizeUri` + `downscaleForScan`, the
*biggest* stage at ~230–320ms/photo) is produced only when something actually needs it — a
detection or crop cache **miss**, or a missing thumbnail. On a fully-warm rerun nothing does,
so decode is skipped entirely: per-photo cost collapses to preprocess + inference. Thumbnails
are persisted keyed by photo uri (`thumbnail.ts`) so they're reused across runs instead of
forcing a decode; they're wiped only by **Clear scan cache**. (Photos whose crops were evicted
from the bounded LRU still decode — raise the cap or detection resolution if that bites.)

## How to measure performance in-app

Everything you need to see the wins is on-screen:

1. **Live scan screen → UI-thread health meter.** Green/amber/red + worst frame-gap in ms.
   This is the jank meter — it should stay green; if not, the JS thread is being blocked.
2. **Results → headline `durationMs` + `/photo`.** End-to-end wall time. The comparison rows
   show `/photo` and median `infer` per run.
3. **Results → "Median time per stage."** detect / crop+align / preprocess / inference — see
   exactly where time goes (this is how we found crop+align = 468ms was the bottleneck).
4. **Results → "cache reuse" line.** detect % / crop % reused. Cold run ≈ 0%; warm reruns
   approach 100%, and `durationMs` drops correspondingly.

### Cold-vs-warm protocol

1. Scan tab → **Reuse cache across runs = Off (cold)**, run model A → note `durationMs` and
   stage medians. This is the true per-model cost.
2. Switch to **On**, run model A again → cache reuse ~100%, `durationMs` collapses to roughly
   preprocess + inference. That delta is what cross-run caching saves.
3. Run models B, C… with reuse **On** → first scan of each new *input-size/align group* pays
   the crop once; the rest reuse it. Compare `/photo` and `infer` across models in the compare
   list — now you're comparing the **models**, not the shared plumbing.
4. **Results → "Clear scan cache"** resets to cold to repeat.

### Settings that move the numbers

- **Parallelism** (Auto / ×1–×6): overlaps native I/O + detection. Auto caps at ×4 to avoid
  GC stalls; lower it if the health meter dips on heavy arcface models.
- **Detection resolution** (Full / 1280 / 1600 / 2048): caps the image before detection — the
  biggest single-run speed-up; lower = faster detect.
- **Reuse cache** (On/Off): cross-run reuse vs cold benchmarking.
- **iOS inference backend** (multi-select CPU / CoreML, iOS only): defaults to **CPU**. ORT's
  CoreML EP rarely engages the ANE for these small 112² models, so it ran on CPU anyway but paid
  a ~1s session compile + per-call dispatch/partition overhead (model load 1.01s, embed 212ms on
  iOS vs 229ms / 126ms on Android CPU). CPU drops both. Select **both** to enable the **ANE
  hybrid**: ORT-CPU and a native CoreML/ANE embedder run in parallel, each pulling faces from a
  shared queue (least-loaded routing → self-balancing). Needs the native module + a converted
  `.mlpackage` — see `ANE_SETUP.md`; falls back to CPU until then.
- **Alignment** (Model tab, spec/force-arcface/force-bbox): arcface = accurate + slow crop;
  bbox = fast crop, lower separation. Per-model default already set in the registry.

## What's left (native, can't verify here)

- **Phase 1** native CoreML/ANE embedder — `BACKGROUND_SCAN.md`-style port of rnbaby's
  modules; the only path to true ANE speed and the real fix for the iOS/Android gap.
- **Phase 3** worklet/native off-thread crop — `WORKLET_OFFLOAD.md`; kills the 468ms on the
  *first* (cold) run too.
- **Phase 4** thermal-aware concurrency (needs the native `thermalState()` rnbaby polls) and a
  detection-confidence gate.
