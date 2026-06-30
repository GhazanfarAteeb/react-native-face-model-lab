# Changelog

## 2026-06-30 — iOS ANE hybrid backend · lazy decode · cross-run thumbnail cache

---

### 1. Native CoreML / Apple Neural Engine inference backend (iOS)

#### `ios/FaceCoreML.mm` — new Objective-C++ native module

A fully new native module that runs face-embedding models on the Apple Neural Engine via the CoreML framework. Key implementation details:

- Registers as `NativeFaceCoreML` via `RCT_EXPORT_MODULE`. Compatible with both the New Architecture (surfaced through the TurboModule interop layer) and the legacy bridge.
- **Three async JS-callable methods**: `load(assetName, inputName, outputName)`, `infer(handle, input)`, `release(handle)`.
- **`load`**: looks up the compiled `<name>.mlmodelc` in the app bundle (Xcode compiles `.mlpackage` → `.mlmodelc` at build time), creates an `MLModel` with `computeUnits = MLComputeUnitsAll` so CoreML can schedule work onto the ANE, resolves input/output feature names from the model description (falls back to the first feature when empty strings are passed), stores a handle integer → model mapping. Returns `-1` (not an exception) on any failure so the JS side can fall back gracefully.
- **`infer`**: copies the pre-processed tensor (already flattened in the model's layout) into an `MLMultiArray`, runs `predictionFromFeatures:`, flattens the output `MLMultiArray` to `NSArray<NSNumber *>`. Handles both fp32 and fp16 output via the flat indexed subscript.
- **`methodQueue`**: runs on its own dedicated serial dispatch queue (`com.facemodellab.coreml`). Because this is a different queue from ORT's internal queue, CoreML inference and ORT-CPU inference can overlap on the device — that overlap is the entire point of the hybrid architecture.
- **`requiresMainQueueSetup`** returns `NO` — no main-thread initialization needed.

#### `src/runtime/NativeFaceCoreML.ts` — new JS bridge handle

Resolves `NativeFaceCoreML` from the TurboModule registry first (new-arch path), then falls back to `NativeModules` (legacy bridge path). Resolves to `null` when the native module isn't compiled into the binary, so nothing crashes before the Xcode target is wired up.

#### `src/runtime/CoreMLEmbedder.ts` — new `Embedder` implementation

Wraps `NativeFaceCoreML` behind the same `Embedder` interface used by `OnnxEmbedder`:

- `CoreMLEmbedder.create(spec)` is the factory. Returns `null` (not a thrown error) in two cases: the native module isn't available, or `spec.coremlAsset` is not defined for this model. Both conditions are checked before any async work.
- Serializes inference with a **per-instance `Mutex`** (not the shared `ortLock` — that covers ORT only). An `OnnxEmbedder` and a `CoreMLEmbedder` holding their respective locks can run simultaneously.
- Reports `fileSizeBytes` from the ONNX model file (not the `.mlpackage`) so the UI shows a consistent, already-tracked size.
- `infer` converts `Float32Array → number[]` (the codegen-supported type for the bridge), calls `NativeFaceCoreML.infer`, converts the result back to `Float32Array`. Returns `{ data, computeMs }`.

#### `src/runtime/HybridEmbedder.ts` — new self-balancing dual-queue

Runs ORT-CPU and CoreML/ANE concurrently with automatic load balancing:

- Maintains an `inflight: number[]` counter per backend. Each `infer` call routes to the backend with the **lower in-flight count** at the moment of arrival, increments that counter, and decrements it in `finally`.
- Ties favour the CPU backend (index 0) — the next concurrent call sees the bump and picks the ANE, so in practice the two backends alternate under load.
- **Self-balancing property**: the faster backend drains its queue sooner and therefore attracts a proportionally larger share of work automatically. No fixed CPU:ANE ratio is configured; if the ANE is 1.5× faster it ends up doing ~1.5× the work.
- `loadMs` is reported as `Math.max(cpu.loadMs, ane.loadMs)` — the headline reflects when both engines are actually ready, not just the first one to finish loading.
- `release` calls both backends' `release` concurrently, ignoring individual errors.

#### `src/runtime/Embedder.ts` — updated interface and factory

- `infer` return type changed from `Promise<Float32Array>` to `Promise<InferResult>` where `InferResult = { data: Float32Array; computeMs: number }`. The `computeMs` field carries pure compute time excluding any lock/queue wait, keeping stage timings comparable across backends.
- New `EmbedderOptions` interface with `iosBackends?: Array<'cpu' | 'coreml'>`.
- `createEmbedder(spec, opts?)` now dispatches to one of three paths based on platform and the requested backends:
  - **Android (any non-iOS)**: always `OnnxEmbedder` with `executionProvider: 'cpu'` — no option respected, because accelerator EPs have a SIGABRT history on this project.
  - **iOS, `['cpu', 'coreml']` (hybrid)**: attempts `CoreMLEmbedder.create`; if the native module is present and the model has a `.mlpackage`, builds a `HybridEmbedder(cpu, ane)`. If the native module is absent, degrades silently to plain CPU — no crash, no user-visible error.
  - **iOS, `['coreml']` only**: `OnnxEmbedder` with `executionProvider: 'coreml'` — benchmarks the ORT CoreML execution provider in isolation.
  - **iOS, `['cpu']` (default)**: `OnnxEmbedder` with `executionProvider: 'cpu'`.

#### `src/runtime/OnnxEmbedder.ts` — CPU-only default + shared ORT lock

- **CoreML EP is no longer in the default provider list.** Previously, iOS always passed `['coreml', 'cpu']` to ORT, but ORT's CoreML EP rarely engages the ANE for 112² models and adds ~1 s of session-compile overhead plus per-call dispatch overhead (measured: embed 212 ms with EP vs 126 ms with plain CPU on the same device). Now defaults to `['cpu']`; the CoreML EP is included only when explicitly requested via `executionProvider: 'coreml'`.
- **`ortLock` is now taken inside `OnnxEmbedder.infer`** (not in `embedFace.ts`). The lock serializes all ORT `session.run` calls (detectors + embedder) to prevent the SIGABRT crashes that have affected this project previously. The lock wait is excluded from `computeMs` — only the time between the lock being acquired and the result being ready is recorded.
- `OnnxEmbedder.create` logs which providers are being used: `'[FML] loading ONNX embedder: <id> providers: cpu ...'`.
- Returns `{ data: copy, computeMs }` instead of the raw `Float32Array`.

#### `src/runtime/ortLock.ts` — clarified ownership

Comment updated to be explicit: the `ortLock` covers **only** `onnxruntime-react-native` calls (ONNX detectors and the ORT-CPU/CoreML-EP embedder). The native `CoreMLEmbedder` uses a separate per-instance mutex and does **not** take `ortLock`, which is what allows ORT-CPU and CoreML/ANE inference to be in-flight at the same time.

---

### 2. iOS inference backend UI control

#### `src/ui/ScanScreen.tsx` — new backend selector (iOS only)

- Renders a `SectionLabel` + two `Chip` controls ("CPU" and "CoreML") visible only on iOS (`Platform.OS === 'ios'`).
- Chips are **multi-select**: either can be active independently, except that deselecting the last active chip is a no-op (the set is never left empty — one engine is always selected).
- When both are active, the section label reads **"iOS inference backend · ANE hybrid"** to make the mode visually distinct.
- **Descriptive hint text** explains each mode:
  - CPU alone: fastest for these small models.
  - CoreML alone: benchmarks the ORT CoreML EP in isolation.
  - Both: enables the ANE hybrid (parallel CPU + native CoreML), requires the native module and a converted `.mlpackage`; falls back to CPU if absent.
- `toggleBackend` implementation: filters the current array to remove the toggled value, or appends it. Guards against empty result by falling back to keeping the just-toggled value in the set.

#### `src/scan/store.tsx` — new default and wire-up

- `DEFAULT_SETTINGS` gains `iosBackends: ['cpu']` — CPU is the default because it outperforms the ORT CoreML EP for 112² models.
- `clearCache` action updated: now also calls `clearThumbs()` fire-and-forget alongside `clearScanCache()`, since thumbnails are now persisted across runs and should be wiped when the user explicitly clears the benchmark cache.

#### `src/types.ts` — `ScanSettings` extended

- `iosBackends: Array<'cpu' | 'coreml'>` added with full doc-comment explaining all three modes and the "must contain at least one entry" invariant.
- `coremlAsset?: string` added to `ModelSpec` — points to the `.mlpackage` file name in the app bundle. When absent the hybrid path skips CoreML for this model.

---

### 3. Model registry: CoreML asset declarations

#### `src/models/registry.ts`

- `coremlAsset: assetName.replace('.onnx', '.mlpackage')` added to every `faceLivt(...)` call — covers all six FaceLiVT variants (v1-S/M, v2-XS/S/M/L) automatically.
- `coremlAsset: 'mobilefacenet.mlpackage'` added to the MobileFaceNet entry.
- Both are safe before any `.mlpackage` is bundled — `CoreMLEmbedder.create` returns `null` when the asset file isn't found, and the pipeline silently stays on CPU.

---

### 4. Lazy decode — fully warm reruns skip the biggest stage

Previously, every photo in every scan unconditionally called `normalizeUri` + `downscaleForScan` (~230–320 ms/photo) before checking any cache. Now that work is deferred.

#### `src/scan/scanner.ts` — `getWork` lazy closure

For each photo in the scan loop, a `getWork: () => Promise<string>` closure is created. The closure:

1. Returns the already-decoded path immediately if decode has happened once (`decoded` variable is set).
2. On first call: runs `normalizeUri(uri, settings.maxImageDim)` → `downscaleForScan` (or passes through if PhotoKit already returned a downscaled copy), stores the result in `decoded`, and records the timing in `sink.decode`.

`getWork` is passed to `embedImage` and to `makeThumb`. On a fully-warm rerun (every detection hits, every crop hits, the thumbnail exists), nothing calls `getWork`, so the decode step is **skipped entirely**. Per-photo wall-clock time on a warm rerun collapses to preprocess + inference + cache lookups.

`clearThumbs()` is **no longer called at scan start** — it was clearing thumbnails that should survive reruns.

Cleanup is also unified: `finally` block now checks `if (decoded)` before calling `decoded.cleanup()`, rather than always calling two separate cleanup functions.

#### `src/scan/embedFace.ts` — lazy path propagated

- `embedImage` signature changed from `path: string` to `getWork: () => Promise<string>`.
- `getWork()` is called (and awaited) only at the two cache-miss points: before the ONNX detector runs (detection miss), and before `cropFace` runs (crop miss). If both hit, the body of `embedImage` never awaits decode.
- The `ortLock` import is removed — locking now lives inside `OnnxEmbedder.infer`. `embedFace.ts` calls `embedder.infer(input)` and destructures `{ data: raw, computeMs }` directly, then pushes `computeMs` to `sink.infer`.

---

### 5. Persistent thumbnails across runs

#### `src/pipeline/thumbnail.ts`

- **New `hashUri` function**: 32-bit FNV-1a hash of the photo URI string, returned as an 8-character hex string. Produces the same value for the same URI on every run, so the thumbnail filename is deterministic across reruns.
- **`makeThumb` signature changed**: was `(srcPath: string, index: number)`, now `(uri: string, getSrc: () => Promise<string>)`.
  - Destination path computed from `hashUri(uri)` → `h_<hash>.jpg` (previously `t_<index>.jpg`).
  - **Fast path**: if `await RNFS.exists(dest)` is true, return the cached thumbnail immediately — `getSrc` is never called, so the photo is not decoded.
  - **Slow path**: call `getSrc()` to get a local image path (this is what triggers decode on a cache miss), then resize and save.
- **`clearThumbs`** doc updated: no longer called per-run; called only when "Clear scan cache" is tapped.
- Net effect: a second benchmarking run over the same gallery reuses every thumbnail without any decode work for the thumbnail stage.

---

### 6. Scan time limit (throughput benchmark mode)

#### `src/types.ts` and `src/scan/store.tsx`

- `scanTimeLimitSec: number` added to `ScanSettings`. `0` means no limit (default). A positive value stops the scan after that many seconds.
- Default in `DEFAULT_SETTINGS` is `0`.

#### `src/scan/scanner.ts`

- `wall0` is recorded at scan start. A `deadline` is computed: `settings.scanTimeLimitSec > 0 ? wall0 + scanTimeLimitSec * 1000 : Infinity`.
- The `stop` callback passed to `runPool` is now: `() => (shouldCancel?.() ?? false) || now() >= deadline`.
- In-flight photos finish; the scan result then reports how many photos were actually scanned plus the faces/s throughput for that window.

#### `src/ui/ScanScreen.tsx`

- "Scan time limit" chip row added with options: Off / 1m / 2m / 5m / 7m / 10m.
- Hint text: "Throughput test: stop after this long and see how many photos were scanned (and faces/s) in the result. Pair with a large count or 'All'. 'Off' scans the whole selection."

---

### 7. Scan all (`maxPhotos: 0`) + CameraRoll pagination

#### `src/scan/gallery.ts`

`recentPhotoUris(limit)` fully rewritten:

- `limit <= 0` now means **"the whole gallery"** — pages through every asset via the CameraRoll cursor until `has_next_page` is false.
- Positive `limit` also pages (in chunks of `PAGE_SIZE = 1000`) so it works correctly on libraries where the OS caps the maximum per-request `first` parameter.
- Stops when `uris.length >= limit` to avoid over-fetching.
- Returns `uris.slice(0, limit)` for positive limits (handles the case where the last page overshoots slightly).
- Previously: a single `CameraRoll.getPhotos({ first: limit })` call — broken for `limit = 0` and unreliable on large libraries.

#### `src/ui/ScanScreen.tsx`

- "Photos to scan" chip row gains an **"All"** option (`n: 0`).
- Hint text updated: `"'All' scans your entire gallery — newest first. Large libraries take a while; the scan is resumable if interrupted."`

---

### 8. Android stability

#### `android/app/src/main/AndroidManifest.xml`

`android:largeHeap="true"` added to the `<application>` element. Provides extra heap headroom when multiple photos are in flight simultaneously, each holding decoded native bitmaps across the detect → crop pipeline stages.

#### `src/scan/deviceProfile.ts`

- **`ANDROID_MAX_CONCURRENCY = 8`** constant introduced. `resolveConcurrency` clamps the effective concurrency to this value on Android regardless of the manual setting, logging the clamp when it fires. Without this, the experimental ×10–×20 levels cause OOM crashes on Android.
- **Auto-concurrency levels raised** because decode is now lazy and per-worker in-flight memory is lower:
  - Flagship (JS bench < 25 ms): ×4 → **×6**
  - Mid-range (JS bench < 60 ms): ×3 → **×4**
  - Low-end: unchanged at ×2
- **Parallelism chip row** in `ScanScreen` expanded from `[0, 1, 2, 3, 4]` to `[0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20]`.
- Hint text updated to explain the Android cap explicitly: "On Android, levels above ×8 are capped to protect memory (it OOMs otherwise); iOS runs the full level."

---

### 9. Results screen duration display

#### `src/ui/format.ts` — new `fmtDuration`

Breaks a millisecond value into human-readable units, dropping leading zero units:

```
  199_920 ms → "3m 19s 920ms"
   45_000 ms → "45s 0ms"
      800 ms → "800ms"
```

`h` is included when nonzero; `m` is included when `h` or `m` is nonzero; `s` and `ms` are always present when the total is ≥ 1 s.

#### `src/ui/ResultsScreen.tsx`

- Big headline scan time switched from `fmtMs` to `fmtDuration` so long scans (minutes+) display readably instead of a 6-digit millisecond count.
- `numberOfLines={1}` + `adjustsFontSizeToFit` + `minimumFontScale={0.6}` added to the `Text` element so the longer duration strings scale down gracefully rather than overflowing.

---

### 10. Scripts and tooling

#### `scripts/onnx-to-coreml.py`

Converts any registry model from `.onnx` to a CoreML `.mlpackage`. Requires Python 3.10–3.11 + `coremltools`, `torch`, `onnx`, `onnx2torch`. Takes a model ID (e.g. `mobilefacenet`) and writes `assets/coreml/<id>.mlpackage`.

#### `scripts/facelivt-to-coreml.py`

FaceLiVT-specific conversion. The ViT / linear-attention architecture needs different export handling than the plain-conv MobileFaceNet. Takes a model variant ID and pulls source weights from `scripts/.facelivt_src/`.

#### `assets/coreml/facelivt_v2_s.mlpackage`

First converted CoreML model bundled in the repo for validation. Includes the CoreML model description (`model.mlmodel`) and compiled weight blob (`weights/weight.bin`).

> **Note**: `scripts/.facelivt_src` is a clone of the upstream FaceLiVT repository with its own `.git`. It is excluded from the commit — to pin it properly, use `git submodule add`.

---

### 11. Documentation

#### `ANE_SETUP.md` (new)

Step-by-step guide for completing the ANE hybrid setup that requires Xcode/device access:

1. Add `ios/FaceCoreML.mm` to the Xcode target (right-click → Add Files; no bridging header needed).
2. One-time Python venv setup for `coremltools` (requires Python 3.10–3.11; system Python 3.14 won't work).
3. Run `scripts/onnx-to-coreml.py mobilefacenet` to produce `assets/coreml/mobilefacenet.mlpackage`.
4. Drag the `.mlpackage` into Xcode → verify it appears in Build Phases → Copy Bundle Resources.
5. Run on a real device (Simulator has no ANE), select "CPU + CoreML", scan, observe embed throughput.
6. Verification: open the `.mlpackage` in Xcode's Performance tab to confirm which layers run on the ANE vs CPU.

#### `PERFORMANCE.md` (updated)

Two new sections added:

- **Lazy decode paragraph**: explains that `normalizeUri` + `downscaleForScan` is now the *biggest* stage but only runs on cache miss; on a fully warm rerun it's skipped entirely, leaving only preprocess + inference.
- **iOS inference backend setting**: explains the CPU default, why ORT's CoreML EP was slower in practice (rarely engaging the ANE for 112² models), and what "select both" enables (hybrid dual-queue, falls back to CPU until the native module is wired).

---

## 2026-06-29 — Scan performance overhaul: concurrency, cross-run cache, resume + live UI

See commit `9bb38ab`.

## Earlier

See commit history.
