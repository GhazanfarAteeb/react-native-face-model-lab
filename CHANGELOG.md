# Changelog

## 2026-06-30 — iOS ANE hybrid backend + lazy decode + cross-run thumbnail cache

### iOS CoreML / Apple Neural Engine hybrid backend

Added a native CoreML inference path that runs on the Apple Neural Engine, alongside the existing ONNX Runtime CPU path.

- **`ios/FaceCoreML.mm`** — Objective-C++ native module wrapping the CoreML framework. Registers as `NativeFaceCoreML` on both new-arch (TurboModule) and legacy bridge.
- **`src/runtime/NativeFaceCoreML.ts`** — JS handle; resolves to `null` when the native module isn't in the binary, so the pipeline degrades to CPU with no crash.
- **`src/runtime/CoreMLEmbedder.ts`** — `Embedder` implementation backed by the native module. `create()` returns `null` when the native module or `.mlpackage` asset is absent.
- **`src/runtime/HybridEmbedder.ts`** — Runs ORT-CPU and CoreML/ANE in parallel. Each face is routed to whichever backend currently has fewer in-flight calls (least-loaded / self-balancing dual-queue). Because CoreML is a separate runtime from ORT, both can be in flight simultaneously.
- **`src/runtime/Embedder.ts`** — `createEmbedder` now accepts an `iosBackends` option (`'cpu'` / `'coreml'`). Wires CPU-only, CoreML-EP, or Hybrid depending on the selection. Android always uses CPU.
- **`src/ui/ScanScreen.tsx`** — iOS-only multi-select "inference backend" control (CPU / CoreML). The scan header shows "· ANE hybrid" when both are selected.
- **`src/models/registry.ts`** — `coremlAsset` field added for MobileFaceNet and all FaceLiVT variants, pointing to the `.mlpackage` the native module loads.
- **`assets/coreml/facelivt_v2_s.mlpackage`** — First converted CoreML model bundled for testing.
- **`scripts/onnx-to-coreml.py`** — Converts any registry model from `.onnx` to CoreML `.mlpackage`.
- **`scripts/facelivt-to-coreml.py`** — FaceLiVT-specific conversion (handles the ViT architecture).
- **`ANE_SETUP.md`** — Step-by-step guide to add `FaceCoreML.mm` to the Xcode target, convert a model, bundle the `.mlpackage`, and verify ANE usage in the CoreML Performance tab.

### ORT defaults to CPU-only on iOS (CoreML EP removed from defaults)

ORT's CoreML execution provider rarely engages the ANE for these small 112² models and adds ~1 s of session-compile time plus per-call dispatch overhead (embed: 212 ms with EP vs 126 ms with plain CPU on the same device). `OnnxEmbedder` now defaults to `['cpu']` on iOS. The CoreML EP is used only when "CoreML only" is explicitly selected as the single backend. The `ortLock` comment was clarified: the native CoreML module does **not** take the ORT lock, which is what enables true ORT-CPU ‖ ANE parallelism in the hybrid.

### Lazy decode — warm reruns skip the biggest per-photo stage

The decode + downscale step (~230–320 ms/photo, the single most expensive stage) is now deferred until something actually needs the working image — a detection cache miss, a crop cache miss, or a missing thumbnail. On a warm rerun where every detection and crop hit and thumbnails already exist, decode is never called. Per-photo cost collapses to preprocess + inference.

- **`src/scan/scanner.ts`** — Wraps decode in a `getWork` closure that memoizes after the first call.
- **`src/scan/embedFace.ts`** — Accepts `getWork: () => Promise<string>` instead of a resolved path; calls it only when the detection or crop cache misses.
- **`src/pipeline/thumbnail.ts`** — `makeThumb` now takes `getSrc` (same lazy pattern) so a thumbnail hit short-circuits before any decode.

### Persistent thumbnails across runs

Thumbnails were previously keyed by per-run index and cleared at scan start. They are now:

- Keyed by a stable FNV-1a hash of the source photo URI, so the same photo gets the same thumbnail file across reruns.
- Persisted between runs — `clearThumbs` is no longer called at scan start; it is called only when the user taps "Clear scan cache."
- Created lazily (via `getWork`) so the first encounter still doesn't double-decode the photo.

### Android stability and concurrency improvements

- **`android/app/src/main/AndroidManifest.xml`** — Added `android:largeHeap="true"` for extra heap headroom when many photos are in-flight simultaneously.
- **`src/scan/deviceProfile.ts`** — Android concurrency is now clamped to ×8 regardless of the user's manual setting, preventing OOM crashes at the experimental ×10–×20 levels. Auto-concurrency levels raised: flagship ×4 → ×6, mid-range ×3 → ×4 (safe now that decode is lazy and per-worker memory is lower).
- **`src/scan/gallery.ts`** — `recentPhotoUris` now pages through the CameraRoll in chunks of 1000 to support "scan all" on large libraries without a single unbounded allocation.

### Embedder interface changes

`Embedder.infer` now returns `{ data: Float32Array, computeMs: number }` instead of a bare `Float32Array`. The `computeMs` field is the pure compute time excluding any lock-wait, keeping stage timings comparable across backends (important for the hybrid, where ORT-CPU and CoreML/ANE have different queue depths). All call sites updated.

---

## 2026-06-29 — Scan performance overhaul: concurrency, cross-run cache, resume + live UI

See commit `9bb38ab`.

## Earlier

See commit history.
