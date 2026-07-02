/**
 * Core types for FaceModelLab.
 *
 * The app reproduces the rnbaby recognition pipeline as a model-comparison tool:
 *
 *   pick references (baby + parents, separately)
 *        │  detect → crop/align → preprocess → embed → L2-normalize
 *        ▼
 *   scan the gallery  ── per face ──►  cosine-match against each reference bucket
 *        ▼
 *   ranked best matches per bucket, with the time the scan took
 *
 * Detection + cropping are identical across models, so swapping the embedding model
 * (Model screen) is the only variable — that's what makes the time/quality numbers a
 * fair comparison.
 */

export type RuntimeKind = 'onnx';
export type TensorLayout = 'NCHW' | 'NHWC';
export type ChannelOrder = 'RGB' | 'BGR';
export type AlignMode = 'arcface' | 'bbox';

/** out = (pixel - mean) / std. Scalar applies to all channels; tuple is per-channel
 *  in the spec's `channels` order. */
export interface Normalization {
  mean: number | [number, number, number];
  std: number | [number, number, number];
  /** FaceNet-style per-image standardization: out = (x − mean(x)) / max(std(x), 1/√N),
   *  computed over the whole crop. When true, `mean`/`std` above are ignored. */
  prewhiten?: boolean;
}

/** A selectable embedding model. Adding one = a registry entry + its weight file. */
export interface ModelSpec {
  id: string;
  label: string;
  family: string;
  runtime: RuntimeKind;
  assetName: string;
  /** iOS ANE: name of the CoreML model in the app bundle (a `.mlpackage`/`.mlmodelc`,
   *  converted offline from the ONNX file). Present → this model can run on the Neural Engine
   *  via the native CoreML embedder (hybrid mode). Absent → CPU/CoreML-EP only. */
  coremlAsset?: string;
  /** True when shipped in the app bundle (loaded via require); else from <DocumentDir>/models. */
  bundled: boolean;
  input: { width: number; height: number; layout: TensorLayout; channels: ChannelOrder };
  norm: Normalization;
  output: { dim: number; l2normalized: boolean; inputName?: string; outputName?: string };
  align: AlignMode;
  license: string;
  notes?: string;
  downloadUrl?: string;
  /** Recommended/ready hint shown in the Model screen. */
  enabled: boolean;
}

export interface FaceBox {
  left: number;
  top: number;
  width: number;
  height: number;
}
export interface Point2D {
  x: number;
  y: number;
}
export interface Landmarks5 {
  leftEye?: Point2D;
  rightEye?: Point2D;
  noseBase?: Point2D;
  mouthLeft?: Point2D;
  mouthRight?: Point2D;
}
export interface DetectedFace {
  box: FaceBox;
  landmarks: Landmarks5;
}
export interface DetectResult {
  faces: DetectedFace[];
  imageWidth: number;
  imageHeight: number;
}
export interface RgbaImage {
  data: Uint8Array;
  width: number;
  height: number;
}

// ─── References ──────────────────────────────────────────────────────────────

export type RefBucket = 'baby' | 'parent';

/** A reference photo the user picked for a bucket. `faceCount` is filled after we run
 *  detection on it so the UI can warn when a picked photo has no usable face. */
export interface ReferenceImage {
  uri: string;
  bucket: RefBucket;
  faceCount: number | null; // null = not analyzed yet
}

/** One reference face embedding (model-specific; recomputed when the model changes). */
export interface ReferenceEmbedding {
  bucket: RefBucket;
  uri: string;
  embedding: Float32Array;
}

// ─── Scanning ────────────────────────────────────────────────────────────────

export interface ScanSettings {
  /** How many gallery photos to scan (most recent first). 0 = the whole gallery. */
  maxPhotos: number;
  /** Stop the scan after this many seconds (0 = no time limit). A throughput benchmark: pair it
   *  with a large/All photo count and read how many photos were scanned + faces/s in the result. */
  scanTimeLimitSec: number;
  /** Cosine score at/above which a face counts as a "match" for a bucket. */
  threshold: number;
  /** 'spec' uses the model's preferred crop; force-* overrides it. */
  align: 'spec' | 'force-arcface' | 'force-bbox';
  minFaceSize: number;
  /** Face detector: ML Kit (native), or an ONNX detector (YuNet / SCRFD / BlazeFace). */
  detector: 'mlkit' | 'yunet' | 'scrfd' | 'blazeface';
  /** How many photos to process concurrently. 0 = Auto (device-adaptive, see
   *  deviceProfile); 1 = the old sequential behaviour; higher overlaps native I/O +
   *  detection across photos (ORT inference stays serialized). */
  concurrency: number;
  /** Cap each photo to this many px on the long side before detection (0 = full res).
   *  Large gallery photos are downscaled uniformly so detection is much faster without
   *  losing real-world face detail. Applies equally to all models (fair comparison). */
  maxImageDim: number;
  /** Reuse detection + aligned crops across runs (see scanCache). Massively speeds up
   *  re-scanning the same gallery with different models; turn off to benchmark cold cost. */
  reuseCache: boolean;
  /** ML Kit only: use the faster (lower-recall) detector instead of the accurate one. */
  fastDetect: boolean;
  /** iOS only: which inference backend(s) to run the embedder on. No effect on Android
   *  (always CPU — accelerator EPs have a SIGABRT history here).
   *   • ['cpu']            → ORT CPU (default; fastest single path for these small models).
   *   • ['coreml']         → ORT CoreML execution provider (benchmark the EP in isolation;
   *                          note it rarely engages the ANE for 112² models).
   *   • ['cpu','coreml']   → HYBRID: ORT-CPU and the native CoreML/ANE embedder run in
   *                          parallel, each pulling faces from a shared queue (work-stealing).
   *                          Requires the native CoreML module; falls back to CPU if absent.
   *  Must contain at least one entry. */
  iosBackends: Array<'cpu' | 'coreml'>;
  /** Android only: which ONNX Runtime execution provider to run the detector + embedder on.
   *  No effect on iOS (see iosBackends there).
   *   • 'cpu'   → ORT CPU (default; always stable).
   *   • 'nnapi' → the NNAPI accelerator EP (device GPU / NPU / DSP), with a CPU fallback per
   *               session. Faster on some devices, but accelerator EPs have a SIGABRT history
   *               on this project — if a model can't load on NNAPI it falls back to CPU, but a
   *               hard native abort during session creation is uncatchable, so treat this as an
   *               experimental benchmark knob. */
  androidBackend: 'cpu' | 'nnapi';
  /** Android only: keep the scan running when the app is backgrounded / screen-off, via a
   *  foreground service + wake lock (persistent notification). When on, an interrupted scan also
   *  auto-resumes on the next launch. Off = scan only while the app is open. No effect on iOS. */
  backgroundScan: boolean;
}

/** A scanned face that matched a bucket. */
export interface ScanMatch {
  uri: string;
  bucket: RefBucket;
  similarity: number;
  faceBox: FaceBox;
  imageWidth: number;
  imageHeight: number;
}

export interface TimingStats {
  n: number;
  mean: number;
  median: number;
  p90: number;
  p99: number;
  min: number;
  max: number;
  std: number;
}

export interface StageTimings {
  decode: TimingStats | null;
  detect: TimingStats | null;
  crop: TimingStats | null;
  preprocess: TimingStats | null;
  infer: TimingStats | null;
  similarity: TimingStats | null;
  total: TimingStats | null;
}

/** Running per-stage means (ms) emitted live during a scan for the on-screen breakdown. */
export interface LiveStageStats {
  decode: number;
  detect: number;
  align: number;
  preprocess: number;
  embed: number;
  similarity: number;
  total: number;
}

/** Result of one scan with one model — the comparable unit. */
export interface ScanRun {
  id: string;
  modelId: string;
  modelLabel: string;
  detectorLabel: string;
  startedAt: number;
  durationMs: number; // wall-clock for the whole scan
  loadMs: number; // model session creation
  fileSizeBytes: number;
  photosScanned: number;
  facesFound: number;
  matches: ScanMatch[]; // all matches (both buckets), ranked desc by similarity
  babyMatchCount: number;
  parentMatchCount: number;
  avgMsPerPhoto: number;
  facesPerSec: number;
  stages: StageTimings;
  /** Mean of each face's best baby-vs-parent score gap — a quick "separation" proxy. */
  separation: number;
  /** Cross-run cache reuse for this scan (detection + crop). Hits ⇒ work skipped vs a
   *  cold run; lets the Results screen show the warm-rerun speed-up. */
  cacheStats?: { detectHits: number; detectMisses: number; cropHits: number; cropMisses: number };
  error?: string;
}

/** Streamed once per scanned photo (as it finishes, possibly out of order) so the live
 *  screen can show matched/unmatched thumbnails in real time. */
export interface ScanPhotoEvent {
  index: number; // 0-based position in the scanned list
  uri: string; // original gallery URI (full-res — do NOT render directly in a grid)
  thumbUri?: string; // small persisted thumbnail (~150px) safe to render in the live grid
  faceCount: number;
  matched: boolean; // best face's similarity met the threshold
  bucket?: RefBucket; // best-matching bucket (set when matched)
  similarity: number; // best similarity across faces, or -1 if no face
  ms: number; // wall-clock to process this photo
}

export interface ScanProgress {
  phase: 'idle' | 'loading' | 'references' | 'scanning' | 'done' | 'error';
  message: string;
  photoIndex: number;
  photoCount: number;
  facesFound: number;
  matchesSoFar: number;
  fraction: number;
  /** Live per-stage means for the on-screen breakdown (present during 'scanning'). */
  liveStages?: LiveStageStats;
}
