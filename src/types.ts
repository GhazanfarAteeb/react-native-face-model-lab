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
  /** How many gallery photos to scan (most recent first). */
  maxPhotos: number;
  /** Cosine score at/above which a face counts as a "match" for a bucket. */
  threshold: number;
  /** 'spec' uses the model's preferred crop; force-* overrides it. */
  align: 'spec' | 'force-arcface' | 'force-bbox';
  minFaceSize: number;
  /** Face detector: ML Kit (native), or an ONNX detector (YuNet / SCRFD / BlazeFace). */
  detector: 'mlkit' | 'yunet' | 'scrfd' | 'blazeface';
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
  detect: TimingStats | null;
  crop: TimingStats | null;
  preprocess: TimingStats | null;
  infer: TimingStats | null;
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
  error?: string;
}

export interface ScanProgress {
  phase: 'idle' | 'loading' | 'references' | 'scanning' | 'done' | 'error';
  message: string;
  photoIndex: number;
  photoCount: number;
  facesFound: number;
  matchesSoFar: number;
  fraction: number;
}
