/**
 * embedFace — the per-image pipeline: detect → (per face) crop → preprocess → infer →
 * L2-normalize. Every stage's elapsed time is pushed into a shared sink so the scanner
 * can report median/p90 per stage. Detection is timed once per image; crop/preprocess/
 * infer once per face.
 */
import { cropFace } from '../pipeline/crop';
import { getImageProcessor } from '../pipeline/processor';
import { l2Normalize } from '../pipeline/similarity';
import { ortLock } from '../runtime/ortLock';
import {
  type CacheStats,
  cropKey,
  detectKey,
  getCrop,
  getDetection,
  setCrop,
  setDetection,
} from './scanCache';
import { now } from '../bench/timing';
import type { Embedder } from '../runtime/Embedder';
import type { Detector } from '../pipeline/detector';
import type { AlignMode, FaceBox, ModelSpec } from '../types';

export interface TimingSink {
  decode: number[]; // image load + downscale (per photo)
  detect: number[]; // detector (per photo)
  crop: number[]; // crop + align (per face)
  preprocess: number[]; // tensor prep (per face)
  infer: number[]; // embedding model (per face)
  similarity: number[]; // cosine match vs references (per photo with faces)
  total: number[]; // end-to-end (per photo)
}

export function emptySink(): TimingSink {
  return { decode: [], detect: [], crop: [], preprocess: [], infer: [], similarity: [], total: [] };
}

export interface FaceEmbedding {
  embedding: Float32Array;
  box: FaceBox;
  imageWidth: number;
  imageHeight: number;
}

/** Lets embedImage reuse the model-agnostic stages (detect, crop) across runs. The keys
 *  are built from the ORIGINAL photo uri (stable), not the per-run temp path. */
export interface EmbedCacheCtx {
  uri: string;
  detectorKind: string;
  maxImageDim: number;
  stats: CacheStats;
}

export async function embedImage(
  path: string,
  embedder: Embedder,
  spec: ModelSpec,
  align: AlignMode,
  minFaceSize: number,
  sink: TimingSink,
  detector: Detector,
  cache?: EmbedCacheCtx,
): Promise<FaceEmbedding[]> {
  // ── Detection (model-agnostic → reused across every model) ──
  let det;
  const dKey = cache ? detectKey(cache.uri, cache.detectorKind, cache.maxImageDim) : undefined;
  const cachedDet = dKey ? getDetection(dKey) : undefined;
  if (cachedDet) {
    det = cachedDet;
    cache!.stats.detectHits += 1;
  } else {
    const t0 = now();
    det = await detector.detect(path, minFaceSize);
    sink.detect.push(now() - t0);
    if (dKey) {
      setDetection(dKey, det);
      cache!.stats.detectMisses += 1;
    }
  }

  const out: FaceEmbedding[] = [];
  for (let fi = 0; fi < det.faces.length; fi++) {
    const face = det.faces[fi];
    // ── Crop/align (shared by every model with the same input size + alignment) ──
    let rgba;
    const cKey = cache
      ? cropKey(cache.uri, fi, spec.input.width, spec.input.height, align, cache.detectorKind, cache.maxImageDim)
      : undefined;
    const cachedCrop = cKey ? getCrop(cKey) : undefined;
    if (cachedCrop) {
      rgba = cachedCrop;
      cache!.stats.cropHits += 1;
    } else {
      const c0 = now();
      rgba = await cropFace(path, face.box, face.landmarks, det.imageWidth, det.imageHeight, spec, align);
      sink.crop.push(now() - c0);
      if (cKey) {
        setCrop(cKey, rgba);
        cache!.stats.cropMisses += 1;
      }
    }

    // Routed through the processor seam so the preprocess math can run off-thread
    // (worklet) without touching this code — see pipeline/processor.ts. Timing here
    // therefore measures the real cost: inline math, or the worker round-trip.
    const p0 = now();
    const input = await getImageProcessor().preprocess(rgba, spec);
    sink.preprocess.push(now() - p0);

    // Serialize the actual inference (concurrent ORT runs can hard-abort). The lock
    // wait sits OUTSIDE the timed region so stage timings reflect compute, not queueing.
    const raw = await ortLock.run(async () => {
      const i0 = now();
      const r = await embedder.infer(input);
      sink.infer.push(now() - i0);
      return r;
    });

    out.push({
      embedding: l2Normalize(raw),
      box: face.box,
      imageWidth: det.imageWidth,
      imageHeight: det.imageHeight,
    });
  }
  return out;
}
