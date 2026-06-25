/**
 * embedFace — the per-image pipeline: detect → (per face) crop → preprocess → infer →
 * L2-normalize. Every stage's elapsed time is pushed into a shared sink so the scanner
 * can report median/p90 per stage. Detection is timed once per image; crop/preprocess/
 * infer once per face.
 */
import { cropFace } from '../pipeline/crop';
import { preprocess } from '../pipeline/preprocess';
import { l2Normalize } from '../pipeline/similarity';
import { now } from '../bench/timing';
import type { Embedder } from '../runtime/Embedder';
import type { Detector } from '../pipeline/detector';
import type { AlignMode, FaceBox, ModelSpec } from '../types';

export interface TimingSink {
  detect: number[];
  crop: number[];
  preprocess: number[];
  infer: number[];
}

export function emptySink(): TimingSink {
  return { detect: [], crop: [], preprocess: [], infer: [] };
}

export interface FaceEmbedding {
  embedding: Float32Array;
  box: FaceBox;
  imageWidth: number;
  imageHeight: number;
}

export async function embedImage(
  path: string,
  embedder: Embedder,
  spec: ModelSpec,
  align: AlignMode,
  minFaceSize: number,
  sink: TimingSink,
  detector: Detector,
): Promise<FaceEmbedding[]> {
  const t0 = now();
  const det = await detector.detect(path, minFaceSize);
  sink.detect.push(now() - t0);

  const out: FaceEmbedding[] = [];
  for (const face of det.faces) {
    const c0 = now();
    const rgba = await cropFace(path, face.box, face.landmarks, det.imageWidth, det.imageHeight, spec, align);
    sink.crop.push(now() - c0);

    const p0 = now();
    const input = preprocess(rgba, spec);
    sink.preprocess.push(now() - p0);

    const i0 = now();
    const raw = await embedder.infer(input);
    sink.infer.push(now() - i0);

    out.push({
      embedding: l2Normalize(raw),
      box: face.box,
      imageWidth: det.imageWidth,
      imageHeight: det.imageHeight,
    });
  }
  return out;
}
