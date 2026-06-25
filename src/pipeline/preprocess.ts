/**
 * preprocess — RGBA crop → normalized Float32 input tensor for a model.
 *
 * Generalizes rnbaby's FaceEmbeddingMath to be spec-driven: it handles RGB/BGR
 * channel order, NCHW (planar) / NHWC (interleaved) layout, and scalar or per-channel
 * mean/std — so one function feeds MobileFaceNet ((x−127.5)/128, RGB, NCHW), SFace
 * (raw 0–255, BGR, NCHW), EdgeFace ((x−127.5)/127.5) and FaceNet (NHWC) alike.
 *
 *   out = (channelValue − mean[c]) / std[c]
 */
import type { ModelSpec, RgbaImage } from '../types';

function triple(v: number | [number, number, number]): [number, number, number] {
  return typeof v === 'number' ? [v, v, v] : v;
}

export function preprocess(rgba: RgbaImage, spec: ModelSpec): Float32Array {
  const { width: W, height: H, layout, channels } = spec.input;
  if (rgba.width !== W || rgba.height !== H) {
    throw new Error(`preprocess: crop is ${rgba.width}×${rgba.height}, expected ${W}×${H}`);
  }
  const src = rgba.data; // RGBA, row-major
  const pixelCount = W * H;
  // Whether output channel index 0,1,2 maps to source R,G,B or B,G,R.
  const srcOrder = channels === 'BGR' ? [2, 1, 0] : [0, 1, 2];

  // FaceNet prewhiten: standardize the whole crop to zero-mean/unit-variance.
  let mean: [number, number, number];
  let std: [number, number, number];
  if (spec.norm.prewhiten) {
    let sum = 0;
    let sumSq = 0;
    for (let p = 0; p < pixelCount; p++) {
      const s = p * 4;
      for (let c = 0; c < 3; c++) {
        const v = src[s + c]; // mean/std are channel-agnostic for prewhiten
        sum += v;
        sumSq += v * v;
      }
    }
    const n = pixelCount * 3;
    const m = sum / n;
    const variance = Math.max(0, sumSq / n - m * m);
    const stdAdj = Math.max(Math.sqrt(variance), 1 / Math.sqrt(n));
    mean = [m, m, m];
    std = [stdAdj, stdAdj, stdAdj];
  } else {
    mean = triple(spec.norm.mean);
    std = triple(spec.norm.std);
  }

  const out = new Float32Array(W * H * 3);
  const plane = W * H;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const px = y * W + x;
      const s = px * 4;
      for (let c = 0; c < 3; c++) {
        const v = (src[s + srcOrder[c]] - mean[c]) / std[c];
        if (layout === 'NCHW') {
          out[c * plane + px] = v; // planar: [C][H*W]
        } else {
          out[px * 3 + c] = v; // interleaved: [H*W][C]
        }
      }
    }
  }
  return out;
}
