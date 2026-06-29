/**
 * processor — the seam for moving the CPU-bound preprocess step OFF the JS thread.
 *
 * `preprocess` (and, if you extend this, jpeg decode + the arcface bilinear warp) is pure
 * TypedArray math — perfect to run on a worklet/worker thread so the UI thread stays free
 * on every device. To keep the app buildable without the native worklet dependency, the
 * pipeline never imports a worklet runtime directly: it calls `getImageProcessor()`, which
 * returns the inline (main-thread) implementation by default. Activating the off-thread
 * path is a one-liner at app start — `setImageProcessor(workletProcessor)` — with NO change
 * to the scan code. See WORKLET_OFFLOAD.md for the worklet implementation and build steps.
 *
 * The interface is async on purpose: inline it resolves immediately; off-thread it resolves
 * when the worker returns the tensor. The scanner already awaits it, so swapping the
 * implementation needs no further changes.
 */
import { preprocess } from './preprocess';
import type { ModelSpec, RgbaImage } from '../types';

export interface ImageProcessor {
  /** Crop RGBA → model input tensor (correct layout/channel order/normalization). */
  preprocess(rgba: RgbaImage, spec: ModelSpec): Promise<Float32Array>;
}

const inlineProcessor: ImageProcessor = {
  async preprocess(rgba, spec) {
    return preprocess(rgba, spec);
  },
};

let active: ImageProcessor = inlineProcessor;

export function getImageProcessor(): ImageProcessor {
  return active;
}

/** Swap in an off-thread implementation (e.g. a worklet-backed processor) at app start.
 *  Pass nothing to revert to the inline main-thread processor. */
export function setImageProcessor(processor: ImageProcessor | null): void {
  active = processor ?? inlineProcessor;
}
