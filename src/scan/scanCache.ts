/**
 * scanCache — session (in-memory) reuse of the model-AGNOSTIC stages across runs.
 *
 * This is a benchmark: you scan the SAME gallery with many models. Detection and the
 * aligned crop don't depend on the embedding model — only on (photo, detector, working
 * resolution) for detection, and additionally (input size, alignment) for the crop. So:
 *
 *   • detection result  → keyed by (uri, detector, maxImageDim) — reused by EVERY model.
 *   • aligned crop (RGBA) → keyed by (uri, faceIndex, W×H, align, detector, maxImageDim) —
 *     reused by every model sharing that input size + alignment (e.g. all 112²/arcface
 *     models: MobileFaceNet, SFace, EdgeFace, FaceLiVT). Only the cheap preprocess +
 *     inference re-run per model.
 *
 * Detection entries are tiny (boxes + landmarks); crops hold W·H·4 bytes each, so the crop
 * cache is a bounded LRU. Every read records a hit/miss into the caller's CacheStats so the
 * speed-up is measurable on the Results screen (cold first run vs warm reruns).
 *
 * Keys include detector + maxImageDim, so changing either simply produces fresh keys (no
 * stale reuse). The cache is correct across reference and model changes because neither
 * detection nor cropping depends on them.
 */
import type { DetectResult, RgbaImage } from '../types';

export interface CacheStats {
  detectHits: number;
  detectMisses: number;
  cropHits: number;
  cropMisses: number;
}

export function emptyCacheStats(): CacheStats {
  return { detectHits: 0, detectMisses: 0, cropHits: 0, cropMisses: 0 };
}

let enabled = true;
let cropCap = 400; // ~20-40MB depending on model input size
const detectCache = new Map<string, DetectResult>();
const cropCache = new Map<string, RgbaImage>(); // insertion order = LRU

export function setCacheEnabled(on: boolean): void {
  enabled = on;
}

export function detectKey(uri: string, detector: string, maxImageDim: number): string {
  return `${detector}|${maxImageDim}|${uri}`;
}

export function cropKey(
  uri: string,
  faceIndex: number,
  w: number,
  h: number,
  align: string,
  detector: string,
  maxImageDim: number,
): string {
  return `${detector}|${maxImageDim}|${w}x${h}|${align}|${faceIndex}|${uri}`;
}

export function getDetection(key: string): DetectResult | undefined {
  return enabled ? detectCache.get(key) : undefined;
}

export function setDetection(key: string, value: DetectResult): void {
  if (enabled) detectCache.set(key, value);
}

export function getCrop(key: string): RgbaImage | undefined {
  if (!enabled) return undefined;
  const v = cropCache.get(key);
  if (v) {
    // LRU bump: move to newest.
    cropCache.delete(key);
    cropCache.set(key, v);
  }
  return v;
}

export function setCrop(key: string, value: RgbaImage): void {
  if (!enabled || cropCap === 0) return;
  cropCache.set(key, value);
  while (cropCache.size > cropCap) {
    const oldest = cropCache.keys().next().value;
    if (oldest === undefined) break;
    cropCache.delete(oldest);
  }
}

export function cacheSizes(): { detect: number; crop: number } {
  return { detect: detectCache.size, crop: cropCache.size };
}

/** Reset to a cold state — use to benchmark the un-cached (first-run) cost. */
export function clearScanCache(): { detect: number; crop: number } {
  const sizes = cacheSizes();
  detectCache.clear();
  cropCache.clear();
  return sizes;
}
