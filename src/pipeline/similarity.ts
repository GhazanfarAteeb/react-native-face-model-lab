/**
 * similarity — L2 normalization + cosine similarity. Mirrors rnbaby's
 * FaceEmbeddingMath.l2Normalize and scan/utils.cosineSimilarity so benchmark
 * scores are directly comparable to production matching.
 */

/** L2-normalize a vector. A zero-magnitude (failed) embedding yields a zero vector
 *  rather than NaNs. Always allocates a fresh array. */
export function l2Normalize(v: Float32Array): Float32Array {
  let sumSq = 0;
  for (let i = 0; i < v.length; i++) sumSq += v[i] * v[i];
  const mag = Math.sqrt(sumSq);
  const out = new Float32Array(v.length);
  if (mag === 0) return out;
  for (let i = 0; i < v.length; i++) out[i] = v[i] / mag;
  return out;
}

/** Cosine similarity. With L2-normalized inputs this is just the dot product, but we
 *  divide by magnitudes defensively so callers can't accidentally compare un-normalized
 *  vectors and read a wrong score. */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}
