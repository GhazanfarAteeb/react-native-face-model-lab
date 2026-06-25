/**
 * Embedder — uniform interface over the inference runtime (ONNX Runtime).
 *
 * `infer` takes an ALREADY-preprocessed input tensor (correct layout, channel order
 * and normalization for the spec) and returns the raw output vector. L2-normalization
 * happens one layer up so it's identical across models.
 */
import type { ModelSpec } from '../types';

export interface Embedder {
  spec: ModelSpec;
  loadMs: number;
  fileSizeBytes: number;
  /** Run one inference. `input` length must equal width*height*3. */
  infer(input: Float32Array): Promise<Float32Array>;
  release(): Promise<void>;
}

export async function createEmbedder(spec: ModelSpec): Promise<Embedder> {
  const { OnnxEmbedder } = await import('./OnnxEmbedder');
  return OnnxEmbedder.create(spec);
}
