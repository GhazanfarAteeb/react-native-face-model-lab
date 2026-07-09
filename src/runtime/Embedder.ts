/**
 * Embedder — uniform interface over the inference runtime(s).
 *
 * `infer` takes an ALREADY-preprocessed input tensor (correct layout, channel order and
 * normalization for the spec) and returns the raw output vector plus the pure COMPUTE time
 * (excluding any queue/lock wait), so stage timings stay comparable across backends.
 * L2-normalization happens one layer up so it's identical across models.
 *
 * Three implementations sit behind this interface:
 *   • OnnxEmbedder   — onnxruntime-react-native (CPU, or the iOS CoreML EP).
 *   • CoreMLEmbedder — native CoreML on the Apple Neural Engine (iOS only).
 *   • HybridEmbedder — runs ORT-CPU and CoreML/ANE in parallel, routing each face to the
 *                      least-loaded backend (the self-balancing dual-queue).
 */
import { Platform } from 'react-native';
import type { ModelSpec } from '../types';

export interface InferResult {
  data: Float32Array;
  /** Pure compute time in ms (no lock/queue wait) — what the "embed" stage reports. */
  computeMs: number;
}

export interface Embedder {
  spec: ModelSpec;
  loadMs: number;
  fileSizeBytes: number;
  /** Run one inference. `input` length must equal width*height*3. */
  infer(input: Float32Array): Promise<InferResult>;
  release(): Promise<void>;
}

export interface EmbedderOptions {
  /** iOS only: which backend(s) to use. 'ane' = fused native path (bypasses this factory). */
  iosBackends?: Array<'cpu' | 'coreml' | 'ane'>;
  /** Android only: 'cpu' (default) or 'nnapi' (GPU/NPU accelerator EP). Ignored on iOS. */
  androidBackend?: 'cpu' | 'nnapi';
}

export async function createEmbedder(spec: ModelSpec, opts?: EmbedderOptions): Promise<Embedder> {
  const { OnnxEmbedder } = await import('./OnnxEmbedder');
  const backends = opts?.iosBackends?.length ? opts.iosBackends : ['cpu'];

  // ANE-only models: force CoreML-only path, ignore iosBackends settings.
  if (spec.aneOnly) {
    if (Platform.OS === 'ios') {
      const { CoreMLEmbedder } = await import('./CoreMLEmbedder');
      const ane = await CoreMLEmbedder.create(spec);
      if (ane) return ane;
      // ANE-only model failed to load on CoreML — can't fall back to CPU.
      throw new Error(`ANE-only model "${spec.label}" failed to load on CoreML/ANE.`);
    }
    throw new Error(`ANE-only model "${spec.label}" requires iOS with Apple Neural Engine.`);
  }

  // Android (or any non-iOS): plain ORT CPU, or the NNAPI accelerator EP when opted in.
  if (Platform.OS !== 'ios') {
    return OnnxEmbedder.create(spec, { executionProvider: opts?.androidBackend === 'nnapi' ? 'nnapi' : 'cpu' });
  }

  const wantCpu = backends.includes('cpu');
  const wantCoreml = backends.includes('coreml');

  // HYBRID: both selected → native CoreML/ANE alongside ORT-CPU, if the native module exists.
  if (wantCpu && wantCoreml) {
    const { CoreMLEmbedder } = await import('./CoreMLEmbedder');
    const ane = await CoreMLEmbedder.create(spec); // null when the native module isn't built
    if (ane) {
      const cpu = await OnnxEmbedder.create(spec, { executionProvider: 'cpu' });
      const { HybridEmbedder } = await import('./HybridEmbedder');
      return new HybridEmbedder(cpu, ane);
    }
    // Native module absent → degrade gracefully to plain CPU (still correct, just single-engine).
    return OnnxEmbedder.create(spec, { executionProvider: 'cpu' });
  }

  // Single backend: CoreML-only → ORT CoreML EP; otherwise CPU.
  return OnnxEmbedder.create(spec, { executionProvider: wantCoreml ? 'coreml' : 'cpu' });
}
