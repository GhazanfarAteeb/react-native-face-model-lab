/**
 * CoreMLEmbedder — native CoreML embedder running on the Apple Neural Engine (iOS only).
 *
 * Wraps the NativeFaceCoreML TurboModule. `create` returns null (rather than throwing) when
 * the native module isn't built yet or the model has no bundled `.mlpackage` (`spec.coremlAsset`),
 * so the runtime can fall back to ORT-CPU transparently.
 *
 * Like OnnxEmbedder it serializes on a per-instance lock and reports pure compute time. Because
 * this is a different native runtime from ORT (CoreML framework, not the onnxruntime-RN bridge),
 * a CoreMLEmbedder and an OnnxEmbedder can run concurrently — that parallelism is the whole
 * point of the hybrid pipeline.
 */
import type { ModelSpec } from '../types';
import type { Embedder, InferResult } from './Embedder';
import { Mutex } from '../scan/concurrency';
import { modelFileSize } from './modelAssets';
import { now } from '../bench/timing';
import NativeFaceCoreML from './NativeFaceCoreML';

export class CoreMLEmbedder implements Embedder {
  private lock = new Mutex();

  private constructor(
    public spec: ModelSpec,
    public loadMs: number,
    public fileSizeBytes: number,
    private handle: number,
  ) {}

  /** Returns a CoreMLEmbedder, or null if the ANE path is unavailable for this model. */
  static async create(spec: ModelSpec): Promise<CoreMLEmbedder | null> {
    if (!NativeFaceCoreML) return null; // native module not built into this binary
    if (!spec.coremlAsset) return null; // no converted .mlpackage shipped for this model

    const inputName = spec.output.inputName ?? '';
    const outputName = spec.output.outputName ?? '';
    const t0 = now();
    try {
      const handle = await NativeFaceCoreML.load(spec.coremlAsset, inputName, outputName);
      if (handle < 0) return null;
      const loadMs = now() - t0;
      // Report the ONNX file size for parity in the UI (the .mlpackage size isn't tracked).
      const fileSizeBytes = await modelFileSize(spec);
      return new CoreMLEmbedder(spec, loadMs, fileSizeBytes, handle);
    } catch {
      return null;
    }
  }

  async infer(input: Float32Array): Promise<InferResult> {
    return this.lock.run(async () => {
      const t0 = now();
      // number[] is the codegen-supported array type; Array.from copies the typed array.
      const out = await NativeFaceCoreML!.infer(this.handle, Array.from(input));
      return { data: Float32Array.from(out), computeMs: now() - t0 };
    });
  }

  async release(): Promise<void> {
    try {
      await NativeFaceCoreML?.release(this.handle);
    } catch {
      // ignore
    }
  }
}
