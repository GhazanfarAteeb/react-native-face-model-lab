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

  /** Returns a CoreMLEmbedder, or null if the ANE path is unavailable for this model.
   *  When `spec.aneOnly` is true, loads with ANE-only compute units (no CPU/GPU fallback). */
  static async create(spec: ModelSpec): Promise<CoreMLEmbedder | null> {
    if (!NativeFaceCoreML) return null; // native module not built into this binary
    if (!spec.coremlAsset) return null; // no converted .mlpackage shipped for this model

    const inputName = spec.output.inputName ?? '';
    const outputName = spec.output.outputName ?? '';
    const t0 = now();
    try {
      const handle = spec.aneOnly
        ? await NativeFaceCoreML.loadANEOnly(spec.coremlAsset, inputName, outputName)
        : await NativeFaceCoreML.load(spec.coremlAsset, inputName, outputName);
      if (handle < 0) {
        console.log(`[CoreML] FAILED to load ${spec.coremlAsset} (ANE-only=${spec.aneOnly})`);
        return null;
      }
      const loadMs = now() - t0;

      // Log ANE/compute device info.
      try {
        const deviceInfo = await NativeFaceCoreML.getDeviceInfo();
        const modelInfo = await NativeFaceCoreML.getModelComputeDevice(handle);
        console.log(`[CoreML] Loaded ${spec.label} in ${loadMs.toFixed(0)}ms`);
        console.log(`[CoreML]   Device: ${deviceInfo.model} (${deviceInfo.systemName} ${deviceInfo.systemVersion})`);
        console.log(`[CoreML]   ANE available: ${deviceInfo.aneAvailable}`);
        console.log(`[CoreML]   Compute units: ${modelInfo.computeUnits}`);
        console.log(`[CoreML]   Inputs: ${modelInfo.modelDescription.inputs.join(', ')}`);
        console.log(`[CoreML]   Outputs: ${modelInfo.modelDescription.outputs.join(', ')}`);
      } catch {
        // getDeviceInfo not available (older native module) — skip logging
      }

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
