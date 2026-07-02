/**
 * OnnxEmbedder — onnxruntime-react-native backend.
 *
 * Runs the embedding model on ORT's CPU provider (default) or, on iOS, the CoreML execution
 * provider. Inference is serialized through the shared `ortLock` together with the ONNX
 * detectors: concurrent `session.run` calls on the ORT-RN bridge have a SIGABRT history on
 * this project. The native CoreMLEmbedder is a separate runtime and does NOT take this lock,
 * so ORT-CPU and CoreML/ANE can run in parallel (the HybridEmbedder relies on that). We copy
 * the output out of ORT's reused buffer before returning, and never dispose tensors.
 */
import * as ort from 'onnxruntime-react-native';
import type { ModelSpec } from '../types';
import type { Embedder, InferResult } from './Embedder';
import { ortLock } from './ortLock';
import { ensureModelFile, modelFileSize } from './modelAssets';
import { now } from '../bench/timing';

export interface OnnxEmbedderOptions {
  /** 'cpu' (default everywhere), 'coreml' (iOS only — adds the CoreML EP ahead of CPU), or
   *  'nnapi' (Android only — adds the NNAPI GPU/NPU EP ahead of CPU). */
  executionProvider?: 'cpu' | 'coreml' | 'nnapi';
}

export class OnnxEmbedder implements Embedder {
  private constructor(
    public spec: ModelSpec,
    public loadMs: number,
    public fileSizeBytes: number,
    private session: ort.InferenceSession,
    private inputName: string,
    private outputName: string,
    private shape: number[],
  ) {}

  static async create(spec: ModelSpec, opts?: OnnxEmbedderOptions): Promise<OnnxEmbedder> {
    const path = await ensureModelFile(spec);
    const fileSizeBytes = await modelFileSize(spec);
    const t0 = now();

    // Accelerator EPs are requested only when explicitly asked for (iOS CoreML hybrid/coreml-only,
    // or Android 'nnapi'). CoreML rarely engages the ANE for these small models and adds a ~1s
    // compile + per-call overhead, so CPU is the default. Android accelerator EPs can hard-abort,
    // so 'nnapi' is an opt-in benchmark knob; both keep CPU as the fallback provider.
    const ep = opts?.executionProvider;
    const providers =
      ep === 'coreml' ? ['coreml', 'cpu'] : ep === 'nnapi' ? ['nnapi', 'cpu'] : ['cpu'];
    console.log('[FML] loading ONNX embedder:', spec.id, 'providers:', providers.join('+'), path);
    let session: ort.InferenceSession;
    try {
      session = await ort.InferenceSession.create(path, {
        executionProviders: providers,
      } as ort.InferenceSession.SessionOptions);
    } catch {
      // Accelerator unavailable for this model — fall back to plain CPU.
      session = await ort.InferenceSession.create(path);
    }
    const loadMs = now() - t0;

    const inputName = spec.output.inputName ?? session.inputNames[0];
    const outputName = spec.output.outputName ?? session.outputNames[0];
    const { width, height, layout } = spec.input;
    const shape = layout === 'NCHW' ? [1, 3, height, width] : [1, height, width, 3];

    return new OnnxEmbedder(spec, loadMs, fileSizeBytes, session, inputName, outputName, shape);
  }

  async infer(input: Float32Array): Promise<InferResult> {
    // Serialize on the shared ORT lock; measure only the compute (lock wait stays outside).
    return ortLock.run(async () => {
      const t0 = now();
      const tensor = new ort.Tensor('float32', input, this.shape);
      const output = await this.session.run({ [this.inputName]: tensor });
      const raw = output[this.outputName].data as Float32Array;
      // ORT reuses the output buffer across runs — copy before the next call.
      const copy = new Float32Array(raw.length);
      copy.set(raw);
      return { data: copy, computeMs: now() - t0 };
    });
  }

  async release(): Promise<void> {
    try {
      await this.session.release();
    } catch {
      // ignore
    }
  }
}
