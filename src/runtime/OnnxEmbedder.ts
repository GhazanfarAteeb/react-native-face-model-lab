/**
 * OnnxEmbedder — onnxruntime-react-native backend.
 *
 * Prefers the platform accelerator (CoreML on iOS, NNAPI/XNNPACK on Android) with a
 * CPU fallback, matching rnbaby's FaceEmbeddingService. Inference is sequential here
 * (one face at a time), so we don't need rnbaby's serial queue — but we DO copy the
 * output out of ORT's reused buffer before returning, and never dispose tensors.
 */
import * as ort from 'onnxruntime-react-native';
import { Platform } from 'react-native';
import type { ModelSpec } from '../types';
import type { Embedder } from './Embedder';
import { ensureModelFile, modelFileSize } from './modelAssets';
import { now } from '../bench/timing';

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

  static async create(spec: ModelSpec): Promise<OnnxEmbedder> {
    const path = await ensureModelFile(spec);
    const fileSizeBytes = await modelFileSize(spec);
    const t0 = now();

    // Android: XNNPACK (fast, stable CPU kernels) + CPU. We deliberately DON'T use NNAPI:
    // it can hard-abort (std::terminate / SIGABRT) inside ORT session creation on graphs
    // it can't partition (e.g. SCRFD), and that native abort is NOT catchable by JS — it
    // crashes the whole app. XNNPACK never does that.
    const providers = Platform.OS === 'ios' ? ['coreml', 'cpu'] : ['xnnpack', 'cpu'];
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

  async infer(input: Float32Array): Promise<Float32Array> {
    const tensor = new ort.Tensor('float32', input, this.shape);
    const output = await this.session.run({ [this.inputName]: tensor });
    const raw = output[this.outputName].data as Float32Array;
    // ORT reuses the output buffer across runs — copy before the next call.
    const copy = new Float32Array(raw.length);
    copy.set(raw);
    return copy;
  }

  async release(): Promise<void> {
    try {
      await this.session.release();
    } catch {
      // ignore
    }
  }
}
