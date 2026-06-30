/**
 * NativeFaceCoreML — JS handle to the native CoreML/ANE embedder (iOS only).
 *
 * Implemented natively in ios/FaceModelLab/FaceCoreML.mm. We resolve it from EITHER the
 * TurboModule registry (new-arch) OR NativeModules (legacy bridge interop), so whichever way
 * the native module registers, this finds it. Resolves to null when the module isn't in the
 * binary (e.g. before you've added FaceCoreML.mm to the target), which lets the pipeline fall
 * back to ORT-CPU with no crash.
 */
import { NativeModules, TurboModuleRegistry } from 'react-native';

export interface FaceCoreMLModule {
  /** Load a bundled CoreML model (compiled `.mlmodelc`). `inputName`/`outputName` may be ''
   *  to use the model's first input/output. Returns a handle >=0, or <0 on failure. */
  load(assetName: string, inputName: string, outputName: string): Promise<number>;
  /** Run one inference; `input` is the preprocessed tensor flattened in the model's layout. */
  infer(handle: number, input: number[]): Promise<number[]>;
  release(handle: number): Promise<void>;
}

function resolve(): FaceCoreMLModule | null {
  try {
    const tm = (TurboModuleRegistry as { get?: (n: string) => unknown }).get?.('NativeFaceCoreML');
    if (tm) return tm as FaceCoreMLModule;
  } catch {
    // not a turbomodule — fall through to the legacy registry
  }
  const legacy = (NativeModules as Record<string, unknown>).NativeFaceCoreML;
  return legacy ? (legacy as FaceCoreMLModule) : null;
}

export default resolve();
