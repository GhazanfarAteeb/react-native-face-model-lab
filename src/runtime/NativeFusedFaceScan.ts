/**
 * NativeFusedFaceScan — JS handle to the fused native ANE scan pipeline (iOS only).
 *
 * Implemented natively in ios/FaceModelLab/FusedFaceScan.mm. Combines face detection
 * (Vision) + crop/align + CoreML embedding into a single native call per photo.
 * This achieves ~70ms per photo vs ~1,300ms on the JS→ORT CPU path.
 *
 * Resolves to null when the native module isn't built, so the pipeline can fall back
 * to the standard ORT path gracefully.
 */
import { NativeModules, TurboModuleRegistry } from 'react-native';

export interface FusedScanResult {
  faces: Array<{ left: number; top: number; width: number; height: number }>;
  embeddings: number[][];
  imageWidth: number;
  imageHeight: number;
  timings: {
    decodeMs: number;
    detectMs: number;
    cropMs: number;
    alignMs: number;
    embedMs: number;
  };
}

export interface FusedFaceScanModule {
  /** Load a CoreML model for fused scanning. inputWidth/inputHeight = model's expected crop size (e.g. 96×112 for SphereFace, 112×112 for MobileFaceNet). */
  loadModel(assetName: string, inputName: string, outputName: string, inputWidth: number, inputHeight: number): Promise<boolean>;
  /** Run fused detect+crop+align+embed on a single photo. */
  scanPhoto(imagePath: string, paddingRatio: number): Promise<FusedScanResult>;
  /** Release the loaded model. */
  release(): Promise<void>;
  /** Get device info (ANE availability, system info). */
  getDeviceInfo(): Promise<{
    aneAvailable: boolean;
    systemName: string;
    systemVersion: string;
    model: string;
  }>;
}

function resolve(): FusedFaceScanModule | null {
  try {
    const tm = (TurboModuleRegistry as { get?: (n: string) => unknown }).get?.(
      'NativeFusedFaceScan',
    );
    if (tm) return tm as FusedFaceScanModule;
  } catch {
    // not a turbomodule — fall through to the legacy registry
  }
  const legacy = (NativeModules as Record<string, unknown>).NativeFusedFaceScan;
  return legacy ? (legacy as FusedFaceScanModule) : null;
}

export default resolve();
