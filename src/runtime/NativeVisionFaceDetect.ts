/**
 * NativeVisionFaceDetect — JS handle to the native Apple Vision face detector (iOS only).
 *
 * Uses Apple's Vision framework (VNDetectFaceRectanglesRequest / VNDetectFaceLandmarksRequest)
 * for face detection. Unlike the JS-based ONNX detectors, this runs on the Neural Engine
 * via the Vision framework and is iOS-only.
 *
 * Returns bounding boxes + optional landmarks in original image pixel coordinates,
 * matching the interface expected by the downstream crop/align/embed pipeline.
 */
import { NativeModules, TurboModuleRegistry, Platform } from 'react-native';

export interface VisionDetectResult {
  faces: Array<{
    left: number;
    top: number;
    width: number;
    height: number;
    yaw: number;
    pitch: number;
    roll: number;
    leftEye?: { x: number; y: number };
    rightEye?: { x: number; y: number };
    noseBase?: { x: number; y: number };
    mouthLeft?: { x: number; y: number };
    mouthRight?: { x: number; y: number };
  }>;
  imageWidth: number;
  imageHeight: number;
}

export interface VisionFaceDetectModule {
  detect(imagePath: string, minFaceSize: number, useLandmarks: boolean): Promise<VisionDetectResult>;
}

function resolve(): VisionFaceDetectModule | null {
  if (Platform.OS !== 'ios') return null;
  try {
    const tm = (TurboModuleRegistry as { get?: (n: string) => unknown }).get?.('NativeVisionFaceDetect');
    if (tm) return tm as VisionFaceDetectModule;
  } catch {
    // not a turbomodule — fall through to the legacy registry
  }
  const legacy = (NativeModules as Record<string, unknown>).NativeVisionFaceDetect;
  return legacy ? (legacy as VisionFaceDetectModule) : null;
}

export default resolve();
