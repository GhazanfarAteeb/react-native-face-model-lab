/**
 * detect — ML Kit face detection. Run ONCE per image; the boxes + 5 landmarks are
 * reused for every model so detection cost and crop geometry are identical across the
 * comparison. Mirrors rnbaby's ML_KIT_OPTIONS_SCAN (accurate + all landmarks).
 */
import FaceDetection from '@react-native-ml-kit/face-detection';
import { Image as RNImage } from 'react-native';
import type { DetectResult, DetectedFace, Landmarks5, Point2D } from '../types';

function toUri(path: string): string {
  return path.startsWith('file://') ? path : `file://${path}`;
}

function imageSize(uri: string): Promise<{ width: number; height: number }> {
  return new Promise(resolve => {
    RNImage.getSize(
      uri,
      (width, height) => resolve({ width, height }),
      () => resolve({ width: 0, height: 0 }),
    );
  });
}

type MLPos = { position?: Point2D } | undefined;
function pos(p: MLPos): Point2D | undefined {
  return p?.position ? { x: p.position.x, y: p.position.y } : undefined;
}

/** Detect faces at least `minFaceSize` px on the shorter side. `path` is a plain file
 *  path (no scheme). */
export async function detectFaces(path: string, minFaceSize: number): Promise<DetectResult> {
  const uri = toUri(path);
  const [faces, dims] = await Promise.all([
    FaceDetection.detect(uri, {
      performanceMode: 'accurate',
      landmarkMode: 'all',
      contourMode: 'none',
      classificationMode: 'none',
      minFaceSize: 0.1,
    }),
    imageSize(uri),
  ]);

  const out: DetectedFace[] = [];
  for (const f of faces ?? []) {
    const frame = f.frame;
    if (!frame) continue;
    const width = frame.width ?? 0;
    const height = frame.height ?? 0;
    if (Math.min(width, height) < minFaceSize) continue;
    const lm = (f as { landmarks?: Record<string, MLPos> }).landmarks;
    const landmarks: Landmarks5 = {
      leftEye: pos(lm?.leftEye),
      rightEye: pos(lm?.rightEye),
      noseBase: pos(lm?.noseBase),
      mouthLeft: pos(lm?.mouthLeft),
      mouthRight: pos(lm?.mouthRight),
    };
    out.push({
      box: { left: frame.left ?? 0, top: frame.top ?? 0, width, height },
      landmarks,
    });
  }
  return { faces: out, imageWidth: dims.width, imageHeight: dims.height };
}
