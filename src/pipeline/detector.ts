/**
 * detector — pluggable face detection behind one interface, so the Scan screen can
 * compare detectors the same way the Model screen compares embedders.
 *
 *   • MLKitDetector — Google ML Kit (default; landmarks + classification, native).
 *   • YuNetDetector — OpenCV Zoo YuNet ONNX run through ONNX Runtime. The model is a
 *     fixed 640×640 input with 12 multi-stride outputs; we decode priors + bbox/kps and
 *     run NMS in JS (ported from OpenCV's FaceDetectorYN postProcess), then map YuNet's
 *     5 landmarks into the same Landmarks5 the ArcFace aligner expects.
 *
 * Both return boxes + landmarks in ORIGINAL image-pixel coordinates, so the downstream
 * crop/align/embed path is identical regardless of detector.
 */
import * as ort from 'onnxruntime-react-native';
import ImageEditor from '@react-native-community/image-editor';
import { Image as RNImage, Platform } from 'react-native';
import RNFS from 'react-native-fs';
import type { DetectResult, DetectedFace, Landmarks5, Point2D } from '../types';
import { detectFaces } from './detect';
import { decodeJpeg } from './decode';
import { ensureNamedAsset } from '../runtime/modelAssets';

export interface Detector {
  label: string;
  detect(path: string, minFaceSize: number): Promise<DetectResult>;
  dispose(): Promise<void>;
}

export type DetectorKind = 'mlkit' | 'yunet';

export async function createDetector(kind: DetectorKind): Promise<Detector> {
  if (kind === 'yunet') return YuNetDetector.create();
  return {
    label: 'ML Kit',
    detect: (path, minFaceSize) => detectFaces(path, minFaceSize),
    dispose: async () => {},
  };
}

// ─── YuNet ─────────────────────────────────────────────────────────────────────

const YUNET_ASSET = 'yunet_2023mar.onnx';
const YUNET_SIZE = 640; // fixed input of the 2023mar export
const STRIDES = [8, 16, 32] as const;
const SCORE_THRESH = 0.6;
const NMS_IOU = 0.3;

const noop = async () => {};
function toUri(p: string): string {
  return p.startsWith('file://') ? p : `file://${p}`;
}
function imageSize(uri: string): Promise<{ width: number; height: number }> {
  return new Promise(resolve =>
    RNImage.getSize(uri, (width, height) => resolve({ width, height }), () => resolve({ width: 0, height: 0 })),
  );
}

interface RawFace {
  score: number;
  x1: number;
  y1: number;
  w: number;
  h: number;
  pts: Point2D[]; // 5: rightEye, leftEye, nose, rightMouth, leftMouth (YuNet order)
}

class YuNetDetector implements Detector {
  label = 'YuNet';
  private constructor(private session: ort.InferenceSession) {}

  static async create(): Promise<YuNetDetector> {
    const path = await ensureNamedAsset(YUNET_ASSET);
    const providers = Platform.OS === 'ios' ? ['coreml', 'cpu'] : ['nnapi', 'xnnpack', 'cpu'];
    let session: ort.InferenceSession;
    try {
      session = await ort.InferenceSession.create(path, {
        executionProviders: providers,
      } as ort.InferenceSession.SessionOptions);
    } catch {
      session = await ort.InferenceSession.create(path);
    }
    return new YuNetDetector(session);
  }

  async detect(path: string, minFaceSize: number): Promise<DetectResult> {
    const uri = toUri(path);
    const { width: origW, height: origH } = await imageSize(uri);
    if (!origW || !origH) return { faces: [], imageWidth: 0, imageHeight: 0 };

    // Stretch-resize the whole image to the model's fixed 640×640 (native, fast).
    const res = (await ImageEditor.cropImage(uri, {
      offset: { x: 0, y: 0 },
      size: { width: origW, height: origH },
      displaySize: { width: YUNET_SIZE, height: YUNET_SIZE },
      resizeMode: 'stretch',
      format: 'jpeg',
      quality: 1,
    })) as { uri: string } | string;
    const cropUri = typeof res === 'string' ? res : res.uri;
    let rgba;
    try {
      rgba = await decodeJpeg(cropUri);
    } finally {
      const p = cropUri.startsWith('file://') ? cropUri.slice(7) : cropUri;
      RNFS.unlink(p).catch(noop);
    }

    // BGR, raw 0–255, NCHW (YuNet uses OpenCV BGR with no normalization).
    const plane = YUNET_SIZE * YUNET_SIZE;
    const input = new Float32Array(plane * 3);
    for (let p = 0; p < plane; p++) {
      const s = p * 4;
      input[p] = rgba.data[s + 2]; // B
      input[plane + p] = rgba.data[s + 1]; // G
      input[plane * 2 + p] = rgba.data[s]; // R
    }

    const inName = this.session.inputNames[0];
    const out = await this.session.run({
      [inName]: new ort.Tensor('float32', input, [1, 3, YUNET_SIZE, YUNET_SIZE]),
    });
    const get = (n: string) => out[n].data as Float32Array;

    const raw: RawFace[] = [];
    for (const stride of STRIDES) {
      const feat = YUNET_SIZE / stride;
      const cls = get(`cls_${stride}`);
      const obj = get(`obj_${stride}`);
      const bbox = get(`bbox_${stride}`);
      const kps = get(`kps_${stride}`);
      for (let i = 0; i < feat * feat; i++) {
        const c = i % feat;
        const r = Math.floor(i / feat);
        const clsv = Math.min(1, Math.max(0, cls[i]));
        const objv = Math.min(1, Math.max(0, obj[i]));
        const score = Math.sqrt(clsv * objv);
        if (score < SCORE_THRESH) continue;
        const cx = (c + bbox[i * 4]) * stride;
        const cy = (r + bbox[i * 4 + 1]) * stride;
        const w = Math.exp(bbox[i * 4 + 2]) * stride;
        const h = Math.exp(bbox[i * 4 + 3]) * stride;
        const pts: Point2D[] = [];
        for (let k = 0; k < 5; k++) {
          pts.push({
            x: (c + kps[i * 10 + 2 * k]) * stride,
            y: (r + kps[i * 10 + 2 * k + 1]) * stride,
          });
        }
        raw.push({ score, x1: cx - w / 2, y1: cy - h / 2, w, h, pts });
      }
    }

    const kept = nms(raw, NMS_IOU);
    const sx = origW / YUNET_SIZE;
    const sy = origH / YUNET_SIZE;
    const faces: DetectedFace[] = [];
    for (const f of kept) {
      const width = f.w * sx;
      const height = f.h * sy;
      if (Math.min(width, height) < minFaceSize) continue;
      const sp = (p: Point2D): Point2D => ({ x: p.x * sx, y: p.y * sy });
      // YuNet kps order: 0 right-eye, 1 left-eye, 2 nose, 3 right-mouth, 4 left-mouth.
      const landmarks: Landmarks5 = {
        rightEye: sp(f.pts[0]),
        leftEye: sp(f.pts[1]),
        noseBase: sp(f.pts[2]),
        mouthRight: sp(f.pts[3]),
        mouthLeft: sp(f.pts[4]),
      };
      faces.push({
        box: { left: f.x1 * sx, top: f.y1 * sy, width, height },
        landmarks,
      });
    }
    return { faces, imageWidth: origW, imageHeight: origH };
  }

  async dispose(): Promise<void> {
    try {
      await this.session.release();
    } catch {
      // ignore
    }
  }
}

function iou(a: RawFace, b: RawFace): number {
  const ax2 = a.x1 + a.w;
  const ay2 = a.y1 + a.h;
  const bx2 = b.x1 + b.w;
  const by2 = b.y1 + b.h;
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const uni = a.w * a.h + b.w * b.h - inter;
  return uni <= 0 ? 0 : inter / uni;
}

function nms(faces: RawFace[], iouThresh: number): RawFace[] {
  const sorted = [...faces].sort((p, q) => q.score - p.score);
  const kept: RawFace[] = [];
  for (const f of sorted) {
    if (kept.every(k => iou(f, k) < iouThresh)) kept.push(f);
  }
  return kept;
}
