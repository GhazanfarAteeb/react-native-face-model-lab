/**
 * crop — turn a detected face into the model's input crop, two ways:
 *
 *   • bbox    : padded bounding-box crop + resize (native ImageEditor, resizeMode
 *               'cover') — exactly rnbaby's default crop.
 *   • arcface : warp the 5 landmarks onto the canonical template (similarity transform),
 *               so eyes/nose/mouth land where ArcFace-family models expect them. The
 *               native ImageEditor produces a modest working crop; the affine warp +
 *               bilinear resample run in JS (a few ms over ~W·H pixels).
 *
 * arcface falls back to bbox whenever landmarks are incomplete or the transform is
 * degenerate — alignment is never guessed.
 */
import ImageEditor from '@react-native-community/image-editor';
import RNFS from 'react-native-fs';
import type { AlignMode, FaceBox, Landmarks5, ModelSpec, Point2D, RgbaImage } from '../types';
import { decodeJpeg } from './decode';
import { alignmentMatrix, applyAffine, invertAffine } from '../models/arcface';

const noop = async () => {};
function toUri(p: string): string {
  return p.startsWith('file://') ? p : `file://${p}`;
}

async function cropRegion(
  srcUri: string,
  offset: Point2D,
  size: { width: number; height: number },
  dispW: number,
  dispH: number,
): Promise<RgbaImage> {
  const res = (await ImageEditor.cropImage(srcUri, {
    offset: { x: offset.x, y: offset.y },
    size,
    displaySize: { width: dispW, height: dispH },
    resizeMode: 'cover',
    format: 'jpeg',
    quality: 1,
  })) as { uri: string } | string;
  const uri = typeof res === 'string' ? res : res.uri;
  try {
    return await decodeJpeg(uri);
  } finally {
    const p = uri.startsWith('file://') ? uri.slice(7) : uri;
    RNFS.unlink(p).catch(noop);
  }
}

async function bboxCrop(
  srcUri: string,
  box: FaceBox,
  imgW: number,
  imgH: number,
  W: number,
  H: number,
): Promise<RgbaImage> {
  const pad = 0.1;
  const paddedW = box.width * (1 + 2 * pad);
  const paddedH = box.height * (1 + 2 * pad);
  const paddedL = Math.max(0, box.left - box.width * pad);
  const paddedT = Math.max(0, box.top - box.height * pad);
  const finalW = imgW > 0 ? Math.min(paddedW, imgW - paddedL) : paddedW;
  const finalH = imgH > 0 ? Math.min(paddedH, imgH - paddedT) : paddedH;
  return cropRegion(srcUri, { x: paddedL, y: paddedT }, { width: finalW, height: finalH }, W, H);
}

function sampleBilinear(img: RgbaImage, fx: number, fy: number, out: Uint8Array, di: number): void {
  const { width: w, height: h, data } = img;
  const x = Math.max(0, Math.min(w - 1, fx));
  const y = Math.max(0, Math.min(h - 1, fy));
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(w - 1, x0 + 1);
  const y1 = Math.min(h - 1, y0 + 1);
  const ax = x - x0;
  const ay = y - y0;
  const i00 = (y0 * w + x0) * 4;
  const i10 = (y0 * w + x1) * 4;
  const i01 = (y1 * w + x0) * 4;
  const i11 = (y1 * w + x1) * 4;
  for (let c = 0; c < 3; c++) {
    const top = data[i00 + c] * (1 - ax) + data[i10 + c] * ax;
    const bot = data[i01 + c] * (1 - ax) + data[i11 + c] * ax;
    out[di + c] = (top * (1 - ay) + bot * ay) | 0;
  }
  out[di + 3] = 255;
}

async function arcfaceCrop(
  srcUri: string,
  box: FaceBox,
  landmarks: Landmarks5,
  imgW: number,
  imgH: number,
  W: number,
  H: number,
): Promise<RgbaImage | null> {
  const { leftEye, rightEye, noseBase, mouthLeft, mouthRight } = landmarks;
  if (!leftEye || !rightEye || !noseBase || !mouthLeft || !mouthRight) return null;

  // Native working crop: a square region around the face, big enough to resample from.
  const cx = box.left + box.width / 2;
  const cy = box.top + box.height / 2;
  let side = Math.max(box.width, box.height) * 2.2;
  if (imgW > 0) side = Math.min(side, imgW);
  if (imgH > 0) side = Math.min(side, imgH);
  let oL = cx - side / 2;
  let oT = cy - side / 2;
  oL = imgW > 0 ? Math.min(Math.max(0, oL), imgW - side) : Math.max(0, oL);
  oT = imgH > 0 ? Math.min(Math.max(0, oT), imgH - side) : Math.max(0, oT);

  const S = 2 * Math.max(W, H);
  const work = await cropRegion(srcUri, { x: oL, y: oT }, { width: side, height: side }, S, S);

  // Map landmarks from source-image space into the working crop, then solve the
  // working→template transform and invert it for sampling.
  const k = S / side;
  const map = (p: Point2D): Point2D => ({ x: (p.x - oL) * k, y: (p.y - oT) * k });
  const workLm: Landmarks5 = {
    leftEye: map(leftEye),
    rightEye: map(rightEye),
    noseBase: map(noseBase),
    mouthLeft: map(mouthLeft),
    mouthRight: map(mouthRight),
  };
  const fwd = alignmentMatrix(workLm, W, H);
  if (!fwd) return null;
  const inv = invertAffine(fwd);
  if (!inv) return null;

  const out = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const s = applyAffine(inv, { x, y });
      sampleBilinear(work, s.x, s.y, out, (y * W + x) * 4);
    }
  }
  return { data: out, width: W, height: H };
}

/** Produce the model's input crop. `resolvedAlign` is the final decision after applying
 *  the global override (see scan settings). */
export async function cropFace(
  srcPath: string,
  box: FaceBox,
  landmarks: Landmarks5,
  imgW: number,
  imgH: number,
  spec: ModelSpec,
  resolvedAlign: AlignMode,
): Promise<RgbaImage> {
  const uri = toUri(srcPath);
  const { width: W, height: H } = spec.input;
  if (resolvedAlign === 'arcface') {
    const aligned = await arcfaceCrop(uri, box, landmarks, imgW, imgH, W, H);
    if (aligned) return aligned;
  }
  return bboxCrop(uri, box, imgW, imgH, W, H);
}
