/**
 * arcface — 5-point similarity-transform alignment, ported from rnbaby's
 * src/utils/faceAlignment.ts and generalized to an arbitrary square output size.
 *
 * ArcFace-family models (MobileFaceNet, GhostFaceNet, EdgeFace, SFace) are trained
 * on faces warped to a canonical 112×112 template via a 5-landmark similarity
 * transform. Feeding them a plain bounding-box crop leaves pose/scale/tilt variance
 * in the embedding — the single biggest accuracy lever. This module produces the
 * forward matrix (image coords → template coords) which crop.ts applies in JS.
 */

import type { Landmarks5, Point2D } from '../types';

export type AffineMatrix = [[number, number, number], [number, number, number]];

/** Canonical ArcFace 5-point template for a 112×112 crop:
 *  left eye, right eye, nose tip, left mouth corner, right mouth corner. */
export const ARCFACE_TEMPLATE_112: readonly Point2D[] = [
  { x: 38.2946, y: 51.6963 },
  { x: 73.5318, y: 51.5014 },
  { x: 56.0252, y: 71.7366 },
  { x: 41.5493, y: 92.3655 },
  { x: 70.7299, y: 92.2041 },
];

/** Template scaled to a target W×H crop (the template is defined for 112×112). */
export function templateFor(width: number, height: number): Point2D[] {
  const sx = width / 112;
  const sy = height / 112;
  return ARCFACE_TEMPLATE_112.map(p => ({ x: p.x * sx, y: p.y * sy }));
}

/** Gaussian elimination with partial pivoting. Returns null if singular. */
function solveLinearSystem(M: number[][], b: number[]): number[] | null {
  const n = b.length;
  const a = M.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    let maxAbs = Math.abs(a[col][col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(a[r][col]);
      if (v > maxAbs) {
        maxAbs = v;
        pivot = r;
      }
    }
    if (maxAbs < 1e-12) return null;
    if (pivot !== col) {
      const tmp = a[pivot];
      a[pivot] = a[col];
      a[col] = tmp;
    }
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = a[r][col] / a[col][col];
      if (factor === 0) continue;
      for (let c = col; c <= n; c++) a[r][c] -= factor * a[col][c];
    }
  }
  const x = new Array(n);
  for (let i = 0; i < n; i++) x[i] = a[i][n] / a[i][i];
  return x;
}

/**
 * Least-squares similarity transform (uniform scale + rotation + translation; no
 * reflection/shear) mapping `src` → `dst`. Parameterized as
 *   x' =  a·x − b·y + tx
 *   y' =  b·x + a·y + ty
 * solved via the 4×4 normal equations.
 */
export function estimateSimilarityTransform(src: Point2D[], dst: Point2D[]): AffineMatrix | null {
  if (src.length !== dst.length || src.length < 2) return null;
  const ATA: number[][] = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ];
  const ATc = [0, 0, 0, 0];
  const addRow = (row: number[], rhs: number) => {
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) ATA[i][j] += row[i] * row[j];
      ATc[i] += row[i] * rhs;
    }
  };
  for (let i = 0; i < src.length; i++) {
    const { x, y } = src[i];
    addRow([x, -y, 1, 0], dst[i].x);
    addRow([y, x, 0, 1], dst[i].y);
  }
  const p = solveLinearSystem(ATA, ATc);
  if (!p) return null;
  const [a, b, tx, ty] = p;
  if (![a, b, tx, ty].every(Number.isFinite)) return null;
  if (Math.hypot(a, b) < 1e-6) return null;
  return [
    [a, -b, tx],
    [b, a, ty],
  ];
}

export function applyAffine(m: AffineMatrix, p: Point2D): Point2D {
  return {
    x: m[0][0] * p.x + m[0][1] * p.y + m[0][2],
    y: m[1][0] * p.x + m[1][1] * p.y + m[1][2],
  };
}

/** Inverse of a 2×3 affine (image warping samples SOURCE per DEST pixel). */
export function invertAffine(m: AffineMatrix): AffineMatrix | null {
  const [a, b, tx] = m[0];
  const [c, d, ty] = m[1];
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-12) return null;
  const ia = d / det;
  const ib = -b / det;
  const ic = -c / det;
  const id = a / det;
  const itx = -(ia * tx + ib * ty);
  const ity = -(ic * tx + id * ty);
  return [
    [ia, ib, itx],
    [ic, id, ity],
  ];
}

/** Forward matrix (image coords → W×H template coords) from 5 landmarks, or null
 *  when any landmark is missing (caller falls back to a bounding-box crop). */
export function alignmentMatrix(landmarks: Landmarks5, width: number, height: number): AffineMatrix | null {
  const { leftEye, rightEye, noseBase, mouthLeft, mouthRight } = landmarks;
  if (!leftEye || !rightEye || !noseBase || !mouthLeft || !mouthRight) return null;
  return estimateSimilarityTransform(
    [leftEye, rightEye, noseBase, mouthLeft, mouthRight],
    templateFor(width, height),
  );
}
