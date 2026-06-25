/**
 * facePicker — extract individual faces from the recent gallery so the user can pick
 * reference faces from a grid of tiles (rnbaby's NewbornPicker/RecentPicker pattern),
 * instead of picking whole photos. Each detected face is cropped (with margin) to a
 * small thumbnail file that doubles as the reference image fed to the scanner later.
 */
import ImageEditor from '@react-native-community/image-editor';
import RNFS from 'react-native-fs';
import { recentPhotoUris } from './gallery';
import { normalizeUri } from '../pipeline/normalizeUri';
import { detectFaces } from '../pipeline/detect';
import type { FaceBox } from '../types';

export interface GalleryFace {
  id: string;
  thumbUri: string; // file:// cropped face (persisted)
  sourceUri: string; // original photo
  box: FaceBox;
  sizeScore: number; // shorter side in px — a cheap quality proxy
}

const THUMBS_DIR = `${RNFS.CachesDirectoryPath}/ref_faces`;
let thumbCounter = 0;

function toUri(p: string): string {
  return p.startsWith('file://') ? p : `file://${p}`;
}

async function cropThumb(path: string, box: FaceBox, imgW: number, imgH: number): Promise<string | null> {
  const pad = 0.3;
  const pw = box.width * (1 + 2 * pad);
  const ph = box.height * (1 + 2 * pad);
  const pl = Math.max(0, box.left - box.width * pad);
  const pt = Math.max(0, box.top - box.height * pad);
  const fw = imgW > 0 ? Math.min(pw, imgW - pl) : pw;
  const fh = imgH > 0 ? Math.min(ph, imgH - pt) : ph;
  if (fw < 20 || fh < 20) return null;
  try {
    const res = (await ImageEditor.cropImage(toUri(path), {
      offset: { x: pl, y: pt },
      size: { width: fw, height: fh },
      displaySize: { width: 220, height: 220 },
      resizeMode: 'cover',
      format: 'jpeg',
      quality: 0.9,
    })) as { uri: string } | string;
    const out = typeof res === 'string' ? res : res.uri;
    thumbCounter += 1;
    const dest = `${THUMBS_DIR}/face_${Date.now()}_${thumbCounter}.jpg`;
    await RNFS.moveFile(out.startsWith('file://') ? out.slice(7) : out, dest);
    return `file://${dest}`;
  } catch {
    return null;
  }
}

export interface ExtractOpts {
  limit: number;
  minFaceSize: number;
  onFace: (face: GalleryFace) => void;
  onProgress: (done: number, total: number, faces: number) => void;
  shouldCancel: () => boolean;
}

/** Walk recent photos, detect faces, emit a thumbnail per face via onFace (so the grid
 *  fills live). Resolves when done or cancelled. */
export async function extractGalleryFaces(opts: ExtractOpts): Promise<void> {
  await RNFS.mkdir(THUMBS_DIR).catch(() => {});
  const uris = await recentPhotoUris(opts.limit);
  let faces = 0;
  for (let i = 0; i < uris.length; i++) {
    if (opts.shouldCancel()) break;
    const norm = await normalizeUri(uris[i]);
    try {
      const det = await detectFaces(norm.path, opts.minFaceSize);
      for (let fi = 0; fi < det.faces.length; fi++) {
        const box = det.faces[fi].box;
        const thumb = await cropThumb(norm.path, box, det.imageWidth, det.imageHeight);
        if (!thumb) continue;
        faces += 1;
        opts.onFace({
          id: `${uris[i]}#${fi}`,
          thumbUri: thumb,
          sourceUri: uris[i],
          box,
          sizeScore: Math.min(box.width, box.height),
        });
      }
    } catch {
      // skip unreadable photo
    } finally {
      await norm.cleanup();
    }
    opts.onProgress(i + 1, uris.length, faces);
    if (i % 2 === 0) await new Promise<void>(r => setTimeout(r, 0)); // keep UI responsive
  }
}
