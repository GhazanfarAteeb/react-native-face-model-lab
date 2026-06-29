/**
 * thumbnail — tiny persisted thumbnails for the live scan grid.
 *
 * The live grid must NOT render the original gallery photos: iOS decodes each `ph://` asset
 * at full resolution (~12MP ≈ 48MB) regardless of the cell size, and streaming hundreds of
 * them OOM-kills the app. Instead we generate a ~150px JPEG per photo (cheap native resize
 * from the already-downscaled working image) and render that — a decoded thumbnail is ~90KB,
 * so the whole visible grid costs a few MB.
 *
 * Best-effort: any failure returns undefined and the grid falls back to a status tile.
 * Files live in the cache dir and are wiped at the start of each scan.
 */
import ImageEditor from '@react-native-community/image-editor';
import { Image as RNImage } from 'react-native';
import RNFS from 'react-native-fs';

const THUMB_DIR = `${RNFS.CachesDirectoryPath}/fml_thumbs`;
const THUMB_SIZE = 150;

let dirReady = false;

function toUri(p: string): string {
  return p.startsWith('file://') ? p : `file://${p}`;
}

function imageSize(uri: string): Promise<{ width: number; height: number }> {
  return new Promise(resolve =>
    RNImage.getSize(uri, (width, height) => resolve({ width, height }), () => resolve({ width: 0, height: 0 })),
  );
}

/** Wipe the thumbnail dir (call at scan start so runs don't accumulate files). */
export async function clearThumbs(): Promise<void> {
  try {
    await RNFS.unlink(THUMB_DIR);
  } catch {
    // didn't exist — fine
  }
  dirReady = false;
}

async function ensureDir(): Promise<void> {
  if (dirReady) return;
  try {
    await RNFS.mkdir(THUMB_DIR);
  } catch {
    // already exists — fine
  }
  dirReady = true;
}

/** Make a small thumbnail of `srcPath` (a local file — e.g. the downscaled working image).
 *  Returns a file:// uri, or undefined on any failure. */
export async function makeThumb(srcPath: string, index: number): Promise<string | undefined> {
  try {
    await ensureDir();
    const uri = toUri(srcPath);
    const { width, height } = await imageSize(uri);
    if (!width || !height) return undefined;
    const res = (await ImageEditor.cropImage(uri, {
      offset: { x: 0, y: 0 },
      size: { width, height },
      displaySize: { width: THUMB_SIZE, height: THUMB_SIZE },
      resizeMode: 'cover',
      format: 'jpeg',
      quality: 0.7,
    })) as { uri: string } | string;
    const outUri = typeof res === 'string' ? res : res.uri;
    const out = outUri.startsWith('file://') ? outUri.slice(7) : outUri;
    const dest = `${THUMB_DIR}/t_${index}.jpg`;
    await RNFS.moveFile(out, dest);
    return `file://${dest}`;
  } catch {
    return undefined;
  }
}
