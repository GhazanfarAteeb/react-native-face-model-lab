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
 * Files live in the cache dir, keyed by a stable hash of the SOURCE photo uri so they
 * survive across runs — a re-scan of the same gallery (benchmark compare) reuses the
 * thumbnail without re-decoding the photo. Wiped only when the scan cache is cleared.
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

/** Stable 32-bit hash of the photo uri → a filename that's identical across runs. */
function hashUri(uri: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < uri.length; i++) {
    h ^= uri.charCodeAt(i);
    h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
  }
  return h.toString(16);
}

function imageSize(uri: string): Promise<{ width: number; height: number }> {
  return new Promise(resolve =>
    RNImage.getSize(uri, (width, height) => resolve({ width, height }), () => resolve({ width: 0, height: 0 })),
  );
}

/** Wipe the thumbnail dir. Tied to "Clear scan cache" — NOT called per-run, so reruns of
 *  the same gallery reuse thumbnails. */
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

/** Thumbnail for the photo at `uri`. Returns a cached thumbnail without decoding anything
 *  when one already exists (the warm-rerun fast path); otherwise calls `getSrc` to obtain a
 *  local image to resize from (this is what triggers the photo decode). Returns a file://
 *  uri, or undefined on any failure. */
export async function makeThumb(uri: string, getSrc: () => Promise<string>): Promise<string | undefined> {
  const dest = `${THUMB_DIR}/h_${hashUri(uri)}.jpg`;
  try {
    await ensureDir();
    // Fast path: a thumbnail for this exact photo already exists — reuse it, no decode.
    if (await RNFS.exists(dest)) return `file://${dest}`;

    const srcUri = toUri(await getSrc());
    const { width, height } = await imageSize(srcUri);
    if (!width || !height) return undefined;
    const res = (await ImageEditor.cropImage(srcUri, {
      offset: { x: 0, y: 0 },
      size: { width, height },
      displaySize: { width: THUMB_SIZE, height: THUMB_SIZE },
      resizeMode: 'cover',
      format: 'jpeg',
      quality: 0.7,
    })) as { uri: string } | string;
    const outUri = typeof res === 'string' ? res : res.uri;
    const out = outUri.startsWith('file://') ? outUri.slice(7) : outUri;
    await RNFS.moveFile(out, dest);
    return `file://${dest}`;
  } catch {
    return undefined;
  }
}
