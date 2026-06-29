/**
 * normalizeUri — turn any gallery URI into a plain file path ML Kit / ImageEditor can
 * read, copying to a temp file when needed (iOS ph://, Android content://).
 *
 * Returns { path, cleanup }. Callers MUST call cleanup() when done with the file to
 * avoid leaking temp copies across a few-hundred-photo scan.
 */
import RNFS from 'react-native-fs';
import { Platform } from 'react-native';

let counter = 0;
function tempPath(ext = 'jpg'): string {
  counter += 1;
  return `${RNFS.CachesDirectoryPath}/fml_${counter}_${Date.now()}.${ext}`;
}

export interface NormalizedUri {
  path: string; // no file:// prefix
  cleanup: () => Promise<void>;
  /** True when the returned file is already downscaled to ~maxDim (caller can skip a
   *  separate downscale pass). */
  downscaled?: boolean;
}

const noop = async () => {};

/** `maxDim` > 0 asks the source to deliver an already-downscaled copy (long side ≤ maxDim).
 *  On iOS this makes PhotoKit render the asset at size, replacing a full-res copy + a
 *  separate ImageEditor downscale with a single, much cheaper fetch. */
export async function normalizeUri(uri: string, maxDim = 0): Promise<NormalizedUri> {
  // Already a local file.
  if (uri.startsWith('file://')) return { path: uri.slice(7), cleanup: noop };
  if (uri.startsWith('/')) return { path: uri, cleanup: noop };

  // iOS Photos asset → copy out to a real file (downscaled by PhotoKit when maxDim>0).
  if (uri.startsWith('ph://') || uri.startsWith('assets-library://')) {
    const dest = tempPath();
    const rnfs = RNFS as typeof RNFS & {
      copyAssetsFileIOS?: (uri: string, dest: string, w: number, h: number) => Promise<string>;
    };
    if (Platform.OS === 'ios' && rnfs.copyAssetsFileIOS) {
      const d = maxDim > 0 ? Math.round(maxDim) : 0; // 0 = full resolution
      // RNFS default resizeMode 'contain' → fits within d×d preserving aspect.
      const out = await rnfs.copyAssetsFileIOS(uri, dest, d, d);
      const path = out.startsWith('file://') ? out.slice(7) : out;
      return { path, cleanup: () => RNFS.unlink(path).catch(noop), downscaled: d > 0 };
    }
  }

  // Android content:// (or anything else) → read bytes and write a temp file.
  if (uri.startsWith('content://')) {
    const dest = tempPath();
    const b64 = await RNFS.readFile(uri, 'base64');
    await RNFS.writeFile(dest, b64, 'base64');
    return { path: dest, cleanup: () => RNFS.unlink(dest).catch(noop) };
  }

  // Fallback: hope it's readable as-is.
  return { path: uri, cleanup: noop };
}
