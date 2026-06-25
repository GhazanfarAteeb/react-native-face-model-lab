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
}

const noop = async () => {};

export async function normalizeUri(uri: string): Promise<NormalizedUri> {
  // Already a local file.
  if (uri.startsWith('file://')) return { path: uri.slice(7), cleanup: noop };
  if (uri.startsWith('/')) return { path: uri, cleanup: noop };

  // iOS Photos asset → copy out to a real file.
  if (uri.startsWith('ph://') || uri.startsWith('assets-library://')) {
    const dest = tempPath();
    const rnfs = RNFS as typeof RNFS & {
      copyAssetsFileIOS?: (uri: string, dest: string, w: number, h: number) => Promise<string>;
    };
    if (Platform.OS === 'ios' && rnfs.copyAssetsFileIOS) {
      const out = await rnfs.copyAssetsFileIOS(uri, dest, 0, 0);
      const path = out.startsWith('file://') ? out.slice(7) : out;
      return { path, cleanup: () => RNFS.unlink(path).catch(noop) };
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
