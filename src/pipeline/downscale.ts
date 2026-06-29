/**
 * downscale — cap a gallery photo to a working resolution before detection.
 *
 * Detection cost scales with pixel count, and gallery photos are often 12MP+ (on iOS
 * normalizeUri copies the asset at full size). Detecting faces on a 4000px image is far
 * slower than on a 1600px one, while faces in real photos stay well above the min size
 * after a uniform downscale — so this is the single biggest per-photo speed-up, on both
 * platforms, for the default ML Kit detector.
 *
 * Downscaling is UNIFORM, so every box/landmark ratio is preserved: detection and cropping
 * both run on the same working image, and ScanMatch stores box + imageWidth/Height in that
 * same space, so any fraction-based overlay is unaffected.
 */
import ImageEditor from '@react-native-community/image-editor';
import { Image as RNImage } from 'react-native';
import RNFS from 'react-native-fs';

const noop = async () => {};

function toUri(p: string): string {
  return p.startsWith('file://') ? p : `file://${p}`;
}

function imageSize(uri: string): Promise<{ width: number; height: number }> {
  return new Promise(resolve =>
    RNImage.getSize(uri, (width, height) => resolve({ width, height }), () => resolve({ width: 0, height: 0 })),
  );
}

export interface Downscaled {
  path: string; // no file:// prefix
  cleanup: () => Promise<void>;
}

/** Working copy capped to `maxDim` on the long side. Returns the original path untouched
 *  when maxDim<=0 or the image is already within the cap, or if the resize fails (never
 *  drops a photo over a resize error). */
export async function downscaleForScan(path: string, maxDim: number): Promise<Downscaled> {
  if (!maxDim || maxDim <= 0) return { path, cleanup: noop };
  const uri = toUri(path);
  const { width, height } = await imageSize(uri);
  if (!width || !height || Math.max(width, height) <= maxDim) return { path, cleanup: noop };

  const scale = maxDim / Math.max(width, height);
  const dw = Math.max(1, Math.round(width * scale));
  const dh = Math.max(1, Math.round(height * scale));
  try {
    const res = (await ImageEditor.cropImage(uri, {
      offset: { x: 0, y: 0 },
      size: { width, height },
      displaySize: { width: dw, height: dh },
      resizeMode: 'cover',
      format: 'jpeg',
      quality: 0.92,
    })) as { uri: string } | string;
    const outUri = typeof res === 'string' ? res : res.uri;
    const outPath = outUri.startsWith('file://') ? outUri.slice(7) : outUri;
    return { path: outPath, cleanup: () => RNFS.unlink(outPath).catch(noop) };
  } catch {
    return { path, cleanup: noop };
  }
}
