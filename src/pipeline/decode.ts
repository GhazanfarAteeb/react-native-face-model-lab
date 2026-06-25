/**
 * decode — JPEG → RGBA pixels, pure JS (jpeg-js).
 *
 * We only ever decode SMALL images here: every crop is produced by the native
 * ImageEditor at the model's input size (≤160²) or a modest working size, so the
 * pure-JS decode is a couple of milliseconds and never touches the full-resolution
 * photo. This keeps the native dependency list short (no Skia) while staying fast.
 */
import { Buffer } from 'buffer';
import RNFS from 'react-native-fs';
import { decode as jpegDecode } from 'jpeg-js';
import type { RgbaImage } from '../types';

export async function decodeJpeg(filePath: string): Promise<RgbaImage> {
  const path = filePath.startsWith('file://') ? filePath.slice(7) : filePath;
  const b64 = await RNFS.readFile(path, 'base64');
  const buf = Buffer.from(b64, 'base64');
  const { width, height, data } = jpegDecode(buf, { useTArray: true, formatAsRGBA: true });
  return { width, height, data: data as Uint8Array };
}
