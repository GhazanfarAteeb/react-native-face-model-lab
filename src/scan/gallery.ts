/**
 * gallery — photo-library access: permission + recent-photo enumeration. References are
 * chosen from detected faces (see facePicker), and scanning walks the recent gallery, so
 * both reference selection and scanning use the camera roll — no system image picker.
 */
import { Platform, PermissionsAndroid } from 'react-native';
import { CameraRoll } from '@react-native-camera-roll/camera-roll';

export async function ensurePhotoPermission(): Promise<boolean> {
  if (Platform.OS !== 'android') return true; // iOS prompts on first access
  const sdk = typeof Platform.Version === 'number' ? Platform.Version : parseInt(String(Platform.Version), 10);
  const perm =
    sdk >= 33
      ? PermissionsAndroid.PERMISSIONS.READ_MEDIA_IMAGES
      : PermissionsAndroid.PERMISSIONS.READ_EXTERNAL_STORAGE;
  const res = await PermissionsAndroid.request(perm);
  return res === PermissionsAndroid.RESULTS.GRANTED;
}

/** How many assets to request per CameraRoll page when paginating. Large enough to keep the
 *  number of native round-trips low, small enough to avoid one giant allocation. */
const PAGE_SIZE = 1000;

/** Most-recent gallery photo URIs (ph:// on iOS, file://|content:// on Android), newest first.
 *  `limit <= 0` means "the whole gallery" — pages through every asset via the CameraRoll
 *  cursor. A positive `limit` still pages (in case the library caps `first` per call) but
 *  stops once it has enough. */
export async function recentPhotoUris(limit: number): Promise<string[]> {
  const ok = await ensurePhotoPermission();
  if (!ok) throw new Error('Photo library permission denied.');

  const all = limit <= 0;
  const uris: string[] = [];
  let after: string | undefined;
  for (;;) {
    const first = all ? PAGE_SIZE : Math.min(PAGE_SIZE, limit - uris.length);
    if (first <= 0) break;
    const res = await CameraRoll.getPhotos({ first, after, assetType: 'Photos' });
    for (const e of res.edges) {
      const u = e.node.image.uri;
      if (u) uris.push(u);
    }
    const info = res.page_info;
    if (!info?.has_next_page || !info.end_cursor) break;
    after = info.end_cursor;
    if (!all && uris.length >= limit) break;
  }
  return all ? uris : uris.slice(0, limit);
}
