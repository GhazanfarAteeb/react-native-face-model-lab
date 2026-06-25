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

/** Most-recent gallery photo URIs (ph:// on iOS, file://|content:// on Android). */
export async function recentPhotoUris(limit: number): Promise<string[]> {
  const ok = await ensurePhotoPermission();
  if (!ok) throw new Error('Photo library permission denied.');
  const res = await CameraRoll.getPhotos({ first: limit, assetType: 'Photos' });
  return res.edges.map(e => e.node.image.uri).filter(Boolean);
}
