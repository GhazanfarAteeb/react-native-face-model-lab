/**
 * BackgroundScan — JS handle to the Android scan foreground service (BackgroundScanModule).
 *
 * Starting the service keeps the process + CPU alive (partial wake lock) so the scan keeps
 * running when the app is backgrounded or the screen is off; results still stream into the
 * store and render live when the app returns to the foreground. No-ops on iOS (the app stays
 * foreground there) and whenever the native module isn't in the binary, so callers don't need
 * platform guards.
 */
import { NativeModules, PermissionsAndroid, Platform } from 'react-native';

interface BackgroundScanNative {
  /** Start the foreground service + wake lock. Must be called while the app is foreground. */
  start(title: string, text: string): Promise<boolean>;
  /** Refresh the notification text (throttle callers — Android rate-limits notifications). */
  updateProgress(text: string): void;
  /** Stop the service and release the wake lock. */
  stop(): void;
}

const native: BackgroundScanNative | null =
  Platform.OS === 'android'
    ? ((NativeModules as Record<string, unknown>).BackgroundScan as BackgroundScanNative | undefined) ?? null
    : null;

let active = false;

/** True when a background scan service is available on this platform/binary. */
export function backgroundScanAvailable(): boolean {
  return native != null;
}

async function ensureNotificationPermission(): Promise<void> {
  // Android 13+ (API 33) gates the visible notification behind a runtime permission. The service
  // still runs if denied — the system shows its own minimal FGS notice — so this is best-effort.
  if (Platform.OS !== 'android' || (Platform.Version as number) < 33) return;
  try {
    await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.POST_NOTIFICATIONS);
  } catch {
    // ignore — proceed without the visible notification
  }
}

export async function startBackgroundScan(title: string, text: string): Promise<void> {
  if (!native || active) return;
  await ensureNotificationPermission();
  try {
    await native.start(title, text);
    active = true;
  } catch {
    active = false; // couldn't start (e.g. app already background) — scan continues foreground-only
  }
}

export function updateBackgroundScan(text: string): void {
  if (!native || !active) return;
  try {
    native.updateProgress(text);
  } catch {
    // ignore transient bridge errors
  }
}

export function stopBackgroundScan(): void {
  if (!native || !active) return;
  try {
    native.stop();
  } finally {
    active = false;
  }
}
