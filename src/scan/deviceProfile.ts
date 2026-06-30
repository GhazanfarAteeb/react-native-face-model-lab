/**
 * deviceProfile — pick a scan concurrency that fits the device.
 *
 * The right concurrency is device-dependent: a flagship can keep 5–6 photos in flight
 * without the JS thread falling behind, while a low-end phone janks (and risks memory
 * pressure) at the same setting. React Native exposes no CPU/RAM count without a native
 * module, so we infer the device's JS-thread headroom from a one-time micro-benchmark —
 * a proxy for "how much per-photo JS work can this device absorb per frame." Cached for
 * the app session (thermal drift over a single scan isn't worth re-measuring).
 *
 * Thresholds are deliberately conservative — when unsure we pick LOWER concurrency, since
 * a smooth-but-slower scan beats a fast-but-janky one on the weak devices we care about.
 */

import { Platform } from 'react-native';

let cached: number | null = null;

/** Android heap is far tighter than iOS, and each in-flight photo holds native bitmaps across
 *  decode → detect → crop. Past this many concurrent photos the app OOM-crashes regardless of
 *  the chosen level, so we clamp the effective concurrency on Android (iOS is unrestricted).
 *  largeHeap is also enabled in the manifest for extra headroom. */
const ANDROID_MAX_CONCURRENCY = 8;

function benchmarkMs(): number {
  const t0 = Date.now();
  let acc = 0;
  for (let i = 0; i < 2_000_000; i++) {
    acc += Math.sqrt(i * 1.0001) * 0.5;
  }
  const ms = Date.now() - t0;
  // Touch `acc` so the loop can't be optimized away.
  return acc > 0 ? ms : ms + 1;
}

/** Concurrency for `concurrency: 0` (Auto). Benchmarks once, then returns the cached tier. */
export function recommendedConcurrency(): number {
  if (cached != null) return cached;
  const ms = benchmarkMs();
  let c: number;
  // Now that per-photo JS work is small (decode is lazy/cached; detect + crop are native and
  // overlap across workers), throughput is gated by how many native stages run at once — so a
  // higher cap helps. The remaining risk at high concurrency is in-flight memory (temp files +
  // decoded buffers) and GC churn on heavy arcface models; the live health meter flags it, and
  // the user can dial it down. Low-end devices stay conservative.
  if (ms < 25) c = 6; // fast / flagship
  else if (ms < 60) c = 4; // mid-range
  else c = 2; // low-end — stay smooth
  cached = c;
  console.log(`[FML] device JS bench ${ms}ms → auto concurrency ×${c}`);
  return c;
}

/** Resolve a settings value (0 = Auto) to an effective concurrency, clamped on Android to a
 *  memory-safe ceiling so the high experimental levels (×10–×20) don't OOM the device. */
export function resolveConcurrency(setting: number): number {
  const c = setting > 0 ? setting : recommendedConcurrency();
  if (Platform.OS === 'android' && c > ANDROID_MAX_CONCURRENCY) {
    console.log(`[FML] clamping concurrency ×${c} → ×${ANDROID_MAX_CONCURRENCY} (Android memory)`);
    return ANDROID_MAX_CONCURRENCY;
  }
  return c;
}
