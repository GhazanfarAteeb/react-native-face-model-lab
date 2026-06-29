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

let cached: number | null = null;

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
  // Capped at 4: beyond that the gain is small but in-flight memory (temp files + decoded
  // buffers) and native ImageEditor contention grow, which on heavy arcface models causes
  // GC stalls. Pick a higher level manually if your device proves it can take it.
  if (ms < 25) c = 4; // fast / flagship
  else if (ms < 60) c = 3; // mid-range
  else c = 2; // low-end — stay smooth
  cached = c;
  console.log(`[FML] device JS bench ${ms}ms → auto concurrency ×${c}`);
  return c;
}

/** Resolve a settings value (0 = Auto) to an effective concurrency. */
export function resolveConcurrency(setting: number): number {
  return setting > 0 ? setting : recommendedConcurrency();
}
