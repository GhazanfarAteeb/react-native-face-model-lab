/**
 * timing — high-resolution clock + distribution summary.
 *
 * We report a full distribution (median / p90 / p99 / std), not just a mean, because
 * on-device inference latency is noisy (GC pauses, thermal throttling, scheduler) and
 * the median + tail are what actually matter for "100s of pics in 10s".
 */
import type { TimingStats } from '../types';

/** Prefer performance.now() (sub-ms, monotonic, provided by Hermes); fall back to
 *  Date.now() where it isn't available. */
export const now: () => number =
  typeof globalThis !== 'undefined' &&
  (globalThis as { performance?: { now?: () => number } }).performance?.now
    ? () => (globalThis as unknown as { performance: { now: () => number } }).performance.now()
    : () => Date.now();

/** Time a synchronous or async function once, returning [result, elapsedMs]. */
export async function timed<T>(fn: () => Promise<T> | T): Promise<[T, number]> {
  const t0 = now();
  const r = await fn();
  return [r, now() - t0];
}

export function summarize(samplesIn: number[]): TimingStats | null {
  const samples = samplesIn.filter(x => Number.isFinite(x));
  if (samples.length === 0) return null;
  const sorted = [...samples].sort((a, b) => a - b);
  const n = sorted.length;
  const sum = sorted.reduce((a, b) => a + b, 0);
  const mean = sum / n;
  const variance = sorted.reduce((a, b) => a + (b - mean) * (b - mean), 0) / n;
  const pct = (p: number) => sorted[Math.min(n - 1, Math.max(0, Math.round((p / 100) * (n - 1))))];
  return {
    n,
    mean,
    median: pct(50),
    p90: pct(90),
    p99: pct(99),
    min: sorted[0],
    max: sorted[n - 1],
    std: Math.sqrt(variance),
  };
}

/**
 * Micro-benchmark a single operation: `warmup` untimed runs (let JIT/caches/NNAPI
 * warm up), then `runs` timed runs. Returns the timing distribution.
 */
export async function microBench(
  fn: () => Promise<unknown> | unknown,
  warmup: number,
  runs: number,
): Promise<TimingStats | null> {
  for (let i = 0; i < warmup; i++) await fn();
  const samples: number[] = [];
  for (let i = 0; i < runs; i++) {
    const t0 = now();
    await fn();
    samples.push(now() - t0);
  }
  return summarize(samples);
}
