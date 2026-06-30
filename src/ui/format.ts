/** Small display formatters shared by the screens. */
import type { TimingStats } from '../types';

export function fmtBytes(n: number): string {
  if (!n) return '—';
  if (n >= 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${n} B`;
}

export function fmtMs(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—';
  if (n >= 1000) return `${(n / 1000).toFixed(2)} s`;
  return `${n.toFixed(n < 10 ? 1 : 0)} ms`;
}

export function fmtMedian(s: TimingStats | null): string {
  return s ? fmtMs(s.median) : '—';
}

/** Break a duration in ms into h / m / s / ms, dropping leading zero units.
 *  e.g. 199920 → "3m 19s 920ms", 45000 → "45s 0ms", 800 → "800ms". */
export function fmtDuration(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms)) return '—';
  const total = Math.max(0, Math.round(ms));
  if (total < 1000) return `${total}ms`;
  const h = Math.floor(total / 3_600_000);
  const m = Math.floor((total % 3_600_000) / 60_000);
  const s = Math.floor((total % 60_000) / 1000);
  const msPart = total % 1000;
  const parts: string[] = [];
  if (h) parts.push(`${h}h`);
  if (h || m) parts.push(`${m}m`);
  parts.push(`${s}s`);
  parts.push(`${msPart}ms`);
  return parts.join(' ');
}

export function pct(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
}
