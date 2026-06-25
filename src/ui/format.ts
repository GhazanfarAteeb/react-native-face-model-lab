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

export function pct(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
}
