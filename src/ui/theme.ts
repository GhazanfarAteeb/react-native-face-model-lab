/** Minimal dark theme for the benchmark tool. Monospace for all numbers so columns
 *  line up and results are easy to scan. */
export const theme = {
  bg: '#0d1117',
  surface: '#161b22',
  surfaceAlt: '#1c2330',
  border: '#2a3038',
  text: '#e6edf3',
  textDim: '#9aa7b4',
  textFaint: '#6e7781',
  accent: '#3b82f6',
  accentDim: '#1e3a8a',
  good: '#3fb950',
  warn: '#d29922',
  bad: '#f85149',
  chip: '#21262d',
} as const;

export const spacing = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24 } as const;

export const mono = 'Menlo';

export const radius = { sm: 6, md: 10, lg: 14 } as const;
