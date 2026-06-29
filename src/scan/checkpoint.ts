/**
 * checkpoint — persist scan progress so an interrupted scan (cancelled, app backgrounded
 * & killed, OS-terminated mid-run) can resume losslessly instead of starting over.
 *
 * We persist only JSON-safe progress: which photo indices are done, the matches found so
 * far, and the running counters. Reference EMBEDDINGS are NOT stored (they're large,
 * Float32, and model-specific) — they're cheap to recompute on resume from the same
 * reference images. A `signature` pins the checkpoint to the exact config that produced
 * it (model, detector, alignment, threshold, min-face-size, references); if any of those
 * change the checkpoint is no longer valid and is ignored.
 */
import RNFS from 'react-native-fs';
import type { ModelSpec, ReferenceImage, ScanMatch, ScanSettings } from '../types';

const FILE = `${RNFS.DocumentDirectoryPath}/fml_scan_checkpoint.json`;

/** The resumable slice of a scan — produced and consumed by runScan. */
export interface ScanProgressState {
  doneIndices: number[];
  matches: ScanMatch[];
  facesFound: number;
  sepSum: number;
  sepCount: number;
}

export interface ScanCheckpoint extends ScanProgressState {
  version: 1;
  signature: string;
  startedAt: number;
  photoUris: string[];
}

/** A stable string over everything that affects WHICH faces match — anything not in here
 *  (e.g. concurrency) is safe to change between a scan and its resume. */
export function signatureOf(spec: ModelSpec, settings: ScanSettings, references: ReferenceImage[]): string {
  const refs = references
    .map(r => `${r.bucket}:${r.uri}`)
    .sort()
    .join('|');
  return JSON.stringify({
    model: spec.id,
    detector: settings.detector,
    fastDetect: settings.fastDetect, // changes which faces/landmarks are found
    align: settings.align,
    threshold: settings.threshold,
    minFaceSize: settings.minFaceSize,
    refs,
  });
}

export async function saveCheckpoint(cp: ScanCheckpoint): Promise<void> {
  try {
    await RNFS.writeFile(FILE, JSON.stringify(cp), 'utf8');
  } catch {
    // Best-effort: a dropped checkpoint just means a longer resume next time.
  }
}

export async function loadCheckpoint(): Promise<ScanCheckpoint | null> {
  try {
    if (!(await RNFS.exists(FILE))) return null;
    const cp = JSON.parse(await RNFS.readFile(FILE, 'utf8')) as ScanCheckpoint;
    if (cp.version !== 1 || !Array.isArray(cp.photoUris) || !Array.isArray(cp.doneIndices)) return null;
    return cp;
  } catch {
    return null;
  }
}

export async function clearCheckpoint(): Promise<void> {
  try {
    if (await RNFS.exists(FILE)) await RNFS.unlink(FILE);
  } catch {
    // ignore
  }
}
