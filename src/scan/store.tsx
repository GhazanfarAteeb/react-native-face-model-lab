/**
 * store — app-wide state shared across the Model / Scan / Results screens:
 * selected model, the two reference buckets, scan settings, live progress, and the
 * history of completed runs (so models can be compared across scans).
 */
import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { MODEL_REGISTRY } from '../models/registry';
import { runScan } from './scanner';
import {
  clearCheckpoint,
  loadCheckpoint,
  saveCheckpoint,
  signatureOf,
  type ScanCheckpoint,
  type ScanProgressState,
} from './checkpoint';
import { cacheSizes, clearScanCache } from './scanCache';
import { clearThumbs } from '../pipeline/thumbnail';
import type {
  ModelSpec,
  RefBucket,
  ReferenceImage,
  ScanPhotoEvent,
  ScanProgress,
  ScanRun,
  ScanSettings,
} from '../types';

const DEFAULT_SETTINGS: ScanSettings = {
  maxPhotos: 200,
  scanTimeLimitSec: 0, // 0 = scan all selected photos; >0 = stop after N seconds (throughput test)
  threshold: 0.5,
  align: 'spec',
  minFaceSize: 40,
  detector: 'mlkit',
  concurrency: 0, // Auto — device-adaptive
  maxImageDim: 1600, // cap working resolution for faster detection
  reuseCache: true, // reuse detection + crops across runs (benchmark re-scans)
  fastDetect: false, // ML Kit accurate mode by default
  iosBackends: ['cpu'], // CPU beats the (non-ANE) CoreML EP; select both for ANE hybrid
};

function dedupe(items: ReferenceImage[]): ReferenceImage[] {
  const seen = new Set<string>();
  return items.filter(i => (seen.has(i.uri) ? false : (seen.add(i.uri), true)));
}

interface StoreValue {
  models: ModelSpec[];
  selectedModelId: string;
  selectedModel: ModelSpec;
  selectModel: (id: string) => void;

  babyRefs: ReferenceImage[];
  parentRefs: ReferenceImage[];
  addRefs: (bucket: RefBucket, uris: string[]) => void;
  removeRef: (bucket: RefBucket, uri: string) => void;
  clearRefs: (bucket: RefBucket) => void;

  settings: ScanSettings;
  setSettings: (patch: Partial<ScanSettings>) => void;

  scanning: boolean;
  progress: ScanProgress | null;
  /** Live per-photo stream for the scanning screen. Mutated in place + versioned so we
   *  can render a fast-growing grid without one React commit per photo. */
  liveEvents: React.MutableRefObject<ScanPhotoEvent[]>;
  liveVersion: number;
  runs: ScanRun[];
  latestRun: ScanRun | null;
  startScan: (photoUris: string[]) => Promise<ScanRun | null>;
  /** A resumable interrupted scan matching the current config, or null. */
  resumable: { done: number; total: number } | null;
  resumeScan: () => Promise<ScanRun | null>;
  cancelScan: () => void;
  clearRuns: () => void;
  /** Drop the in-memory detection/crop cache (reset to a cold benchmark). Returns sizes. */
  clearCache: () => { detect: number; crop: number };
  cacheSizes: () => { detect: number; crop: number };
}

const Ctx = createContext<StoreValue | null>(null);

export function useStore(): StoreValue {
  const v = useContext(Ctx);
  if (!v) throw new Error('useStore must be used inside <StoreProvider>');
  return v;
}

export function StoreProvider({ children }: { children: React.ReactNode }) {
  const [selectedModelId, setSelectedModelId] = useState('mobilefacenet');
  const [babyRefs, setBabyRefs] = useState<ReferenceImage[]>([]);
  const [parentRefs, setParentRefs] = useState<ReferenceImage[]>([]);
  const [settings, setSettingsState] = useState<ScanSettings>(DEFAULT_SETTINGS);
  const [scanning, setScanning] = useState(false);
  const [progress, setProgress] = useState<ScanProgress | null>(null);
  const [runs, setRuns] = useState<ScanRun[]>([]);
  const cancelRef = useRef(false);
  const liveEvents = useRef<ScanPhotoEvent[]>([]);
  const [liveVersion, setLiveVersion] = useState(0);
  const flushScheduled = useRef(false);
  const [savedCp, setSavedCp] = useState<ScanCheckpoint | null>(null);
  const cpSaving = useRef(false);
  const cpPending = useRef<ScanCheckpoint | null>(null);
  const lastCp = useRef<ScanCheckpoint | null>(null);

  // Look for an interrupted scan once on mount.
  useEffect(() => {
    loadCheckpoint().then(setSavedCp);
  }, []);

  const selectedModel = useMemo(
    () => MODEL_REGISTRY.find(m => m.id === selectedModelId) ?? MODEL_REGISTRY[0],
    [selectedModelId],
  );

  const addRefs = useCallback((bucket: RefBucket, uris: string[]) => {
    const items: ReferenceImage[] = uris.map(uri => ({ uri, bucket, faceCount: null }));
    const setter = bucket === 'baby' ? setBabyRefs : setParentRefs;
    setter(prev => dedupe([...prev, ...items]));
  }, []);

  const removeRef = useCallback((bucket: RefBucket, uri: string) => {
    const setter = bucket === 'baby' ? setBabyRefs : setParentRefs;
    setter(prev => prev.filter(r => r.uri !== uri));
  }, []);

  const clearRefs = useCallback((bucket: RefBucket) => {
    (bucket === 'baby' ? setBabyRefs : setParentRefs)([]);
  }, []);

  const setSettings = useCallback((patch: Partial<ScanSettings>) => {
    setSettingsState(prev => ({ ...prev, ...patch }));
  }, []);

  // Buffer per-photo events into the ref and commit a version bump at most ~10×/sec, so a
  // 1000-photo scan doesn't trigger 1000 React renders of the live grid.
  const pushLive = useCallback((e: ScanPhotoEvent) => {
    liveEvents.current.push(e);
    if (!flushScheduled.current) {
      flushScheduled.current = true;
      setTimeout(() => {
        flushScheduled.current = false;
        setLiveVersion(v => v + 1);
      }, 100);
    }
  }, []);

  // Single-flight checkpoint writer: keep only the newest snapshot and never overlap
  // writes, so frequent checkpoints can't pile up or corrupt the file.
  const queueCheckpoint = useCallback((cp: ScanCheckpoint) => {
    cpPending.current = cp;
    if (cpSaving.current) return;
    cpSaving.current = true;
    (async () => {
      try {
        while (cpPending.current) {
          const next = cpPending.current;
          cpPending.current = null;
          await saveCheckpoint(next);
        }
      } finally {
        cpSaving.current = false;
      }
    })().catch(() => {});
  }, []);

  const runScanCore = useCallback(
    async (photoUris: string[], resume?: ScanProgressState): Promise<ScanRun | null> => {
      if (scanning) return null;
      cancelRef.current = false;
      liveEvents.current = [];
      setLiveVersion(v => v + 1);
      setScanning(true);
      setProgress(null);

      const references = [...babyRefs, ...parentRefs];
      const signature = signatureOf(selectedModel, settings, references);
      const startedAt = Date.now();
      try {
        const run = await runScan({
          spec: selectedModel,
          references,
          photoUris,
          settings,
          onProgress: setProgress,
          onPhoto: pushLive,
          resume,
          onCheckpoint: state => {
            const cp: ScanCheckpoint = { version: 1, signature, startedAt, photoUris, ...state };
            lastCp.current = cp;
            queueCheckpoint(cp);
          },
          shouldCancel: () => cancelRef.current,
        });
        setRuns(prev => [run, ...prev]);
        // Cancelled → keep the checkpoint so it can be resumed; finished/errored → drop it.
        // Use the in-memory snapshot (race-free vs the async disk write).
        if (cancelRef.current && !run.error) {
          setSavedCp(lastCp.current);
        } else {
          await clearCheckpoint();
          setSavedCp(null);
        }
        return run;
      } finally {
        setLiveVersion(v => v + 1); // flush any buffered tail
        setScanning(false);
      }
    },
    [scanning, babyRefs, parentRefs, selectedModel, settings, pushLive, queueCheckpoint],
  );

  const startScan = useCallback(
    async (photoUris: string[]): Promise<ScanRun | null> => {
      // Fresh scan: discard any stale checkpoint up front so an immediate kill can't
      // resurface an unrelated one.
      await clearCheckpoint();
      setSavedCp(null);
      return runScanCore(photoUris);
    },
    [runScanCore],
  );

  const resumeScan = useCallback(async (): Promise<ScanRun | null> => {
    const cp = savedCp;
    if (!cp) return null;
    return runScanCore(cp.photoUris, {
      doneIndices: cp.doneIndices,
      matches: cp.matches,
      facesFound: cp.facesFound,
      sepSum: cp.sepSum,
      sepCount: cp.sepCount,
    });
  }, [savedCp, runScanCore]);

  // A checkpoint is resumable only if it matches the CURRENT config and isn't complete.
  const resumable = useMemo(() => {
    if (!savedCp || scanning) return null;
    const sig = signatureOf(selectedModel, settings, [...babyRefs, ...parentRefs]);
    if (sig !== savedCp.signature) return null;
    const done = savedCp.doneIndices.length;
    if (done >= savedCp.photoUris.length) return null;
    return { done, total: savedCp.photoUris.length };
  }, [savedCp, scanning, selectedModel, settings, babyRefs, parentRefs]);

  const cancelScan = useCallback(() => {
    cancelRef.current = true;
  }, []);

  const clearRuns = useCallback(() => setRuns([]), []);

  const value: StoreValue = {
    models: MODEL_REGISTRY,
    selectedModelId,
    selectedModel,
    selectModel: setSelectedModelId,
    babyRefs,
    parentRefs,
    addRefs,
    removeRef,
    clearRefs,
    settings,
    setSettings,
    scanning,
    progress,
    liveEvents,
    liveVersion,
    runs,
    latestRun: runs[0] ?? null,
    startScan,
    resumable,
    resumeScan,
    cancelScan,
    clearRuns,
    // Clearing the benchmark cache also wipes the persisted thumbnails (they're keyed by
    // photo and reused across runs); fire-and-forget so the UI gets sizes synchronously.
    clearCache: () => {
      const sizes = clearScanCache();
      clearThumbs().catch(() => {});
      return sizes;
    },
    cacheSizes,
  };

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
