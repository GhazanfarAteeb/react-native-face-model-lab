/**
 * store — app-wide state shared across the Model / Scan / Results screens:
 * selected model, the two reference buckets, scan settings, live progress, and the
 * history of completed runs (so models can be compared across scans).
 */
import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { AppState } from 'react-native';
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
import { startBackgroundScan, updateBackgroundScan, stopBackgroundScan } from '../runtime/BackgroundScan';
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
  androidBackend: 'cpu', // CPU is always stable; 'nnapi' opts into the GPU/NPU accelerator EP
  backgroundScan: true, // keep scanning when backgrounded (Android FGS) + auto-resume after a kill
};

function dedupe(items: ReferenceImage[]): ReferenceImage[] {
  const seen = new Set<string>();
  return items.filter(i => (seen.has(i.uri) ? false : (seen.add(i.uri), true)));
}

/** The resumable progress slice of a checkpoint (what runScan needs to continue). */
function progressOf(cp: ScanCheckpoint): ScanProgressState {
  return {
    doneIndices: cp.doneIndices,
    matches: cp.matches,
    facesFound: cp.facesFound,
    sepSum: cp.sepSum,
    sepCount: cp.sepCount,
  };
}

/** The scan config a checkpoint ran with, resolved back to a runnable form — or null for an
 *  old-format checkpoint that didn't persist its config (then resume uses the current config). */
function configOf(cp: ScanCheckpoint): { spec: ModelSpec; settings: ScanSettings; references: ReferenceImage[] } | null {
  if (!cp.modelId || !cp.settings || !cp.references) return null;
  const spec = MODEL_REGISTRY.find(m => m.id === cp.modelId);
  if (!spec) return null;
  return { spec, settings: cp.settings, references: cp.references };
}

function isIncomplete(cp: ScanCheckpoint): boolean {
  return cp.doneIndices.length < cp.photoUris.length;
}

type ScanConfig = { spec: ModelSpec; settings: ScanSettings; references: ReferenceImage[] };
type RunScanCore = (photoUris: string[], resume?: ScanProgressState, override?: ScanConfig) => Promise<ScanRun | null>;

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
  // Refs so the mount + AppState listeners (registered once) always see the latest values.
  const autoResumeTriedRef = useRef(false);
  const scanningRef = useRef(scanning);
  const savedCpRef = useRef<ScanCheckpoint | null>(savedCp);
  const runScanCoreRef = useRef<RunScanCore | null>(null);
  scanningRef.current = scanning;
  savedCpRef.current = savedCp;

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
    async (
      photoUris: string[],
      resume?: ScanProgressState,
      override?: { spec: ModelSpec; settings: ScanSettings; references: ReferenceImage[] },
    ): Promise<ScanRun | null> => {
      if (scanning) return null;
      cancelRef.current = false;
      liveEvents.current = [];
      setLiveVersion(v => v + 1);
      setScanning(true);
      setProgress(null);

      // Resume-from-checkpoint passes the config the scan originally ran with (references, model,
      // settings survive a cold start via the checkpoint); a fresh scan uses the current UI state.
      const spec = override?.spec ?? selectedModel;
      const activeSettings = override?.settings ?? settings;
      const references = override?.references ?? [...babyRefs, ...parentRefs];
      const signature = signatureOf(spec, activeSettings, references);
      const startedAt = Date.now();

      // Keep the scan alive when backgrounded (Android foreground service; no-op on iOS / when the
      // native module is absent) — only when the user opted into background scanning. Fire-and-
      // forget so scanning starts without waiting on the notification-permission prompt.
      if (activeSettings.backgroundScan) {
        startBackgroundScan('Scanning photos', `Starting ${spec.label}…`).catch(() => {});
      }
      // Refresh the notification text on progress, throttled — Android rate-limits notifications.
      let lastNotif = 0;
      const notifyProgress = (p: ScanProgress) => {
        const t = Date.now();
        if (t - lastNotif < 1000) return;
        lastNotif = t;
        const text = p.photoCount
          ? `${p.photoIndex}/${p.photoCount} photos · ${p.matchesSoFar} matches`
          : p.message;
        updateBackgroundScan(text);
      };

      try {
        const run = await runScan({
          spec,
          references,
          photoUris,
          settings: activeSettings,
          onProgress: p => {
            setProgress(p);
            notifyProgress(p);
          },
          onPhoto: pushLive,
          resume,
          onCheckpoint: state => {
            // Persist the full config alongside progress so an OS-killed background scan can be
            // reconstructed and auto-resumed on the next cold start (see attemptAutoResume).
            const cp: ScanCheckpoint = {
              version: 1,
              signature,
              startedAt,
              photoUris,
              modelId: spec.id,
              settings: activeSettings,
              references,
              autoResumable: true,
              ...state,
            };
            lastCp.current = cp;
            queueCheckpoint(cp);
          },
          shouldCancel: () => cancelRef.current,
        });
        setRuns(prev => [run, ...prev]);
        // Cancelled → keep the checkpoint so it can be resumed MANUALLY, but flag it so it is
        // never AUTO-resumed on launch (the user chose to stop). Finished/errored → drop it.
        // Use the in-memory snapshot (race-free vs the async disk write).
        if (cancelRef.current && !run.error) {
          const finalCp = lastCp.current ? { ...lastCp.current, autoResumable: false } : null;
          if (finalCp) await saveCheckpoint(finalCp);
          setSavedCp(finalCp);
        } else {
          await clearCheckpoint();
          setSavedCp(null);
        }
        return run;
      } finally {
        stopBackgroundScan(); // tear down the foreground service + wake lock
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
    // Prefer the checkpoint's own persisted config (survives a cold start); fall back to the
    // current in-memory config for old-format checkpoints.
    return runScanCore(cp.photoUris, progressOf(cp), configOf(cp) ?? undefined);
  }, [savedCp, runScanCore]);

  // A checkpoint is resumable if it isn't complete and we can reconstruct its config — either it
  // persisted its own (new format, works after a cold start) or the current in-memory config
  // still matches its signature (old format).
  const resumable = useMemo(() => {
    if (!savedCp || scanning || !isIncomplete(savedCp)) return null;
    const hasOwnConfig = configOf(savedCp) != null;
    if (!hasOwnConfig) {
      const sig = signatureOf(selectedModel, settings, [...babyRefs, ...parentRefs]);
      if (sig !== savedCp.signature) return null;
    }
    return { done: savedCp.doneIndices.length, total: savedCp.photoUris.length };
  }, [savedCp, scanning, selectedModel, settings, babyRefs, parentRefs]);

  runScanCoreRef.current = runScanCore;

  // Put a killed background scan's config back into the UI so a resume shows the right refs/model.
  const restoreFromCheckpoint = useCallback((cp: ScanCheckpoint) => {
    if (cp.modelId) setSelectedModelId(cp.modelId);
    if (cp.settings) setSettingsState(cp.settings);
    if (cp.references) {
      setBabyRefs(cp.references.filter(r => r.bucket === 'baby'));
      setParentRefs(cp.references.filter(r => r.bucket === 'parent'));
    }
  }, []);

  // Auto-resume an interrupted scan (OS-killed while backgrounded) once — on launch or when the
  // app next returns to the foreground. Only for scans that opted into background mode and weren't
  // explicitly cancelled; a user-cancelled scan stays behind the manual "Resume" card.
  const attemptAutoResume = useCallback((cpArg?: ScanCheckpoint | null) => {
    if (autoResumeTriedRef.current) return;
    const cp = cpArg ?? savedCpRef.current;
    if (!cp || scanningRef.current) return;
    if (cp.autoResumable === false || !cp.settings?.backgroundScan || !isIncomplete(cp)) return;
    const config = configOf(cp);
    if (!config) return; // old-format checkpoint without persisted config → needs manual resume
    autoResumeTriedRef.current = true;
    runScanCoreRef.current?.(cp.photoUris, progressOf(cp), config).catch(() => {});
  }, []);

  // On launch: load any interrupted scan, restore its config into the UI, and auto-resume it if
  // eligible (otherwise it waits behind the manual Resume card).
  useEffect(() => {
    loadCheckpoint().then(cp => {
      setSavedCp(cp);
      if (!cp) return;
      if (configOf(cp)) restoreFromCheckpoint(cp);
      attemptAutoResume(cp);
    });
  }, [restoreFromCheckpoint, attemptAutoResume]);

  // Re-check on every return to the foreground — covers a checkpoint that loaded after mount, or a
  // process the OS killed while backgrounded and relaunched into the foreground.
  useEffect(() => {
    const sub = AppState.addEventListener('change', s => {
      if (s === 'active') attemptAutoResume();
    });
    return () => sub.remove();
  }, [attemptAutoResume]);

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
