/**
 * store — app-wide state shared across the Model / Scan / Results screens:
 * selected model, the two reference buckets, scan settings, live progress, and the
 * history of completed runs (so models can be compared across scans).
 */
import React, { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';
import { MODEL_REGISTRY } from '../models/registry';
import { runScan } from './scanner';
import type {
  ModelSpec,
  RefBucket,
  ReferenceImage,
  ScanProgress,
  ScanRun,
  ScanSettings,
} from '../types';

const DEFAULT_SETTINGS: ScanSettings = {
  maxPhotos: 200,
  threshold: 0.5,
  align: 'spec',
  minFaceSize: 40,
  detector: 'mlkit',
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
  runs: ScanRun[];
  latestRun: ScanRun | null;
  startScan: (photoUris: string[]) => Promise<ScanRun | null>;
  cancelScan: () => void;
  clearRuns: () => void;
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

  const startScan = useCallback(
    async (photoUris: string[]): Promise<ScanRun | null> => {
      if (scanning) return null;
      cancelRef.current = false;
      setScanning(true);
      setProgress(null);
      try {
        const references = [...babyRefs, ...parentRefs];
        const run = await runScan({
          spec: selectedModel,
          references,
          photoUris,
          settings,
          onProgress: setProgress,
          shouldCancel: () => cancelRef.current,
        });
        setRuns(prev => [run, ...prev]);
        return run;
      } finally {
        setScanning(false);
      }
    },
    [scanning, babyRefs, parentRefs, selectedModel, settings],
  );

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
    runs,
    latestRun: runs[0] ?? null,
    startScan,
    cancelScan,
    clearRuns,
  };

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
