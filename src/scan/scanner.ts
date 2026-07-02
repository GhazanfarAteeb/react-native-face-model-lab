/**
 * scanner — runs ONE model over a set of photos and produces a comparable ScanRun.
 *
 * Flow: load model → embed references (baby + parent, separately) → for each photo,
 * embed every face and cosine-match it against both buckets → collect matches above
 * threshold, ranked by similarity, plus per-stage timing and a baby/parent separation
 * proxy. Errors on individual photos are skipped (one unreadable photo never aborts a
 * 300-photo scan); a fatal error (e.g. missing model file) returns an errored run.
 */
import { createEmbedder } from '../runtime/Embedder';
import type { Embedder } from '../runtime/Embedder';
import { createDetector, type Detector } from '../pipeline/detector';
import { embedImage, emptySink, type EmbedCacheCtx, type TimingSink } from './embedFace';
import { normalizeUri } from '../pipeline/normalizeUri';
import { downscaleForScan } from '../pipeline/downscale';
import { makeThumb } from '../pipeline/thumbnail';
import { cosineSimilarity } from '../pipeline/similarity';
import { runPool, withTimeout } from './concurrency';
import { emptyCacheStats, setCacheEnabled } from './scanCache';
import { resolveConcurrency } from './deviceProfile';
import { FrameBudget } from './scheduler';
import type { ScanProgressState } from './checkpoint';
import { now, summarize } from '../bench/timing';
import type {
  AlignMode,
  ModelSpec,
  RefBucket,
  ReferenceEmbedding,
  ReferenceImage,
  ScanMatch,
  ScanPhotoEvent,
  ScanProgress,
  ScanRun,
  ScanSettings,
} from '../types';

function resolveAlign(spec: ModelSpec, settings: ScanSettings): AlignMode {
  if (settings.align === 'force-arcface') return 'arcface';
  if (settings.align === 'force-bbox') return 'bbox';
  return spec.align;
}

function maxSim(e: Float32Array, refs: ReferenceEmbedding[]): number {
  let m = -1;
  for (const r of refs) {
    const s = cosineSimilarity(e, r.embedding);
    if (s > m) m = s;
  }
  return m;
}

export interface RunScanParams {
  spec: ModelSpec;
  references: ReferenceImage[];
  photoUris: string[];
  settings: ScanSettings;
  onProgress?: (p: ScanProgress) => void;
  onPhoto?: (e: ScanPhotoEvent) => void;
  /** Prior progress to continue from (skips already-done photos, keeps prior matches). */
  resume?: ScanProgressState;
  /** Periodic snapshot of resumable progress; persist this to survive interruption. */
  onCheckpoint?: (state: ScanProgressState) => void;
  shouldCancel?: () => boolean;
}

/** Persist a checkpoint roughly every this-many newly-processed photos. */
const CHECKPOINT_EVERY = 20;

/** Hard ceiling for one photo's detect+crop+embed. A malformed image that hangs a native
 *  call would otherwise wedge a worker; on timeout the photo is counted as error + skipped. */
const PER_PHOTO_TIMEOUT_MS = 30_000;

/** Per-reference face counts so the UI can flag refs where no face was found. */
export type RefFaceCounts = Record<string, number>;

async function buildRefs(
  references: ReferenceImage[],
  embedder: Embedder,
  spec: ModelSpec,
  align: AlignMode,
  minFaceSize: number,
  sink: TimingSink,
  detector: Detector,
  maxImageDim: number,
): Promise<{ embeddings: ReferenceEmbedding[]; counts: RefFaceCounts }> {
  const embeddings: ReferenceEmbedding[] = [];
  const counts: RefFaceCounts = {};
  for (const ref of references) {
    const norm = await normalizeUri(ref.uri, maxImageDim);
    // Same working resolution as scanned photos, so reference embeddings match.
    const work = norm.downscaled
      ? { path: norm.path, cleanup: async () => {} }
      : await downscaleForScan(norm.path, maxImageDim);
    try {
      const faces = await embedImage(async () => work.path, embedder, spec, align, minFaceSize, sink, detector);
      counts[ref.uri] = faces.length;
      if (faces.length) {
        // Primary subject = largest face (matches rnbaby's reference convention).
        let best = faces[0];
        for (const f of faces) {
          if (f.box.width * f.box.height > best.box.width * best.box.height) best = f;
        }
        embeddings.push({ bucket: ref.bucket, uri: ref.uri, embedding: best.embedding });
      }
    } catch {
      counts[ref.uri] = 0;
    } finally {
      await work.cleanup();
      await norm.cleanup();
    }
  }
  return { embeddings, counts };
}

let runCounter = 0;

function erroredRun(
  id: string,
  spec: ModelSpec,
  detectorLabel: string,
  startedAt: number,
  durationMs: number,
  error: string,
): ScanRun {
  return {
    id,
    modelId: spec.id,
    modelLabel: spec.label,
    detectorLabel,
    startedAt,
    durationMs,
    loadMs: 0,
    fileSizeBytes: 0,
    photosScanned: 0,
    facesFound: 0,
    matches: [],
    babyMatchCount: 0,
    parentMatchCount: 0,
    avgMsPerPhoto: 0,
    facesPerSec: 0,
    stages: { decode: null, detect: null, crop: null, preprocess: null, infer: null, similarity: null, total: null },
    separation: 0,
    error,
  };
}

export async function runScan(params: RunScanParams): Promise<ScanRun> {
  const { spec, references, photoUris, settings, onProgress, onPhoto, resume, onCheckpoint, shouldCancel } = params;
  const align = resolveAlign(spec, settings);
  const startedAt = Date.now();
  const wall0 = now();
  runCounter += 1;
  const id = `run_${startedAt}_${runCounter}`;

  const total = photoUris.length;
  const emit = (
    phase: ScanProgress['phase'],
    message: string,
    fraction: number,
    extra?: Partial<ScanProgress>,
  ) =>
    onProgress?.({
      phase,
      message,
      photoIndex: 0,
      photoCount: total,
      facesFound: 0,
      matchesSoFar: 0,
      fraction,
      ...extra,
    });

  let embedder: Embedder | undefined;
  let detector: Detector | undefined;
  const sink = emptySink();
  const refSink = emptySink();

  try {
    emit('loading', `Loading ${spec.label}…`, 0);
    embedder = await createEmbedder(spec, { iosBackends: settings.iosBackends, androidBackend: settings.androidBackend });
    detector = await createDetector(settings.detector, { accurate: !settings.fastDetect, androidBackend: settings.androidBackend });
    const loadMs = embedder.loadMs;
    const fileSizeBytes = embedder.fileSizeBytes;

    emit('references', `Embedding references (${detector.label})…`, 0.02);
    const { embeddings: refEmb } = await buildRefs(
      references,
      embedder,
      spec,
      align,
      settings.minFaceSize,
      refSink,
      detector,
      settings.maxImageDim,
    );
    const babyRefs = refEmb.filter(r => r.bucket === 'baby');
    const parentRefs = refEmb.filter(r => r.bucket === 'parent');
    if (refEmb.length === 0) {
      throw new Error('No face found in any reference image. Pick clearer, front-facing photos.');
    }

    // Seed from a prior checkpoint when resuming; otherwise start clean.
    const done = new Set<number>(resume?.doneIndices ?? []);
    const matches: ScanMatch[] = resume?.matches ? [...resume.matches] : [];
    let facesFound = resume?.facesFound ?? 0;
    let processed = done.size;
    let sepSum = resume?.sepSum ?? 0;
    let sepCount = resume?.sepCount ?? 0;

    const snapshot = (): ScanProgressState => ({
      doneIndices: [...done],
      matches: [...matches],
      facesFound,
      sepSum,
      sepCount,
    });

    const mean = (a: number[]): number => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);
    const liveStages = () => ({
      decode: mean(sink.decode),
      detect: mean(sink.detect),
      align: mean(sink.crop),
      preprocess: mean(sink.preprocess),
      embed: mean(sink.infer),
      similarity: mean(sink.similarity),
      total: mean(sink.total),
    });

    // Process several photos at once so each one's native I/O (asset copy), detection and
    // ORT inference overlap with the others' JS work. ORT runs themselves stay serialized
    // (see ortLock), so the speed-up comes from hiding I/O + detection latency, not from
    // running multiple inferences in parallel. The fold below (matches/counters) runs
    // synchronously after each `await`, so it's atomic on JS's single thread.
    // 0 = Auto (device-adaptive); otherwise the user's explicit choice.
    const concurrency = resolveConcurrency(settings.concurrency);
    // Shared across workers (one JS thread) so total contiguous JS time between yields is
    // bounded regardless of how many photos are in flight.
    const frame = new FrameBudget(8);

    // Cross-run cache: reuse detection + crops (model-agnostic) so reruns with other models
    // skip those stages. Stats let Results show the warm-rerun speed-up.
    setCacheEnabled(settings.reuseCache);
    const cacheStats = emptyCacheStats();
    const cacheCtx: EmbedCacheCtx = {
      // detectorKind drives the cache key; the ML Kit fast/accurate mode changes which faces
      // are found AND their landmarks (→ crops), so it must be part of the key or a mode
      // switch would wrongly reuse the other mode's cached detections/crops.
      detectorKind: settings.detector === 'mlkit' && settings.fastDetect ? 'mlkit-fast' : settings.detector,
      maxImageDim: settings.maxImageDim,
      uri: '',
      stats: cacheStats,
    };

    const scanPhoto = async (i: number): Promise<void> => {
      if (done.has(i)) return; // already processed in a previous (resumed) run
      const uri = photoUris[i];
      const t0 = now();
      let faceCount = 0;
      let bestSim = -1;
      let bestBucket: RefBucket = 'baby';
      let thumbUri: string | undefined;

      // Lazy, memoized decode: the working image is produced only the first time something
      // actually needs it (a detection or crop cache MISS, or a missing thumbnail). On a warm
      // rerun where detection + every crop hit and the thumbnail exists, this never runs — so
      // the biggest per-photo stage (decode, ~230-320ms) is skipped entirely.
      let decoded: { path: string; cleanup: () => Promise<void> } | undefined;
      const getWork = async (): Promise<string> => {
        if (decoded) return decoded.path;
        const dec0 = now();
        // Ask the source for an already-downscaled copy (iOS PhotoKit); only fall back to a
        // separate ImageEditor downscale when it didn't (Android content://, local files).
        const norm = await normalizeUri(uri, settings.maxImageDim);
        const work = norm.downscaled
          ? { path: norm.path, cleanup: async () => {} }
          : await downscaleForScan(norm.path, settings.maxImageDim);
        decoded = {
          path: work.path,
          cleanup: async () => {
            await work.cleanup();
            await norm.cleanup();
          },
        };
        sink.decode.push(now() - dec0);
        return decoded.path;
      };

      try {
        // Per-photo cache context (same stats object, this photo's uri) + hard timeout so a
        // hung native call on a malformed image can't wedge this worker forever.
        const faces = await withTimeout(
          embedImage(getWork, embedder!, spec, align, settings.minFaceSize, sink, detector!, {
            ...cacheCtx,
            uri,
          }),
          PER_PHOTO_TIMEOUT_MS,
          'embedImage',
        );
        faceCount = faces.length;
        const sim0 = now();
        for (const f of faces) {
          facesFound += 1;
          const simBaby = babyRefs.length ? maxSim(f.embedding, babyRefs) : -1;
          const simParent = parentRefs.length ? maxSim(f.embedding, parentRefs) : -1;
          if (babyRefs.length && parentRefs.length) {
            sepSum += Math.abs(simBaby - simParent);
            sepCount += 1;
          }
          const bucket: RefBucket = simBaby >= simParent ? 'baby' : 'parent';
          const sim = Math.max(simBaby, simParent);
          if (sim > bestSim) {
            bestSim = sim;
            bestBucket = bucket;
          }
          if (sim >= settings.threshold) {
            matches.push({
              uri,
              bucket,
              similarity: sim,
              faceBox: f.box,
              imageWidth: f.imageWidth,
              imageHeight: f.imageHeight,
            });
          }
        }
        if (faces.length) sink.similarity.push(now() - sim0);
        // Small thumbnail for the live grid. Cached by photo uri across runs, so a rerun
        // reuses it without decoding; only a first encounter calls getWork. Best-effort.
        thumbUri = await makeThumb(uri, getWork);
      } catch {
        // Unreadable / undecodable photo — skip it.
      } finally {
        if (decoded) await decoded.cleanup();
      }
      done.add(i);
      processed += 1;
      sink.total.push(now() - t0);

      if (processed % CHECKPOINT_EVERY === 0) onCheckpoint?.(snapshot());

      const matched = bestSim >= settings.threshold;
      onPhoto?.({
        index: i,
        uri,
        thumbUri,
        faceCount,
        matched,
        bucket: matched ? bestBucket : undefined,
        similarity: bestSim,
        ms: now() - t0,
      });

      if (processed % 3 === 0 || processed === total) {
        emit('scanning', `Scanning ${processed}/${total}…`, 0.05 + 0.95 * (processed / Math.max(1, total)), {
          photoIndex: processed,
          facesFound,
          matchesSoFar: matches.length,
          liveStages: liveStages(),
        });
      }

      // Breathe: hand the JS thread back to the UI if we've held it past the frame budget.
      await frame.tick();
    };

    // Optional time-budget: stop pulling new photos once the limit elapses (throughput test).
    // In-flight photos finish; the result reports how many were scanned in the window + faces/s.
    const deadline = settings.scanTimeLimitSec > 0 ? wall0 + settings.scanTimeLimitSec * 1000 : Infinity;
    const stop = () => (shouldCancel?.() ?? false) || now() >= deadline;
    await runPool(total, concurrency, scanPhoto, stop);
    onCheckpoint?.(snapshot()); // capture final state (matters on cancel / interruption)

    matches.sort((a, b) => b.similarity - a.similarity);
    const durationMs = now() - wall0;
    await embedder.release();
    await detector.dispose();

    emit('done', 'Done', 1, { photoIndex: processed, facesFound, matchesSoFar: matches.length });

    return {
      id,
      modelId: spec.id,
      modelLabel: spec.label,
      detectorLabel: detector.label,
      startedAt,
      durationMs,
      loadMs,
      fileSizeBytes,
      photosScanned: processed,
      facesFound,
      matches,
      babyMatchCount: matches.filter(m => m.bucket === 'baby').length,
      parentMatchCount: matches.filter(m => m.bucket === 'parent').length,
      avgMsPerPhoto: processed ? durationMs / processed : 0,
      facesPerSec: durationMs > 0 ? facesFound / (durationMs / 1000) : 0,
      stages: {
        decode: summarize(sink.decode),
        detect: summarize(sink.detect),
        crop: summarize(sink.crop),
        preprocess: summarize(sink.preprocess),
        infer: summarize(sink.infer),
        similarity: summarize(sink.similarity),
        total: summarize(sink.total),
      },
      separation: sepCount ? sepSum / sepCount : 0,
      cacheStats,
    };
  } catch (err) {
    if (embedder) await embedder.release().catch(() => {});
    if (detector) await detector.dispose().catch(() => {});
    const message = err instanceof Error ? err.message : String(err);
    emit('error', message, 1);
    const dLabel = detector?.label ?? (settings.detector === 'yunet' ? 'YuNet' : 'ML Kit');
    return erroredRun(id, spec, dLabel, startedAt, now() - wall0, message);
  }
}
