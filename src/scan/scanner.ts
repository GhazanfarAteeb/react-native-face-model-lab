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
import { embedImage, emptySink, type TimingSink } from './embedFace';
import { normalizeUri } from '../pipeline/normalizeUri';
import { cosineSimilarity } from '../pipeline/similarity';
import { now, summarize } from '../bench/timing';
import type {
  AlignMode,
  ModelSpec,
  RefBucket,
  ReferenceEmbedding,
  ReferenceImage,
  ScanMatch,
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
  shouldCancel?: () => boolean;
}

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
): Promise<{ embeddings: ReferenceEmbedding[]; counts: RefFaceCounts }> {
  const embeddings: ReferenceEmbedding[] = [];
  const counts: RefFaceCounts = {};
  for (const ref of references) {
    const norm = await normalizeUri(ref.uri);
    try {
      const faces = await embedImage(norm.path, embedder, spec, align, minFaceSize, sink, detector);
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
    stages: { detect: null, crop: null, preprocess: null, infer: null },
    separation: 0,
    error,
  };
}

export async function runScan(params: RunScanParams): Promise<ScanRun> {
  const { spec, references, photoUris, settings, onProgress, shouldCancel } = params;
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
    embedder = await createEmbedder(spec);
    detector = await createDetector(settings.detector);
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
    );
    const babyRefs = refEmb.filter(r => r.bucket === 'baby');
    const parentRefs = refEmb.filter(r => r.bucket === 'parent');
    if (refEmb.length === 0) {
      throw new Error('No face found in any reference image. Pick clearer, front-facing photos.');
    }

    const matches: ScanMatch[] = [];
    let facesFound = 0;
    let processed = 0;
    let sepSum = 0;
    let sepCount = 0;

    for (let i = 0; i < total; i++) {
      if (shouldCancel?.()) break;
      const uri = photoUris[i];
      const norm = await normalizeUri(uri);
      try {
        const faces = await embedImage(norm.path, embedder, spec, align, settings.minFaceSize, sink, detector);
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
      } catch {
        // Unreadable / undecodable photo — skip it.
      } finally {
        await norm.cleanup();
      }
      processed += 1;

      if (i % 3 === 0 || i === total - 1) {
        emit('scanning', `Scanning ${processed}/${total}…`, 0.05 + 0.95 * (processed / Math.max(1, total)), {
          photoIndex: processed,
          facesFound,
          matchesSoFar: matches.length,
        });
        await new Promise<void>(r => setTimeout(r, 0)); // yield to keep UI responsive
      }
    }

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
        detect: summarize(sink.detect),
        crop: summarize(sink.crop),
        preprocess: summarize(sink.preprocess),
        infer: summarize(sink.infer),
      },
      separation: sepCount ? sepSum / sepCount : 0,
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
