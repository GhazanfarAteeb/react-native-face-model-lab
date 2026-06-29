/**
 * ResultsScreen — the best matches for a run (per bucket, ranked by similarity) plus
 * the time the scan took (headline + per-stage medians), and a comparison row for every
 * run so "which model is best" is answerable at a glance.
 */
import React, { useMemo, useState } from 'react';
import {
  Alert,
  Dimensions,
  FlatList,
  Image,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import RNFS from 'react-native-fs';
import { useStore } from '../scan/store';
import { Badge, Button, Card, Muted, SectionLabel, Segmented, Stat } from './components';
import { fmtBytes, fmtMedian, fmtMs } from './format';
import { theme, spacing, radius, mono } from './theme';
import type { RefBucket, ScanRun } from '../types';

const COLS = 3;
const GAP = spacing.sm;
const THUMB = (Dimensions.get('window').width - spacing.lg * 2 - GAP * (COLS - 1)) / COLS;

function StageTimings({ run }: { run: ScanRun }) {
  const rows: [string, string][] = [
    ['decode image', fmtMedian(run.stages.decode)],
    [`detect (${run.detectorLabel})`, fmtMedian(run.stages.detect)],
    ['crop + align', fmtMedian(run.stages.crop)],
    ['preprocess', fmtMedian(run.stages.preprocess)],
    [`embed (${run.modelLabel})`, fmtMedian(run.stages.infer)],
    ['similarity search', fmtMedian(run.stages.similarity)],
    ['total / photo', fmtMedian(run.stages.total)],
  ];
  return (
    <View style={styles.stageBox}>
      <Text style={styles.stageTitle}>Median time per stage</Text>
      {rows.map(([k, v]) => (
        <View key={k} style={styles.stageRow}>
          <Text style={styles.stageKey}>{k}</Text>
          <Text style={styles.stageVal}>{v}</Text>
        </View>
      ))}
    </View>
  );
}

function CacheLine({ run }: { run: ScanRun }) {
  const c = run.cacheStats;
  if (!c) return null;
  const detTotal = c.detectHits + c.detectMisses;
  const cropTotal = c.cropHits + c.cropMisses;
  if (detTotal + cropTotal === 0) return null;
  const pct = (h: number, t: number) => (t ? Math.round((h / t) * 100) : 0);
  // A warm rerun (high reuse) shows the cache skipping detect/crop; a cold run is ~0%.
  return (
    <View style={styles.cacheBox}>
      <Text style={styles.cacheText}>
        cache reuse · detect {pct(c.detectHits, detTotal)}%  ·  crop {pct(c.cropHits, cropTotal)}%
      </Text>
      <Text style={styles.cacheSub}>
        {c.detectHits} detections + {c.cropHits} crops reused from earlier runs
      </Text>
    </View>
  );
}

// One match thumbnail. Rendered inside a WINDOWED FlatList (see ResultsScreen) so only the
// visible images decode — mounting all matches at once decodes a full-res (~48MB) bitmap
// each and exhausts memory on large galleries.
const MatchCell = React.memo(({ uri, similarity }: { uri: string; similarity: number }) => {
  return (
    <View style={[styles.cell, { width: THUMB, height: THUMB }]}>
      <Image source={{ uri }} style={styles.cellImg} resizeMethod="resize" />
      <View style={styles.simBadge}>
        <Text style={styles.simText}>{(similarity * 100).toFixed(0)}%</Text>
      </View>
    </View>
  );
});

export default function ResultsScreen() {
  const { runs, latestRun, clearRuns, clearCache, cacheSizes } = useStore();
  const [viewedId, setViewedId] = useState<string | null>(null);
  const [bucket, setBucket] = useState<RefBucket>('baby');
  const [cacheSz, setCacheSz] = useState(() => cacheSizes());

  const run = useMemo(
    () => runs.find(r => r.id === viewedId) ?? latestRun,
    [runs, viewedId, latestRun],
  );

  if (!run) {
    return (
      <View style={styles.empty}>
        <Text style={styles.emptyTitle}>No scans yet</Text>
        <Muted>Pick references on the Scan tab and run a scan to see matches here.</Muted>
      </View>
    );
  }

  const onExport = async () => {
    try {
      const dir = `${RNFS.DocumentDirectoryPath}/results`;
      await RNFS.mkdir(dir).catch(() => {});
      const path = `${dir}/${run.id}.json`;
      await RNFS.writeFile(path, JSON.stringify(run, null, 2), 'utf8');
      Alert.alert('Exported', path);
    } catch (e) {
      Alert.alert('Export failed', String(e instanceof Error ? e.message : e));
    }
  };

  const fastestInfer = Math.min(
    ...runs.filter(r => r.stages.infer).map(r => r.stages.infer!.median),
  );

  // Bucket-filtered matches drive the windowed grid (data of the FlatList).
  const gridMatches = run.error ? [] : run.matches.filter(m => m.bucket === bucket).slice(0, 120);

  const header = (
    <>
      {run.error ? (
        <Card style={{ borderColor: theme.bad }}>
          <Badge text="ERROR" tone="bad" />
          <Text style={styles.errText}>{run.error}</Text>
        </Card>
      ) : (
        <Card>
          <SectionLabel>
            {run.modelLabel} · {run.detectorLabel}
          </SectionLabel>
          <View style={styles.headline}>
            <Text style={styles.bigTime}>{fmtMs(run.durationMs)}</Text>
            <Text style={styles.bigTimeSub}>
              to scan {run.photosScanned} photos · {fmtMs(run.avgMsPerPhoto)}/photo
            </Text>
          </View>
          <View style={styles.statRow}>
            <Stat label="faces" value={String(run.facesFound)} />
            <Stat label="faces/s" value={run.facesPerSec.toFixed(1)} />
            <Stat label="baby" value={String(run.babyMatchCount)} />
            <Stat label="parent" value={String(run.parentMatchCount)} />
          </View>
          <View style={styles.statRow}>
            <Stat label="model load" value={fmtMs(run.loadMs)} />
            <Stat label="size" value={fmtBytes(run.fileSizeBytes)} />
            <Stat label="separation" value={run.separation.toFixed(3)} hint="baby vs parent gap" />
          </View>
          <StageTimings run={run} />
          <CacheLine run={run} />
        </Card>
      )}

      {!run.error && (
        <View style={styles.matchHeader}>
          <SectionLabel>Best matches</SectionLabel>
          <Segmented
            value={bucket}
            onChange={setBucket}
            options={[
              { label: `Baby (${run.babyMatchCount})`, value: 'baby' },
              { label: `Parent (${run.parentMatchCount})`, value: 'parent' },
            ]}
          />
          {gridMatches.length === 0 && (
            <Muted style={{ marginTop: spacing.md }}>No {bucket} matches above the threshold.</Muted>
          )}
        </View>
      )}
    </>
  );

  const footer = (
    <>
      {!run.error && (
        <Card style={{ marginTop: spacing.md }}>
          <Button title="Export run as JSON" variant="secondary" onPress={onExport} />
        </Card>
      )}

      {!run.error && (
        <Card>
          <SectionLabel>Scan cache</SectionLabel>
          <Muted>
            {cacheSz.detect} detections · {cacheSz.crop} crops held in memory. Clear to benchmark a cold (un-cached) run.
          </Muted>
          <View style={{ height: spacing.sm }} />
          <Button
            title="Clear scan cache"
            variant="secondary"
            onPress={() => {
              clearCache();
              setCacheSz(cacheSizes());
            }}
          />
        </Card>
      )}

      {runs.length > 1 && (
        <Card>
          <SectionLabel>Compare runs</SectionLabel>
          <Muted>Tap a run to view its matches above.</Muted>
          <View style={{ height: spacing.sm }} />
          {runs.map(r => {
            const active = r.id === run.id;
            const isFastest = r.stages.infer && r.stages.infer.median === fastestInfer;
            return (
              <Pressable key={r.id} onPress={() => setViewedId(r.id)}>
                <View style={[styles.cmpRow, active && styles.cmpRowActive]}>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.cmpModel}>{r.modelLabel}</Text>
                    <Text style={styles.cmpSub}>
                      {r.error
                        ? 'error'
                        : `${r.detectorLabel} · ${r.photosScanned} photos · ${r.babyMatchCount + r.parentMatchCount} matches`}
                    </Text>
                  </View>
                  <View style={styles.cmpNums}>
                    <Text style={styles.cmpInfer}>{fmtMedian(r.stages.infer)}</Text>
                    <Text style={styles.cmpInferLabel}>infer{isFastest ? ' ⚡' : ''}</Text>
                  </View>
                  <View style={styles.cmpNums}>
                    <Text style={styles.cmpInfer}>{fmtMs(r.avgMsPerPhoto)}</Text>
                    <Text style={styles.cmpInferLabel}>/photo</Text>
                  </View>
                </View>
              </Pressable>
            );
          })}
          <View style={{ height: spacing.md }} />
          <Button title="Clear history" variant="secondary" onPress={clearRuns} />
        </Card>
      )}
      <View style={{ height: spacing.xl }} />
    </>
  );

  return (
    <FlatList
      style={styles.scroll}
      contentContainerStyle={styles.content}
      data={gridMatches}
      key={COLS}
      numColumns={COLS}
      keyExtractor={(m, i) => `${m.uri}#${i}`}
      renderItem={({ item }) => <MatchCell uri={item.uri} similarity={item.similarity} />}
      columnWrapperStyle={styles.gridRow}
      ListHeaderComponent={header}
      ListFooterComponent={footer}
      removeClippedSubviews
      initialNumToRender={9}
      maxToRenderPerBatch={9}
      windowSize={5}
    />
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: theme.bg },
  content: { padding: spacing.lg },
  empty: { flex: 1, backgroundColor: theme.bg, alignItems: 'center', justifyContent: 'center', padding: spacing.xl },
  emptyTitle: { color: theme.text, fontSize: 18, fontWeight: '700', marginBottom: spacing.sm },
  headline: { marginVertical: spacing.md },
  bigTime: { color: theme.text, fontSize: 40, fontWeight: '800', fontFamily: mono },
  bigTimeSub: { color: theme.textDim, fontSize: 13, marginTop: 2 },
  statRow: { flexDirection: 'row', flexWrap: 'wrap', marginTop: spacing.sm },
  stageBox: { marginTop: spacing.md, padding: spacing.md, backgroundColor: theme.surfaceAlt, borderRadius: 8 },
  stageTitle: { color: theme.textDim, fontSize: 11, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.6, marginBottom: spacing.sm },
  stageRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 3 },
  stageKey: { color: theme.textFaint, fontSize: 13 },
  stageVal: { color: theme.text, fontSize: 13, fontFamily: mono },
  cacheBox: { marginTop: spacing.sm, paddingTop: spacing.sm, borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: theme.border },
  cacheText: { color: theme.accent, fontSize: 12, fontWeight: '700', fontFamily: mono },
  cacheSub: { color: theme.textFaint, fontSize: 11, marginTop: 1 },
  matchHeader: { marginBottom: spacing.sm },
  gridRow: { gap: GAP, marginBottom: GAP },
  cell: { borderRadius: radius.sm, overflow: 'hidden', backgroundColor: theme.surfaceAlt },
  cellImg: { width: '100%', height: '100%' },
  simBadge: { position: 'absolute', bottom: 4, left: 4, backgroundColor: 'rgba(0,0,0,0.7)', borderRadius: 4, paddingHorizontal: 5, paddingVertical: 1 },
  simText: { color: '#fff', fontSize: 11, fontWeight: '700', fontFamily: mono },
  errText: { color: theme.text, fontSize: 14, marginTop: spacing.sm, lineHeight: 20 },
  cmpRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: spacing.sm, paddingHorizontal: spacing.sm, borderRadius: 8 },
  cmpRowActive: { backgroundColor: theme.surfaceAlt },
  cmpModel: { color: theme.text, fontSize: 14, fontWeight: '600' },
  cmpSub: { color: theme.textFaint, fontSize: 11, marginTop: 1 },
  cmpNums: { alignItems: 'flex-end', marginLeft: spacing.md, minWidth: 56 },
  cmpInfer: { color: theme.text, fontSize: 14, fontFamily: mono },
  cmpInferLabel: { color: theme.textFaint, fontSize: 10 },
});
