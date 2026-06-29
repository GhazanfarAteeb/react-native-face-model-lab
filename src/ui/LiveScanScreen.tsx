/**
 * LiveScanScreen — the screen shown while a scan runs. Beyond the headline progress, it
 * streams every scanned photo into a grid (newest first) tagged matched / unmatched, with
 * a filter and a live JS-thread health meter so you can SEE whether the (now concurrent)
 * scan pipeline is starving the UI thread.
 *
 * IMPORTANT — the cells are lightweight STATUS TILES, not photo thumbnails. Rendering the
 * original gallery photos here decoded a full-resolution (~12MP ≈ 48MB) bitmap per cell
 * regardless of the tile size; streaming hundreds of them exhausted memory and the OS
 * killed the app. Tiles cost nothing, so the health meter now reflects the SCAN, not the
 * grid's own image decoding. The actual matched photos are shown on the Results screen
 * (windowed, so only visible ones decode).
 */
import React, { useMemo, useState } from 'react';
import { FlatList, Image, Pressable, StyleSheet, Text, View, useWindowDimensions } from 'react-native';
import { useStore } from '../scan/store';
import JsHealthMeter from './JsHealthMeter';
import { Button, ProgressBar } from './components';
import { theme, spacing, radius, mono } from './theme';
import type { LiveStageStats, ScanPhotoEvent, ScanProgress } from '../types';

type Filter = 'all' | 'matched' | 'unmatched';

const DETECTOR_LABEL: Record<string, string> = {
  mlkit: 'ML Kit',
  yunet: 'YuNet',
  scrfd: 'SCRFD',
  blazeface: 'BlazeFace',
};

const COLS = 4;
const GAP = spacing.xs;
// Tiles are cheap (no image decode); cap only to bound the per-flush reverse/diff cost.
const LIVE_CAP = 240;

function Cell({ item, size }: { item: ScanPhotoEvent; size: number }) {
  const matched = item.matched;
  const simPct = item.similarity >= 0 ? Math.round(item.similarity * 100) : null;
  return (
    <View
      style={[
        styles.cell,
        { width: size, height: size, backgroundColor: matched ? theme.accentDim : theme.surfaceAlt, borderColor: matched ? theme.good : theme.border },
      ]}>
      {/* Small generated thumbnail (~150px) — safe to decode. Falls back to a plain tile
          when the thumb isn't ready / failed. */}
      {item.thumbUri ? (
        <Image source={{ uri: item.thumbUri }} style={styles.cellImg} fadeDuration={0} />
      ) : null}
      <View style={styles.cellBadge}>
        {matched ? (
          <Text style={styles.cellSim}>✓ {simPct}% {item.bucket}</Text>
        ) : (
          <Text style={styles.cellMiss}>{item.faceCount ? '✕' : '·'}</Text>
        )}
      </View>
      <Text style={styles.cellIdx}>#{item.index + 1}</Text>
    </View>
  );
}

const MemoCell = React.memo(Cell);

function fmtMs(v: number): string {
  return `${v >= 10 ? Math.round(v) : v.toFixed(1)} ms`;
}

/** Live per-stage breakdown (running means) — updates ~10×/sec during a scan. */
function LiveStats({ stages, detectorLabel, modelLabel }: { stages?: LiveStageStats; detectorLabel: string; modelLabel: string }) {
  if (!stages) return null;
  const rows: Array<[string, number]> = [
    ['Decode Image', stages.decode],
    [detectorLabel, stages.detect],
    ['Alignment', stages.align],
    ['Preprocess', stages.preprocess],
    [modelLabel, stages.embed],
    ['Similarity Search', stages.similarity],
  ];
  return (
    <View style={styles.stats}>
      {rows.map(([k, v]) => (
        <View key={k} style={styles.statRow}>
          <Text style={styles.statKey} numberOfLines={1}>{k}</Text>
          <Text style={styles.statVal}>{fmtMs(v)}</Text>
        </View>
      ))}
      <View style={[styles.statRow, styles.statTotalRow]}>
        <Text style={styles.statKeyTotal}>Total</Text>
        <Text style={styles.statValTotal}>{fmtMs(stages.total)}</Text>
      </View>
    </View>
  );
}

function FilterTab({ label, count, active, color, onPress }: { label: string; count: number; active: boolean; color: string; onPress: () => void }) {
  return (
    <Pressable style={[styles.filterTab, active && { backgroundColor: theme.surfaceAlt, borderColor: color }]} onPress={onPress}>
      <Text style={[styles.filterCount, { color: active ? color : theme.text }]}>{count}</Text>
      <Text style={styles.filterLabel}>{label}</Text>
    </Pressable>
  );
}

export default function LiveScanScreen({ progress }: { progress: ScanProgress }) {
  const { selectedModel, settings, liveEvents, liveVersion, cancelScan } = useStore();
  const { width } = useWindowDimensions();
  const [filter, setFilter] = useState<Filter>('all');

  const cellSize = Math.floor((width - spacing.md * 2 - GAP * (COLS - 1)) / COLS);

  // Recomputed on each throttled version bump. `liveEvents.current` is mutated in place.
  // Cells are lightweight tiles, so this is only bounded to keep the reverse/diff cheap.
  // Counts below are computed over the full history.
  const { data, matched, unmatched, total } = useMemo(() => {
    const all = liveEvents.current;
    let m = 0;
    for (let i = 0; i < all.length; i++) if (all[i].matched) m++;
    const filtered =
      filter === 'all' ? all : filter === 'matched' ? all.filter(e => e.matched) : all.filter(e => !e.matched);
    // Take the tail (newest) then reverse so the latest finished photo is first.
    return { data: filtered.slice(-LIVE_CAP).reverse(), matched: m, unmatched: all.length - m, total: all.length };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveVersion, filter]);

  const pct = Math.round(progress.fraction * 100);
  const detLabel = DETECTOR_LABEL[settings.detector] ?? settings.detector;
  const done = progress.phase === 'done';

  return (
    <View style={styles.root}>
      <View style={styles.header}>
        <View style={styles.headerTop}>
          <Text style={styles.pct}>{pct}%</Text>
          <View style={{ flex: 1 }}>
            <Text style={styles.title}>{done ? 'Scan complete' : 'Finding matching photos'}</Text>
            <Text style={styles.subtitle}>
              {selectedModel.label} · {detLabel} · {settings.concurrency === 0 ? 'auto' : `×${settings.concurrency}`} parallel
            </Text>
          </View>
          <Text style={styles.counts}>{progress.photoIndex}/{progress.photoCount}</Text>
        </View>
        <View style={{ marginTop: spacing.sm }}>
          <ProgressBar fraction={progress.fraction} />
        </View>
      </View>

      <View style={styles.meterWrap}>
        <JsHealthMeter />
        <LiveStats stages={progress.liveStages} detectorLabel={detLabel} modelLabel={selectedModel.label} />
      </View>

      <View style={styles.filters}>
        <FilterTab label="ALL" count={matched + unmatched} active={filter === 'all'} color={theme.accent} onPress={() => setFilter('all')} />
        <FilterTab label="MATCHED" count={matched} active={filter === 'matched'} color={theme.good} onPress={() => setFilter('matched')} />
        <FilterTab label="UNMATCHED" count={unmatched} active={filter === 'unmatched'} color={theme.textDim} onPress={() => setFilter('unmatched')} />
      </View>

      {total > data.length ? (
        <Text style={styles.capNote}>showing latest {data.length} of {total} scanned</Text>
      ) : null}

      <FlatList
        data={data}
        key={COLS}
        numColumns={COLS}
        keyExtractor={e => String(e.index)}
        renderItem={({ item }) => <MemoCell item={item} size={cellSize} />}
        columnWrapperStyle={{ gap: GAP }}
        contentContainerStyle={styles.grid}
        removeClippedSubviews
        initialNumToRender={12}
        maxToRenderPerBatch={8}
        windowSize={3}
        updateCellsBatchingPeriod={80}
        ListEmptyComponent={<Text style={styles.empty}>Warming up…</Text>}
      />

      <View style={styles.footer}>
        <Button title="Cancel scan" variant="danger" onPress={cancelScan} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: theme.bg },
  header: { paddingHorizontal: spacing.md, paddingTop: spacing.md },
  headerTop: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  pct: { color: theme.text, fontSize: 30, fontWeight: '800', fontFamily: mono, minWidth: 78 },
  title: { color: theme.text, fontSize: 16, fontWeight: '700' },
  subtitle: { color: theme.accent, fontSize: 12, fontWeight: '600', marginTop: 1 },
  counts: { color: theme.textDim, fontSize: 13, fontFamily: mono },
  meterWrap: { paddingHorizontal: spacing.md, marginTop: spacing.md, gap: spacing.sm },
  stats: { backgroundColor: theme.surface, borderRadius: 10, borderWidth: StyleSheet.hairlineWidth, borderColor: theme.border, paddingVertical: spacing.xs, paddingHorizontal: spacing.md },
  statRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 2 },
  statKey: { color: theme.textDim, fontSize: 12, flex: 1, marginRight: spacing.sm },
  statVal: { color: theme.text, fontSize: 12, fontFamily: mono },
  statTotalRow: { borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: theme.border, marginTop: 2, paddingTop: 4 },
  statKeyTotal: { color: theme.text, fontSize: 12, fontWeight: '700', flex: 1 },
  statValTotal: { color: theme.accent, fontSize: 13, fontWeight: '700', fontFamily: mono },
  filters: { flexDirection: 'row', gap: spacing.sm, paddingHorizontal: spacing.md, marginTop: spacing.md },
  filterTab: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: spacing.sm,
    borderRadius: radius.md,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: theme.border,
    backgroundColor: theme.surface,
  },
  filterCount: { fontSize: 18, fontWeight: '800', fontFamily: mono },
  filterLabel: { color: theme.textFaint, fontSize: 9, fontWeight: '700', letterSpacing: 0.8, marginTop: 1 },
  capNote: { color: theme.textFaint, fontSize: 11, textAlign: 'center', marginTop: spacing.sm },
  grid: { padding: spacing.md, gap: GAP },
  cell: { borderRadius: radius.sm, borderWidth: 1.5, alignItems: 'center', justifyContent: 'center', overflow: 'hidden' },
  cellImg: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 },
  cellBadge: { position: 'absolute', left: 0, right: 0, bottom: 0, paddingVertical: 2, alignItems: 'center', backgroundColor: 'rgba(0,0,0,0.55)' },
  cellSim: { color: theme.good, fontSize: 11, fontWeight: '800', fontFamily: mono },
  cellMiss: { color: '#cfd6dd', fontSize: 12, fontWeight: '700' },
  cellIdx: { position: 'absolute', top: 2, right: 4, color: '#fff', fontSize: 9, fontFamily: mono, opacity: 0.8 },
  empty: { color: theme.textFaint, textAlign: 'center', marginTop: spacing.xl },
  footer: { padding: spacing.md, borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: theme.border },
});
