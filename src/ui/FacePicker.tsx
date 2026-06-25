/**
 * FacePicker — pick reference faces from a grid of faces detected across the recent
 * gallery (rnbaby's NewbornPicker/RecentPicker pattern). Faces stream in as they're
 * found; tap tiles to select; "Use N faces" returns the chosen thumbnails as references.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Dimensions, FlatList, Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { extractGalleryFaces, type GalleryFace } from '../scan/facePicker';
import { useStore } from '../scan/store';
import { ProgressRing } from './ProgressRing';
import { theme, spacing, radius, mono } from './theme';
import type { RefBucket } from '../types';

const COLS = 3;
const GAP = spacing.sm;
const TILE = (Dimensions.get('window').width - spacing.lg * 2 - GAP * (COLS - 1)) / COLS;
const EXTRACT_LIMIT = 120; // recent photos to scan for reference faces

export default function FacePicker({
  bucket,
  onClose,
  onConfirm,
}: {
  bucket: RefBucket;
  onClose: () => void;
  onConfirm: (thumbUris: string[]) => void;
}) {
  const { settings } = useStore();
  const [faces, setFaces] = useState<GalleryFace[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [progress, setProgress] = useState({ done: 0, total: 0, faces: 0 });
  const [running, setRunning] = useState(true);
  const cancelRef = useRef(false);

  useEffect(() => {
    cancelRef.current = false;
    extractGalleryFaces({
      limit: EXTRACT_LIMIT,
      minFaceSize: settings.minFaceSize,
      onFace: f => setFaces(prev => [...prev, f]),
      onProgress: (done, total, n) => setProgress({ done, total, faces: n }),
      shouldCancel: () => cancelRef.current,
    })
      .catch(() => {})
      .finally(() => setRunning(false));
    return () => {
      cancelRef.current = true;
    };
  }, [settings.minFaceSize]);

  const sorted = useMemo(() => [...faces].sort((a, b) => b.sizeScore - a.sizeScore), [faces]);
  const maxSize = useMemo(() => sorted.reduce((m, f) => Math.max(m, f.sizeScore), 1), [sorted]);

  const toggle = useCallback((id: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const confirm = () => {
    const uris = sorted.filter(f => selected.has(f.id)).map(f => f.thumbUri);
    onConfirm(uris);
  };

  const title = bucket === 'baby' ? 'Pick the baby’s faces' : 'Pick the parents’ faces';

  return (
    <View style={styles.root}>
      <View style={styles.header}>
        <Pressable onPress={onClose} hitSlop={10} style={styles.close}>
          <Text style={styles.closeText}>✕</Text>
        </Pressable>
        <View style={{ flex: 1 }}>
          <Text style={styles.title}>{title}</Text>
          <Text style={styles.sub}>
            {running
              ? `Scanning ${progress.done}/${progress.total || '…'} · ${progress.faces} faces`
              : `${faces.length} faces found · tap to select`}
          </Text>
        </View>
      </View>

      {faces.length === 0 && running ? (
        <View style={styles.loading}>
          <ProgressRing size={120} stroke={8} progress={progress.total ? progress.done / progress.total : 0}>
            <Text style={styles.loadingPct}>
              {progress.total ? Math.round((progress.done / progress.total) * 100) : 0}%
            </Text>
          </ProgressRing>
          <Text style={styles.loadingText}>Finding faces in your gallery…</Text>
        </View>
      ) : (
        <FlatList
          data={sorted}
          keyExtractor={f => f.id}
          numColumns={COLS}
          contentContainerStyle={styles.grid}
          columnWrapperStyle={{ gap: GAP }}
          renderItem={({ item }) => {
            const isSel = selected.has(item.id);
            const best = item.sizeScore >= maxSize * 0.7;
            return (
              <Pressable onPress={() => toggle(item.id)} style={[styles.tile, isSel && styles.tileSel]}>
                <Image source={{ uri: item.thumbUri }} style={styles.tileImg} />
                <View style={[styles.qual, best ? styles.qualBest : styles.qualGood]}>
                  <Text style={styles.qualText}>{best ? '★ Best' : 'Good'}</Text>
                </View>
                {isSel && (
                  <View style={styles.check}>
                    <Text style={styles.checkText}>✓</Text>
                  </View>
                )}
              </Pressable>
            );
          }}
          ListFooterComponent={
            running ? <Text style={styles.footer}>Scanning {progress.done}/{progress.total}…</Text> : null
          }
        />
      )}

      <View style={styles.bar}>
        <Pressable
          onPress={confirm}
          disabled={selected.size === 0}
          style={[styles.doneBtn, selected.size === 0 && { opacity: 0.4 }]}>
          <Text style={styles.doneText}>
            {selected.size === 0 ? 'Select faces' : `Use ${selected.size} face${selected.size > 1 ? 's' : ''}`}
          </Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: theme.bg },
  header: { flexDirection: 'row', alignItems: 'center', padding: spacing.lg, gap: spacing.md },
  close: { width: 32, height: 32, borderRadius: 16, backgroundColor: theme.surfaceAlt, alignItems: 'center', justifyContent: 'center' },
  closeText: { color: theme.text, fontSize: 16 },
  title: { color: theme.text, fontSize: 18, fontWeight: '700' },
  sub: { color: theme.textDim, fontSize: 12, marginTop: 1 },
  loading: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: spacing.lg },
  loadingPct: { color: theme.text, fontSize: 24, fontWeight: '700', fontFamily: mono },
  loadingText: { color: theme.textDim, fontSize: 14 },
  grid: { paddingHorizontal: spacing.lg, paddingBottom: spacing.lg, gap: GAP },
  tile: { width: TILE, height: TILE, borderRadius: radius.md, overflow: 'hidden', backgroundColor: theme.surfaceAlt, borderWidth: 2, borderColor: 'transparent' },
  tileSel: { borderColor: theme.accent },
  tileImg: { width: '100%', height: '100%' },
  qual: { position: 'absolute', top: 5, right: 5, borderRadius: 4, paddingHorizontal: 5, paddingVertical: 1 },
  qualBest: { backgroundColor: 'rgba(212,83,126,0.92)' },
  qualGood: { backgroundColor: 'rgba(0,0,0,0.6)' },
  qualText: { color: '#fff', fontSize: 9, fontWeight: '700' },
  check: { position: 'absolute', bottom: 5, left: 5, width: 22, height: 22, borderRadius: 11, backgroundColor: theme.accent, alignItems: 'center', justifyContent: 'center' },
  checkText: { color: '#fff', fontSize: 13, fontWeight: '800' },
  footer: { color: theme.textFaint, fontSize: 12, textAlign: 'center', paddingVertical: spacing.md },
  bar: { padding: spacing.lg, borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: theme.border },
  doneBtn: { backgroundColor: theme.accent, borderRadius: radius.md, paddingVertical: 14, alignItems: 'center' },
  doneText: { color: '#fff', fontSize: 15, fontWeight: '700' },
});
