/**
 * ScanScreen — the reference + scan flow, modeled on rnbaby's ReferencePhotoOnboarding:
 *   • Baby + Parent reference "slots" filled from a grid of detected gallery faces.
 *   • A "Start scan" CTA that runs the selected model over the gallery.
 *   • While scanning, the live ring + stats view; on finish, jumps to Results.
 */
import React, { useCallback, useState } from 'react';
import { Alert, Image, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { useStore } from '../scan/store';
import { recentPhotoUris } from '../scan/gallery';
import { Button, Card, Chip, Muted, SectionLabel } from './components';
import FacePicker from './FacePicker';
import ScanningView from './ScanningView';
import { theme, spacing, radius, mono } from './theme';
import type { RefBucket } from '../types';

function Slots({ bucket, onAdd }: { bucket: RefBucket; onAdd: () => void }) {
  const { babyRefs, parentRefs, removeRef } = useStore();
  const refs = bucket === 'baby' ? babyRefs : parentRefs;
  return (
    <ScrollView horizontal showsHorizontalScrollIndicator={false} style={{ marginTop: spacing.sm }}>
      <Pressable style={styles.addSlot} onPress={onAdd}>
        <Text style={styles.addPlus}>＋</Text>
      </Pressable>
      {refs.map(r => (
        <Pressable key={r.uri} style={styles.slot} onPress={() => removeRef(bucket, r.uri)}>
          <Image source={{ uri: r.uri }} style={styles.slotImg} />
          <View style={styles.slotRemove}>
            <Text style={styles.slotRemoveText}>✕</Text>
          </View>
        </Pressable>
      ))}
    </ScrollView>
  );
}

export default function ScanScreen({ goToResults }: { goToResults: () => void }) {
  const {
    selectedModel,
    babyRefs,
    parentRefs,
    settings,
    setSettings,
    scanning,
    progress,
    addRefs,
    startScan,
  } = useStore();
  const [picking, setPicking] = useState<RefBucket | null>(null);
  const [preparing, setPreparing] = useState(false);

  const hasRefs = babyRefs.length + parentRefs.length > 0;

  const onScan = useCallback(async () => {
    setPreparing(true);
    try {
      const photoUris = await recentPhotoUris(settings.maxPhotos);
      if (!photoUris.length) {
        Alert.alert('No photos', 'Your gallery looks empty.');
        return;
      }
      setPreparing(false);
      const run = await startScan(photoUris);
      if (run) goToResults();
    } catch (e) {
      Alert.alert('Scan failed', String(e instanceof Error ? e.message : e));
    } finally {
      setPreparing(false);
    }
  }, [settings.maxPhotos, startScan, goToResults]);

  // Live scan ring takes over the whole tab while a scan runs.
  if (scanning && progress) return <ScanningView progress={progress} />;

  // Face picker takes over the whole tab while choosing references.
  if (picking) {
    return (
      <FacePicker
        bucket={picking}
        onClose={() => setPicking(null)}
        onConfirm={uris => {
          if (uris.length) addRefs(picking, uris);
          setPicking(null);
        }}
      />
    );
  }

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
      <View style={styles.modelLine}>
        <Muted>Model</Muted>
        <Text style={styles.modelName}>{selectedModel.label}</Text>
      </View>

      <Card>
        <SectionLabel>Baby references · {babyRefs.length}</SectionLabel>
        <Muted>Pick the baby’s face from your gallery — these define who we’re looking for.</Muted>
        <Slots bucket="baby" onAdd={() => setPicking('baby')} />
      </Card>

      <Card>
        <SectionLabel>Parent references · {parentRefs.length}</SectionLabel>
        <Muted>Optional. Helps tell the baby apart from adults in the same photos.</Muted>
        <Slots bucket="parent" onAdd={() => setPicking('parent')} />
      </Card>

      <Card>
        <SectionLabel>Photos to scan (max)</SectionLabel>
        <View style={styles.chipRow}>
          {[50, 100, 200, 500, 1000].map(n => (
            <Chip key={n} label={String(n)} active={settings.maxPhotos === n} onPress={() => setSettings({ maxPhotos: n })} />
          ))}
        </View>
        <SectionLabel>Match threshold (cosine)</SectionLabel>
        <View style={styles.stepper}>
          <Pressable style={styles.stepBtn} onPress={() => setSettings({ threshold: Math.max(0.1, +(settings.threshold - 0.05).toFixed(2)) })}>
            <Text style={styles.stepBtnText}>−</Text>
          </Pressable>
          <Text style={styles.stepValue}>{settings.threshold.toFixed(2)}</Text>
          <Pressable style={styles.stepBtn} onPress={() => setSettings({ threshold: Math.min(0.95, +(settings.threshold + 0.05).toFixed(2)) })}>
            <Text style={styles.stepBtnText}>＋</Text>
          </Pressable>
          <Muted style={{ flex: 1, marginLeft: spacing.md }}>Detector & alignment are on the Model tab.</Muted>
        </View>
      </Card>

      <Button
        title={hasRefs ? `Start finding matches` : 'Add a baby reference first'}
        onPress={onScan}
        disabled={!hasRefs}
        loading={preparing}
      />
      <View style={{ height: spacing.xl }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: theme.bg },
  content: { padding: spacing.lg },
  modelLine: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: spacing.md },
  modelName: { color: theme.text, fontWeight: '700', fontSize: 14 },
  addSlot: { width: 64, height: 64, borderRadius: 32, borderWidth: 1.5, borderColor: theme.accent, borderStyle: 'dashed', alignItems: 'center', justifyContent: 'center', marginRight: spacing.sm },
  addPlus: { color: theme.accent, fontSize: 24 },
  slot: { width: 64, height: 64, borderRadius: 32, marginRight: spacing.sm },
  slotImg: { width: 64, height: 64, borderRadius: 32, backgroundColor: theme.surfaceAlt },
  slotRemove: { position: 'absolute', top: -2, right: -2, width: 20, height: 20, borderRadius: 10, backgroundColor: theme.bad, alignItems: 'center', justifyContent: 'center' },
  slotRemoveText: { color: '#fff', fontSize: 10, fontWeight: '700' },
  chipRow: { flexDirection: 'row', flexWrap: 'wrap', marginBottom: spacing.sm },
  stepper: { flexDirection: 'row', alignItems: 'center' },
  stepBtn: { width: 38, height: 38, borderRadius: radius.sm, backgroundColor: theme.surfaceAlt, alignItems: 'center', justifyContent: 'center' },
  stepBtnText: { color: theme.text, fontSize: 20, fontWeight: '700' },
  stepValue: { color: theme.text, fontSize: 18, fontFamily: mono, marginHorizontal: spacing.md, minWidth: 52, textAlign: 'center' },
});
