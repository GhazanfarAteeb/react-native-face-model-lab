/**
 * ScanningView — the live scan screen (rnbaby LiveScanScreen pattern): a circular
 * progress ring with the percentage, the model that's running, and live stats
 * (photos scanned, faces found, matches). Shown while a scan is in progress.
 */
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { useStore } from '../scan/store';
import { ProgressRing } from './ProgressRing';
import { Button } from './components';
import { theme, spacing, mono } from './theme';
import type { ScanProgress } from '../types';

function StatRow({ icon, label, value, color }: { icon: string; label: string; value: string; color: string }) {
  return (
    <View style={styles.statRow}>
      <View style={[styles.statIcon, { borderColor: color }]}>
        <Text style={[styles.statIconText, { color }]}>{icon}</Text>
      </View>
      <Text style={styles.statLabel}>{label}</Text>
      <Text style={[styles.statValue, { color }]}>{value}</Text>
    </View>
  );
}

export default function ScanningView({ progress }: { progress: ScanProgress }) {
  const { selectedModel, settings, cancelScan } = useStore();
  const pct = Math.round(progress.fraction * 100);

  return (
    <View style={styles.root}>
      <View style={styles.ringWrap}>
        <ProgressRing progress={progress.fraction} size={196} stroke={12}>
          <Text style={styles.pct}>{pct}%</Text>
          <Text style={styles.scanning}>
            {progress.phase === 'done' ? 'DONE' : 'SCANNING'}
          </Text>
        </ProgressRing>
      </View>

      <Text style={styles.title}>Finding matching photos</Text>
      <Text style={styles.subtitle}>
        {selectedModel.label} · {settings.detector === 'yunet' ? 'YuNet' : 'ML Kit'}
      </Text>
      <Text style={styles.message}>{progress.message}</Text>

      <View style={styles.stats}>
        <StatRow
          icon="◷"
          label="Photos scanned"
          value={`${progress.photoIndex} / ${progress.photoCount}`}
          color={theme.textDim}
        />
        <StatRow icon="☺" label="Faces found" value={String(progress.facesFound)} color={theme.accent} />
        <StatRow icon="✓" label="Matches" value={String(progress.matchesSoFar)} color={theme.good} />
      </View>

      <View style={styles.cancel}>
        <Button title="Cancel scan" variant="danger" onPress={cancelScan} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: theme.bg, alignItems: 'center', justifyContent: 'center', padding: spacing.xl },
  ringWrap: { marginBottom: spacing.xl },
  pct: { color: theme.text, fontSize: 40, fontWeight: '800', fontFamily: mono },
  scanning: { color: theme.textDim, fontSize: 11, fontWeight: '700', letterSpacing: 2, marginTop: 2 },
  title: { color: theme.text, fontSize: 20, fontWeight: '700' },
  subtitle: { color: theme.accent, fontSize: 13, fontWeight: '600', marginTop: 2 },
  message: { color: theme.textFaint, fontSize: 12, marginTop: spacing.sm },
  stats: { width: '100%', marginTop: spacing.xl, gap: spacing.sm },
  statRow: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.surface, borderRadius: 10, padding: spacing.md, gap: spacing.md },
  statIcon: { width: 28, height: 28, borderRadius: 14, borderWidth: 1.5, alignItems: 'center', justifyContent: 'center' },
  statIconText: { fontSize: 13, fontWeight: '700' },
  statLabel: { flex: 1, color: theme.textDim, fontSize: 14 },
  statValue: { fontSize: 16, fontWeight: '700', fontFamily: mono },
  cancel: { width: '100%', marginTop: spacing.xl },
});
