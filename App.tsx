/**
 * FaceModelLab — a face-recognition model benchmark built around the rnbaby pipeline.
 *
 * Three screens (custom tab bar — no navigation lib, keeps the native surface small):
 *   Model   — choose the embedding model the scan uses
 *   Scan    — pick baby + parent references, scan the gallery
 *   Results — best matches per bucket + the time the scan took + model comparison
 */
import React, { useState } from 'react';
import { Pressable, StatusBar, StyleSheet, Text, View } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { StoreProvider } from './src/scan/store';
import ModelScreen from './src/ui/ModelScreen';
import ScanScreen from './src/ui/ScanScreen';
import ResultsScreen from './src/ui/ResultsScreen';
import { theme, spacing } from './src/ui/theme';

type Tab = 'model' | 'scan' | 'results';
const TABS: { key: Tab; label: string; icon: string }[] = [
  { key: 'model', label: 'Model', icon: '▢' },
  { key: 'scan', label: 'Scan', icon: '◎' },
  { key: 'results', label: 'Results', icon: '≣' },
];

export default function App() {
  const [tab, setTab] = useState<Tab>('scan');
  return (
    <SafeAreaProvider>
      <StoreProvider>
        <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
          <StatusBar barStyle="light-content" backgroundColor={theme.bg} />
          <View style={styles.header}>
            <Text style={styles.title}>FaceModelLab</Text>
            <Text style={styles.subtitle}>face-recognition model benchmark</Text>
          </View>

          <View style={styles.body}>
            {tab === 'model' && <ModelScreen />}
            {tab === 'scan' && <ScanScreen goToResults={() => setTab('results')} />}
            {tab === 'results' && <ResultsScreen />}
          </View>

          <View style={styles.tabbar}>
            {TABS.map(t => {
              const active = tab === t.key;
              return (
                <Pressable key={t.key} style={styles.tab} onPress={() => setTab(t.key)}>
                  <Text style={[styles.tabIcon, active && styles.tabActive]}>{t.icon}</Text>
                  <Text style={[styles.tabLabel, active && styles.tabActive]}>{t.label}</Text>
                </Pressable>
              );
            })}
          </View>
        </SafeAreaView>
      </StoreProvider>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: theme.bg },
  header: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
    paddingBottom: spacing.md,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: theme.border,
  },
  title: { color: theme.text, fontSize: 22, fontWeight: '800' },
  subtitle: { color: theme.textFaint, fontSize: 12, marginTop: 1 },
  body: { flex: 1 },
  tabbar: {
    flexDirection: 'row',
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: theme.border,
    backgroundColor: theme.surface,
    paddingTop: spacing.sm,
  },
  tab: { flex: 1, alignItems: 'center', paddingVertical: spacing.xs },
  tabIcon: { fontSize: 18, color: theme.textFaint },
  tabLabel: { fontSize: 11, color: theme.textFaint, marginTop: 2 },
  tabActive: { color: theme.accent },
});
