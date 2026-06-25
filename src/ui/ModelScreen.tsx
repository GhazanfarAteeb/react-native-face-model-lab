/**
 * ModelScreen — choose which embedding model the scan uses. Shows each model's specs
 * and whether its weight file is available (bundled or already pushed to the device).
 */
import React, { useEffect, useState } from 'react';
import { Linking, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { useStore } from '../scan/store';
import { isModelAvailable, MODELS_DIR } from '../runtime/modelAssets';
import { Badge, Card, Muted, SectionLabel, Segmented } from './components';
import { theme, spacing, mono } from './theme';
import type { ModelSpec } from '../types';

function Spec({ k, v }: { k: string; v: string }) {
  return (
    <View style={styles.specRow}>
      <Text style={styles.specKey}>{k}</Text>
      <Text style={styles.specVal}>{v}</Text>
    </View>
  );
}

export default function ModelScreen() {
  const { models, selectedModelId, selectModel, settings, setSettings } = useStore();
  const [avail, setAvail] = useState<Record<string, boolean>>({});

  useEffect(() => {
    let alive = true;
    (async () => {
      const entries = await Promise.all(
        models.map(async m => [m.id, await isModelAvailable(m)] as const),
      );
      if (alive) setAvail(Object.fromEntries(entries));
    })();
    return () => {
      alive = false;
    };
  }, [models]);

  const norm = (m: ModelSpec) =>
    typeof m.norm.mean === 'number'
      ? `(x−${m.norm.mean})/${typeof m.norm.std === 'number' ? m.norm.std : '…'}`
      : 'per-channel';

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
      <Card>
        <SectionLabel>Detector</SectionLabel>
        <Segmented
          value={settings.detector}
          onChange={v => setSettings({ detector: v })}
          options={[
            { label: 'ML Kit', value: 'mlkit' },
            { label: 'YuNet', value: 'yunet' },
            { label: 'SCRFD', value: 'scrfd' },
            { label: 'Blaze', value: 'blazeface' },
          ]}
        />
        <Muted style={{ marginTop: spacing.xs }}>
          ML Kit is BlazeFace-family. SCRFD gives 5 landmarks (aligns); BlazeFace gives a
          mouth-centre only, so it falls back to bbox crop.
        </Muted>
        <View style={{ height: spacing.md }} />
        <SectionLabel>Alignment</SectionLabel>
        <Segmented
          value={settings.align}
          onChange={v => setSettings({ align: v })}
          options={[
            { label: 'Per-model', value: 'spec' },
            { label: 'ArcFace', value: 'force-arcface' },
            { label: 'Bbox', value: 'force-bbox' },
          ]}
        />
      </Card>

      <SectionLabel>Embedding model</SectionLabel>
      <Muted>
        The detector + crop are shared across models, so picking a model here is the only
        thing that changes between scans — that's what makes the time/quality numbers
        comparable. Re-pick after a scan and run again to compare on the Results tab.
      </Muted>
      <View style={{ height: spacing.md }} />

      {models.map(m => {
        const selected = m.id === selectedModelId;
        const ready = avail[m.id];
        return (
          <Pressable key={m.id} onPress={() => selectModel(m.id)}>
            <Card style={selected ? { borderColor: theme.accent } : undefined}>
              <View style={styles.header}>
                <View style={styles.radio}>
                  {selected ? <View style={styles.radioDot} /> : null}
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={styles.title}>{m.label}</Text>
                  <Text style={styles.sub}>
                    {m.family} · {m.runtime.toUpperCase()}
                  </Text>
                </View>
                {ready === undefined ? null : ready ? (
                  <Badge text={m.bundled ? 'BUNDLED' : 'ON DEVICE'} tone="good" />
                ) : (
                  <Badge text="NEEDS FILE" tone="warn" />
                )}
              </View>

              <View style={styles.specs}>
                <Spec k="input" v={`${m.input.width}×${m.input.height} ${m.input.channels} ${m.input.layout}`} />
                <Spec k="embed" v={`${m.output.dim}-D`} />
                <Spec k="norm" v={norm(m)} />
                <Spec k="align" v={m.align} />
                <Spec k="license" v={m.license} />
              </View>

              {m.notes ? <Muted style={{ marginTop: spacing.sm }}>{m.notes}</Muted> : null}

              {ready === false ? (
                <View style={styles.fetch}>
                  <Text style={styles.fetchText}>
                    Push <Text style={styles.code}>{m.assetName}</Text> to{'\n'}
                    <Text style={styles.code}>{MODELS_DIR}/</Text>
                  </Text>
                  {m.downloadUrl ? (
                    <Text style={styles.link} onPress={() => Linking.openURL(m.downloadUrl!)}>
                      {m.downloadUrl}
                    </Text>
                  ) : null}
                </View>
              ) : null}
            </Card>
          </Pressable>
        );
      })}

      <Muted style={{ marginTop: spacing.sm, marginBottom: spacing.xl }}>
        Add a model: drop its .onnx in the models dir above and add one entry to
        src/models/registry.ts (input size, channel order, normalization, embed dim).
      </Muted>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: theme.bg },
  content: { padding: spacing.lg },
  header: { flexDirection: 'row', alignItems: 'center' },
  radio: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: theme.accent,
    marginRight: spacing.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  radioDot: { width: 10, height: 10, borderRadius: 5, backgroundColor: theme.accent },
  title: { color: theme.text, fontSize: 16, fontWeight: '700' },
  sub: { color: theme.textDim, fontSize: 12, marginTop: 1 },
  specs: { marginTop: spacing.md },
  specRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 2 },
  specKey: { color: theme.textFaint, fontSize: 12 },
  specVal: { color: theme.textDim, fontSize: 12, fontFamily: mono },
  fetch: {
    marginTop: spacing.md,
    padding: spacing.md,
    backgroundColor: theme.surfaceAlt,
    borderRadius: 8,
  },
  fetchText: { color: theme.textDim, fontSize: 12, lineHeight: 18 },
  code: { fontFamily: mono, color: theme.text, fontSize: 11 },
  link: { color: theme.accent, fontSize: 11, marginTop: 6, fontFamily: mono },
});
