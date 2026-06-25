/** Shared UI atoms for the benchmark tool. Dark, compact, monospace numbers. */
import React from 'react';
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  View,
  ViewStyle,
} from 'react-native';
import { theme, spacing, radius, mono } from './theme';

export function Card({ children, style }: { children: React.ReactNode; style?: ViewStyle }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

export function SectionLabel({ children }: { children: React.ReactNode }) {
  return <Text style={styles.sectionLabel}>{children}</Text>;
}

export function Muted({ children, style }: { children: React.ReactNode; style?: object }) {
  return <Text style={[styles.muted, style]}>{children}</Text>;
}

export function Button({
  title,
  onPress,
  variant = 'primary',
  disabled,
  loading,
}: {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'danger';
  disabled?: boolean;
  loading?: boolean;
}) {
  const bg =
    variant === 'primary' ? theme.accent : variant === 'danger' ? theme.bad : theme.surfaceAlt;
  const fg = variant === 'secondary' ? theme.text : '#fff';
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled || loading}
      style={({ pressed }) => [
        styles.button,
        { backgroundColor: bg, opacity: disabled ? 0.4 : pressed ? 0.85 : 1 },
      ]}>
      {loading ? (
        <ActivityIndicator color={fg} size="small" />
      ) : (
        <Text style={[styles.buttonText, { color: fg }]}>{title}</Text>
      )}
    </Pressable>
  );
}

export function Chip({
  label,
  active,
  onPress,
}: {
  label: string;
  active?: boolean;
  onPress?: () => void;
}) {
  return (
    <Pressable
      onPress={onPress}
      style={[styles.chip, active && { backgroundColor: theme.accentDim, borderColor: theme.accent }]}>
      <Text style={[styles.chipText, active && { color: theme.text }]}>{label}</Text>
    </Pressable>
  );
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { label: string; value: T }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <View style={styles.segmented}>
      {options.map(o => (
        <Pressable
          key={o.value}
          onPress={() => onChange(o.value)}
          style={[styles.segment, value === o.value && styles.segmentActive]}>
          <Text style={[styles.segmentText, value === o.value && { color: '#fff' }]}>{o.label}</Text>
        </Pressable>
      ))}
    </View>
  );
}

export function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <View style={styles.stat}>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
      {hint ? <Text style={styles.statHint}>{hint}</Text> : null}
    </View>
  );
}

export function Badge({ text, tone = 'dim' }: { text: string; tone?: 'good' | 'warn' | 'bad' | 'dim' }) {
  const color =
    tone === 'good' ? theme.good : tone === 'warn' ? theme.warn : tone === 'bad' ? theme.bad : theme.textFaint;
  return (
    <View style={[styles.badge, { borderColor: color }]}>
      <Text style={[styles.badgeText, { color }]}>{text}</Text>
    </View>
  );
}

export function ProgressBar({ fraction }: { fraction: number }) {
  return (
    <View style={styles.progressTrack}>
      <View style={[styles.progressFill, { width: `${Math.max(0, Math.min(1, fraction)) * 100}%` }]} />
    </View>
  );
}

export function Divider() {
  return <View style={styles.divider} />;
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: theme.surface,
    borderRadius: radius.lg,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: theme.border,
    padding: spacing.lg,
    marginBottom: spacing.md,
  },
  sectionLabel: {
    color: theme.textDim,
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 1,
    textTransform: 'uppercase',
    marginBottom: spacing.sm,
  },
  muted: { color: theme.textDim, fontSize: 13, lineHeight: 18 },
  button: {
    paddingVertical: 13,
    paddingHorizontal: spacing.lg,
    borderRadius: radius.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonText: { fontSize: 15, fontWeight: '700' },
  chip: {
    paddingVertical: 7,
    paddingHorizontal: 13,
    borderRadius: 999,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: theme.border,
    backgroundColor: theme.chip,
    marginRight: spacing.sm,
    marginBottom: spacing.sm,
  },
  chipText: { color: theme.textDim, fontSize: 13, fontWeight: '600' },
  segmented: {
    flexDirection: 'row',
    backgroundColor: theme.chip,
    borderRadius: radius.md,
    padding: 3,
  },
  segment: { flex: 1, paddingVertical: 9, alignItems: 'center', borderRadius: radius.sm },
  segmentActive: { backgroundColor: theme.accent },
  segmentText: { color: theme.textDim, fontSize: 13, fontWeight: '600' },
  stat: { minWidth: 92, marginRight: spacing.lg, marginBottom: spacing.sm },
  statValue: { color: theme.text, fontSize: 20, fontWeight: '700', fontFamily: mono },
  statLabel: { color: theme.textDim, fontSize: 11, marginTop: 2 },
  statHint: { color: theme.textFaint, fontSize: 10, marginTop: 1 },
  badge: {
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: radius.sm,
    paddingHorizontal: 7,
    paddingVertical: 2,
  },
  badgeText: { fontSize: 10, fontWeight: '700', letterSpacing: 0.4 },
  progressTrack: {
    height: 8,
    backgroundColor: theme.chip,
    borderRadius: 999,
    overflow: 'hidden',
  },
  progressFill: { height: '100%', backgroundColor: theme.accent, borderRadius: 999 },
  divider: { height: StyleSheet.hairlineWidth, backgroundColor: theme.border, marginVertical: spacing.md },
});
