/**
 * JsHealthMeter — a live readout of JS-thread responsiveness while a scan runs.
 *
 * Two independent signals, both driven on the JS thread (NOT the native driver), so they
 * stutter exactly when the scan pipeline hogs the thread:
 *   • a requestAnimationFrame loop measures the worst frame gap over a short window and
 *     reports it as a number + colour (green / amber / red);
 *   • a continuously-rotating square — if it freezes, the JS thread is blocked. This is
 *     the most direct "is the scanning making the UI lag?" cue.
 */
import React, { useEffect, useRef, useState } from 'react';
import { Animated, Easing, StyleSheet, Text, View } from 'react-native';
import { theme, spacing, mono } from './theme';

const WINDOW_MS = 300; // report the worst frame gap seen in each window

function tone(frameMs: number): { color: string; label: string } {
  if (frameMs < 24) return { color: theme.good, label: 'SMOOTH' };
  if (frameMs < 50) return { color: theme.warn, label: 'MINOR JANK' };
  return { color: theme.bad, label: 'LAGGING' };
}

export default function JsHealthMeter() {
  const [worstMs, setWorstMs] = useState(16);
  const spin = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    let raf = 0;
    let alive = true;
    let last = Date.now();
    let windowStart = last;
    let windowWorst = 0;

    const tick = () => {
      const t = Date.now();
      const dt = t - last;
      last = t;
      if (dt > windowWorst) windowWorst = dt;
      if (t - windowStart >= WINDOW_MS) {
        setWorstMs(windowWorst);
        windowStart = t;
        windowWorst = 0;
      }
      if (alive) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    // JS-driven rotation (useNativeDriver:false): freezes when the thread is blocked.
    const loop = Animated.loop(
      Animated.timing(spin, {
        toValue: 1,
        duration: 900,
        easing: Easing.linear,
        useNativeDriver: false,
      }),
    );
    loop.start();

    return () => {
      alive = false;
      cancelAnimationFrame(raf);
      loop.stop();
    };
  }, [spin]);

  const t = tone(worstMs);
  const rotate = spin.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '360deg'] });

  return (
    <View style={styles.row}>
      <Animated.View style={[styles.spinner, { borderColor: t.color, transform: [{ rotate }] }]} />
      <View style={{ flex: 1 }}>
        <Text style={styles.label}>UI THREAD</Text>
        <Text style={[styles.status, { color: t.color }]}>{t.label}</Text>
      </View>
      <Text style={[styles.ms, { color: t.color }]}>{Math.round(worstMs)}ms</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    backgroundColor: theme.surface,
    borderRadius: 10,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: theme.border,
    padding: spacing.md,
  },
  spinner: { width: 22, height: 22, borderWidth: 3, borderRadius: 5 },
  label: { color: theme.textFaint, fontSize: 10, fontWeight: '700', letterSpacing: 1 },
  status: { fontSize: 14, fontWeight: '700', marginTop: 1 },
  ms: { fontSize: 18, fontWeight: '700', fontFamily: mono, minWidth: 64, textAlign: 'right' },
});
