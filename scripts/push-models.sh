#!/usr/bin/env bash
#
# push-models.sh — copy face-model weight files into the app's on-device models dir,
# so non-bundled models (SFace, EdgeFace, GhostFaceNet, FaceNet, …) become loadable.
#
# Source: any .onnx / .tflite files in ./assets/models (or pass paths as arguments).
# Targets the app's Documents/models on a running Android device/emulator and the
# booted iOS simulator. For a physical iOS device, use Finder → Files sharing instead
# (UIFileSharingEnabled is on): drag the files into the FaceModelLab Documents folder.
#
# Usage:
#   ./scripts/push-models.sh                # push everything in assets/models
#   ./scripts/push-models.sh path/to/x.onnx # push specific files
set -euo pipefail

ANDROID_PKG="com.facemodellab"
IOS_BUNDLE="org.reactjs.native.example.FaceModelLab"
DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ "$#" -gt 0 ]; then
  FILES=("$@")
else
  shopt -s nullglob
  FILES=("$DIR"/assets/models/*.onnx "$DIR"/assets/models/*.tflite)
fi

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "No model files found. Put .onnx/.tflite in assets/models or pass paths."
  exit 1
fi

pushed_any=0

# ── Android (adb, debug build only — uses run-as) ──────────────────────────────
if command -v adb >/dev/null 2>&1 && adb get-state >/dev/null 2>&1; then
  echo "▶ Android device detected → pushing to $ANDROID_PKG files/models/"
  adb shell run-as "$ANDROID_PKG" mkdir -p files/models 2>/dev/null || true
  for f in "${FILES[@]}"; do
    base="$(basename "$f")"
    echo "  • $base"
    adb push "$f" "/data/local/tmp/$base" >/dev/null
    adb shell run-as "$ANDROID_PKG" cp "/data/local/tmp/$base" "files/models/$base"
    adb shell rm "/data/local/tmp/$base" >/dev/null 2>&1 || true
  done
  pushed_any=1
else
  echo "▷ No Android device (skip). Start an emulator/device + a debug build to push."
fi

# ── iOS Simulator (booted) ─────────────────────────────────────────────────────
if command -v xcrun >/dev/null 2>&1; then
  if CONTAINER="$(xcrun simctl get_app_container booted "$IOS_BUNDLE" data 2>/dev/null)"; then
    echo "▶ iOS simulator detected → $CONTAINER/Documents/models/"
    mkdir -p "$CONTAINER/Documents/models"
    for f in "${FILES[@]}"; do
      echo "  • $(basename "$f")"
      cp "$f" "$CONTAINER/Documents/models/"
    done
    pushed_any=1
  else
    echo "▷ No booted iOS simulator with the app installed (skip)."
  fi
fi

if [ "$pushed_any" -eq 0 ]; then
  echo
  echo "Nothing pushed. Boot a simulator/emulator with FaceModelLab installed, or use"
  echo "the iOS Files app (Finder file sharing) to drop the files into Documents."
fi
echo "Done."
