# Background scanning — activation guide

Goal: keep a scan running when the app is backgrounded / screen-off, and finish (or make
progress) without the user staring at it.

**This is asymmetric by OS — and that's a platform limit, not a missing feature:**

| | Android | iOS |
|---|---|---|
| True background compute | ✅ Foreground Service (persistent notification) — runs to completion | ❌ Not allowed |
| What you actually get | Run the whole scan in the background | ~30s grace after backgrounding + opportunistic `BGProcessingTask` (runs only when idle/charging, system-scheduled) |

So: **Android can finish a scan in the background; iOS can only chip away opportunistically
and finish when the user reopens the app.** Both rely on the **checkpoint/resume** that's
already shipped (`src/scan/checkpoint.ts`) — the scan stops and resumes losslessly, which is
exactly what a background task that may be paused/killed needs.

> ⚠️ Needs native dependencies + a full rebuild, and can't be compiled/verified from the
> environment this guide was written in. Verify package APIs against the versions you install.

---

## 0. Prerequisite — make the scan request headless-resumable

A background task has no React tree, so it must reconstruct `runScan` params without the
store. The checkpoint already persists `photoUris` + a config `signature`; extend it to carry
the full request so a headless task can rebuild params:

```ts
// checkpoint.ts → ScanCheckpoint, add:
request: {
  modelId: string;
  settings: ScanSettings;
  references: ReferenceImage[]; // uri + bucket (embeddings are recomputed)
};
```

Populate it in `store.tsx` where the checkpoint is built (you already have `selectedModel`,
`settings`, `references` there). Then a headless runner can do:

```ts
// src/scan/headlessScan.ts
import { loadCheckpoint, saveCheckpoint, clearCheckpoint } from './checkpoint';
import { runScan } from './scanner';
import { MODEL_REGISTRY } from '../models/registry';

export async function resumeScanHeadless(shouldStop: () => boolean) {
  const cp = await loadCheckpoint();
  if (!cp || cp.doneIndices.length >= cp.photoUris.length) return;
  const spec = MODEL_REGISTRY.find(m => m.id === cp.request.modelId) ?? MODEL_REGISTRY[0];
  await runScan({
    spec,
    references: cp.request.references,
    photoUris: cp.photoUris,
    settings: cp.request.settings,
    resume: { doneIndices: cp.doneIndices, matches: cp.matches, facesFound: cp.facesFound, sepSum: cp.sepSum, sepCount: cp.sepCount },
    onCheckpoint: state => saveCheckpoint({ ...cp, ...state }),
    shouldCancel: shouldStop,
  });
  if (!shouldStop()) await clearCheckpoint();
}
```

This is pure TS and could be added + verified now — it's the bridge both platforms use.

---

## 1. Android — Foreground Service (true background)

Recommended: **`react-native-background-actions`** (runs an async JS task inside a foreground
service with a persistent notification).

```sh
npm install react-native-background-actions
# Android: it autolinks; add the FOREGROUND_SERVICE permissions to AndroidManifest.xml
#   <uses-permission android:name="android.permission.FOREGROUND_SERVICE"/>
#   <uses-permission android:name="android.permission.POST_NOTIFICATIONS"/>  (Android 13+)
```

Start it when a scan begins (instead of, or alongside, the in-app run):

```ts
import BackgroundService from 'react-native-background-actions';
import { resumeScanHeadless } from './src/scan/headlessScan';

let stop = false;
await BackgroundService.start(
  async () => {
    stop = false;
    await resumeScanHeadless(() => stop);
    await BackgroundService.stop();
  },
  { taskName: 'FaceScan', taskTitle: 'Scanning photos…', taskDesc: 'Finding matches',
    taskIcon: { name: 'ic_launcher', type: 'mipmap' } },
);
// to cancel: stop = true; BackgroundService.stop();
```

Update the notification from `onProgress` via `BackgroundService.updateNotification({ taskDesc: \`${i}/${total}\` })`. The scan now survives backgrounding and screen-off; if Android kills it under memory pressure, the checkpoint resumes it next launch.

---

## 2. iOS — `BGProcessingTask` (opportunistic only)

iOS will not run the scan in the background on demand. The realistic design:

- **Foreground:** run normally (already does), at full speed.
- **Backgrounding:** the ~30s `beginBackgroundTask` grace is enough to flush a checkpoint — the
  periodic checkpoints already cover this, so nothing extra is strictly required.
- **Opportunistic continuation:** register a `BGProcessingTask` that, when the OS runs it
  (device idle, usually charging), calls `resumeScanHeadless` for the granted window and
  re-schedules itself if not finished.

Easiest wiring: **`react-native-background-fetch`** (supports scheduling a `BGProcessingTask`,
not just fetch):

```sh
npm install react-native-background-fetch
cd ios && pod install && cd ..
# Xcode: enable Background Modes → Background processing; add the task id to Info.plist
#   <key>BGTaskSchedulerPermittedIdentifiers</key><array><string>com.fml.scan</string></array>
```

```ts
import BackgroundFetch from 'react-native-background-fetch';

BackgroundFetch.scheduleTask({ taskId: 'com.fml.scan', delay: 0, requiresCharging: false });

BackgroundFetch.configure({ minimumFetchInterval: 15 }, async taskId => {
  let timedOut = false;
  // The OS gives a limited window; bail cooperatively, the checkpoint keeps progress.
  setTimeout(() => { timedOut = true; }, 25_000);
  await resumeScanHeadless(() => timedOut);
  // Not done yet? ask the OS to run us again later.
  const cp = await /* loadCheckpoint */ null;
  if (cp /* && not complete */) BackgroundFetch.scheduleTask({ taskId: 'com.fml.scan', delay: 0 });
  BackgroundFetch.finish(taskId);
}, () => {}, );
```

Set expectations in the UI on iOS: "continues when you reopen the app; may make progress in
the background when charging." Don't promise on-demand background completion — it will fail
review and on-device reality.

---

## 3. Verify

- **Android:** start a scan, background the app, lock the screen. The notification persists and
  progress advances; reopen → it's further along or done. Kill from recents mid-scan → reopen
  → Resume card (already implemented) shows remaining count.
- **iOS:** background mid-scan, reopen within a minute → resumes from the last checkpoint
  (≤20 photos redone). For `BGProcessingTask`, use Xcode's
  `e -l objc -- (void)[[BGTaskScheduler sharedScheduler] _simulateLaunchForTaskWithIdentifier:@"com.fml.scan"]`
  to force a run and confirm `resumeScanHeadless` advances the checkpoint.

---

## Summary

The heavy lifting (lossless stop/resume) is already done and verified. What remains is native
plumbing: a foreground service on Android (real background), and an opportunistic
`BGProcessingTask` on iOS (best-effort, the platform ceiling). Step 0 (`headlessScan` +
storing the request in the checkpoint) is pure TS and the right first move — it's the seam
both platforms call into.
