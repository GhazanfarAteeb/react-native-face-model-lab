# Off-thread preprocessing (worklet) — activation guide

The scan pipeline runs detection + inference off the JS thread already (native bridges),
but the **pixel math** — jpeg decode, `preprocess`, the arcface bilinear warp — runs on the
single JS thread between native calls. On a low-end device a burst of that work can blow a
frame budget and make the UI stutter. Moving it to a **worklet** (a second JS runtime on its
own thread) keeps the UI thread free *by construction*, on every device, both platforms.

The codebase is already wired for this behind a seam (`src/pipeline/processor.ts`). The scan
calls `getImageProcessor().preprocess(...)`; by default that's the inline main-thread
implementation. Activating the worklet is **one call at app start** —
`setImageProcessor(workletProcessor)` — with no change to the scan code.

> ⚠️ This needs a native dependency + a full rebuild, and could not be compiled/verified in
> the environment where the seam was written. Treat the code below as a correct-by-inspection
> starting point and verify the worklet-runtime API against the version you install.

---

## 1. Install

Recommended package: **`react-native-worklets-core`** (the "run a JS function on a background
thread and await its result" library, as used by VisionCamera). `react-native-worklets`
(the Reanimated-4 split) also works but is oriented toward UI worklets.

```sh
npm install react-native-worklets-core
# iOS
cd ios && pod install && cd ..
# Add the Babel plugin (required so 'worklet' functions are transformed):
#   babel.config.js → plugins: ['react-native-worklets-core/plugin']
# Then a FULL rebuild (New Architecture codegen):
npm run ios      # or: npm run android
```

Metro will not bundle the worklet module until something imports it, so the app keeps
building until you wire step 3 — install + rebuild first, then wire.

---

## 2. The worklet processor

Create `src/pipeline/processor.worklet.ts`. The kernel is a self-contained `'worklet'`
function (worklets can't import modules, so the normalization logic from `preprocess.ts` is
inlined). It takes an `ArrayBuffer` + primitives and returns a `Float32Array` — all
worklet-shareable.

```ts
import { Worklets } from 'react-native-worklets-core';
import type { ImageProcessor } from './processor';
import type { ModelSpec, RgbaImage } from '../types';

const ctx = Worklets.createContext('FMLPreprocess');

// Pure, self-contained — mirrors src/pipeline/preprocess.ts. Runs on the worker thread.
function preprocessKernel(
  rgba: ArrayBuffer,
  W: number,
  H: number,
  nchw: boolean,
  bgr: boolean,
  prewhiten: boolean,
  meanIn: number[],
  stdIn: number[],
): ArrayBuffer {
  'worklet';
  const src = new Uint8Array(rgba);
  const srcOrder = bgr ? [2, 1, 0] : [0, 1, 2];
  const plane = W * H;
  let mean = meanIn;
  let std = stdIn;
  if (prewhiten) {
    let sum = 0;
    let sumSq = 0;
    for (let p = 0; p < plane; p++) {
      const s = p * 4;
      for (let c = 0; c < 3; c++) {
        const v = src[s + c];
        sum += v;
        sumSq += v * v;
      }
    }
    const n = plane * 3;
    const m = sum / n;
    const variance = Math.max(0, sumSq / n - m * m);
    const sd = Math.max(Math.sqrt(variance), 1 / Math.sqrt(n));
    mean = [m, m, m];
    std = [sd, sd, sd];
  }
  const out = new Float32Array(W * H * 3);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const px = y * W + x;
      for (let c = 0; c < 3; c++) {
        const v = (src[px * 4 + srcOrder[c]] - mean[c]) / std[c];
        if (nchw) out[c * plane + px] = v;
        else out[px * 3 + c] = v;
      }
    }
  }
  return out.buffer;
}

const runOnWorker = Worklets.createRunInContextFn(preprocessKernel, ctx);

function triple(v: number | [number, number, number]): number[] {
  return typeof v === 'number' ? [v, v, v] : [v[0], v[1], v[2]];
}

export const workletProcessor: ImageProcessor = {
  async preprocess(rgba: RgbaImage, spec: ModelSpec): Promise<Float32Array> {
    const { width: W, height: H, layout, channels } = spec.input;
    const buf = await runOnWorker(
      // Copy out a tight ArrayBuffer (the decoded buffer may be larger than W*H*4).
      rgba.data.buffer.slice(rgba.data.byteOffset, rgba.data.byteOffset + W * H * 4),
      W,
      H,
      layout === 'NCHW',
      channels === 'BGR',
      !!spec.norm.prewhiten,
      triple(spec.norm.mean),
      triple(spec.norm.std),
    );
    return new Float32Array(buf);
  },
};
```

> API note: `Worklets.createContext` / `createRunInContextFn` is the worklets-core ≤1.x
> shape. Newer versions expose `context.runAsync(() => { 'worklet'; ... })` instead — adapt
> the two `Worklets.*` lines accordingly. The `preprocessKernel` body is package-agnostic.

---

## 3. Activate at app start

In `App.tsx` (or any module that runs once at startup):

```ts
import { setImageProcessor } from './src/pipeline/processor';
import { workletProcessor } from './src/pipeline/processor.worklet';

setImageProcessor(workletProcessor); // off-thread
// setImageProcessor(null);          // revert to inline main-thread
```

Nothing else changes — `embedFace` already awaits the seam.

---

## 4. Verify

Run a scan and watch the **UI-thread health meter** on the live screen:

- Inline (before): on a low-end device + high parallelism it dips amber/red on heavy configs.
- Worklet (after): it should stay **green** even at higher parallelism, because the pixel
  math no longer runs on the UI thread.

Also confirm matches are **identical** to inline — the kernel is the same math, so a model +
references + threshold combo must produce the same matches. If they differ, the marshaling
(channel order, layout, buffer offset) is wrong.

---

## 5. Going further (bigger wins for heavy configs)

The seam above offloads `preprocess`. The remaining JS-thread pixel work, in order of impact:

1. **Detector channel-packing** (`src/pipeline/detector.ts`) — the YuNet/SCRFD/BlazeFace path
   builds a 640²×3 (~1.2M-iteration) input array on the JS thread. That's the single largest
   sync block; give it the same worklet treatment. (ML Kit, the default detector, does this
   natively and is unaffected.)
2. **jpeg decode** (`src/pipeline/decode.ts`) — move the `jpeg-js` decode into the worklet too
   (pass the file bytes in, decode + preprocess in one worker hop) to remove another main-
   thread cost and avoid shipping the decoded buffer across threads twice.
3. **arcface bilinear warp** (`src/pipeline/crop.ts`) — pure resample over W·H; worklet-able
   the same way, relevant when alignment is `arcface`.

Each is the same pattern: extract a self-contained `'worklet'` kernel over ArrayBuffers +
primitives, run it on the context, await the result. The `ImageProcessor` interface can grow
methods (e.g. `decodeAndPreprocess`, `warp`) as you offload more.
