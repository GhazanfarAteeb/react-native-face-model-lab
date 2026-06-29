/**
 * scheduler — cooperative yielding so the scan never blocks the JS thread long enough to
 * drop a frame.
 *
 * Even with the concurrent pool, the CPU-bound JS work (jpeg decode, preprocess, the
 * arcface bilinear warp, detector channel-packing) runs on the single JS thread between
 * native `await`s. On a low-end device a run of these back-to-back can exceed a frame
 * budget and make the UI stutter. FrameBudget tracks how long we've run since last
 * returning to the event loop and, once past the budget, yields — letting React commit,
 * gestures land, and JS-driven animations advance before we continue.
 *
 * It does NOT make the work faster; it trades a little throughput for a responsive UI,
 * which is the right call across both low- and high-end devices. (True parallelism needs
 * a second thread — see the worklet/native offload path.)
 */

const yieldToEventLoop = (): Promise<void> =>
  new Promise<void>(resolve => {
    // setImmediate yields with less latency than setTimeout(0) (no timer clamp) and still
    // lets queued native→JS callbacks + React work run before we resume.
    if (typeof setImmediate === 'function') setImmediate(resolve);
    else setTimeout(resolve, 0);
  });

export class FrameBudget {
  private last = Date.now();

  /** budgetMs: max contiguous JS time before we insist on yielding. 8ms leaves ~half a
   *  60fps frame for the UI to actually paint. */
  constructor(private budgetMs = 8) {}

  /** Yield to the event loop iff we've run past the budget since the last yield. Cheap
   *  (a clock read) on the common path, so it's safe to call once per face/photo. */
  async tick(): Promise<void> {
    const now = Date.now();
    if (now - this.last >= this.budgetMs) {
      await yieldToEventLoop();
      this.last = Date.now();
    }
  }
}
