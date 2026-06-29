/**
 * concurrency — tiny async primitives for the scan pipeline.
 *
 *   • Mutex   — serialize an async section (we use it to guarantee only ONE ORT
 *               `session.run` executes at a time; concurrent runs on the native
 *               ORT bridge have a SIGABRT history on this project).
 *   • runPool — process indices 0..count-1 with at most `concurrency` tasks in flight.
 *               Tasks complete out of order (that's the point — while one awaits a
 *               native call, another's JS work runs), so each task must carry its own
 *               index. Workers stop pulling new work as soon as `shouldStop()` is true.
 */

export class Mutex {
  private tail: Promise<void> = Promise.resolve();

  /** Run `fn` once all previously-queued sections have finished. */
  async run<T>(fn: () => Promise<T>): Promise<T> {
    const prev = this.tail;
    let release!: () => void;
    this.tail = new Promise<void>(r => (release = r));
    await prev;
    try {
      return await fn();
    } finally {
      release();
    }
  }
}

/** Reject if `p` doesn't settle within `ms`. Bounds a hung native call (a malformed
 *  image that wedges ImageEditor/ORT) so it can't freeze a whole worker indefinitely. */
export async function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout>;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
  });
  try {
    return await Promise.race([p, timeout]);
  } finally {
    clearTimeout(timer!);
  }
}

export async function runPool(
  count: number,
  concurrency: number,
  task: (index: number) => Promise<void>,
  shouldStop?: () => boolean,
): Promise<void> {
  let next = 0;
  const worker = async (): Promise<void> => {
    for (;;) {
      if (shouldStop?.()) return;
      const i = next++;
      if (i >= count) return;
      await task(i);
    }
  };
  const workers = Math.max(1, Math.min(Math.floor(concurrency) || 1, count));
  await Promise.all(Array.from({ length: workers }, () => worker()));
}
