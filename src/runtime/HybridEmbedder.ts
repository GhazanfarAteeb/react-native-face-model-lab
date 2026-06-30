/**
 * HybridEmbedder — run ORT-CPU and native CoreML/ANE in parallel, balancing automatically.
 *
 * This is the self-balancing dual-queue: faces arrive concurrently (the scan worker pool calls
 * `infer` from several photos at once) and each one is routed to whichever backend currently has
 * the fewest in-flight. Each backend serializes its own runs (its instance lock), so "fewest
 * in-flight" means "the one that will start me soonest" — the faster engine naturally drains its
 * queue quicker and so keeps pulling a larger share. No fixed CPU:ANE ratio, no thermal tuning:
 * if the ANE is 1.5× the CPU it ends up doing ~1.5× the work; if the CPU throttles, the ANE
 * picks up the slack — exactly the "whoever's free grabs the next one" behaviour.
 *
 * `loadMs`/`fileSizeBytes`/`spec` are reported from the CPU backend (the always-present one).
 */
import type { Embedder, InferResult } from './Embedder';
import type { ModelSpec } from '../types';

export class HybridEmbedder implements Embedder {
  spec: ModelSpec;
  loadMs: number;
  fileSizeBytes: number;

  private backends: Embedder[];
  private inflight: number[];

  constructor(cpu: Embedder, ane: Embedder) {
    this.backends = [cpu, ane];
    this.inflight = [0, 0];
    this.spec = cpu.spec;
    // Surface the slower (larger) load so the headline reflects readiness of both engines.
    this.loadMs = Math.max(cpu.loadMs, ane.loadMs);
    this.fileSizeBytes = cpu.fileSizeBytes;
  }

  async infer(input: Float32Array): Promise<InferResult> {
    // Route to the least-loaded backend. Ties favour index 0 (CPU), which is fine — the next
    // concurrent call will see the bump and pick the other.
    const i = this.inflight[0] <= this.inflight[1] ? 0 : 1;
    this.inflight[i] += 1;
    try {
      return await this.backends[i].infer(input);
    } finally {
      this.inflight[i] -= 1;
    }
  }

  async release(): Promise<void> {
    await Promise.all(this.backends.map(b => b.release().catch(() => {})));
  }
}
