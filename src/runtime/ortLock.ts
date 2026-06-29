/**
 * ortLock — one app-wide mutex serializing every ONNX Runtime `session.run`.
 *
 * The scan pipeline now runs several photos concurrently (see runPool) so that native
 * I/O, ML Kit detection and JS preprocessing overlap. ORT inference, however, must NOT
 * overlap: concurrent `session.run` calls on the native onnxruntime-react-native bridge
 * can hard-abort (SIGABRT, uncatchable by JS) — the same failure mode this project
 * already fought with execution providers. Routing every embedder/detector run through
 * this single lock keeps at most one ORT call in flight while everything else parallelizes.
 */
import { Mutex } from '../scan/concurrency';

export const ortLock = new Mutex();
