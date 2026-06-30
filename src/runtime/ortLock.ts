/**
 * ortLock — one app-wide mutex serializing every onnxruntime-react-native `session.run`.
 *
 * Concurrent runs on the ORT-RN bridge can hard-abort (SIGABRT, uncatchable by JS), so EVERY
 * ORT call — the ONNX detectors (YuNet/SCRFD/BlazeFace) AND the ORT-CPU/CoreML-EP embedder —
 * goes through this single lock. At most one ORT run is in flight while the rest of the
 * pipeline (native I/O, ML Kit detection, JS preprocessing) parallelizes.
 *
 * The native CoreML embedder (CoreMLEmbedder) is a DIFFERENT runtime — it does NOT use this
 * lock, so it can run on the ANE concurrently with an ORT-CPU run. That parallelism is exactly
 * what the HybridEmbedder exploits.
 */
import { Mutex } from '../scan/concurrency';

export const ortLock = new Mutex();
