/**
 * Model registry — the SINGLE place that defines every benchmarkable model.
 *
 * Adding a model is one entry here + dropping its weight file in either:
 *   • assets/models/<assetName>   (bundled: true  → shipped in the app, loaded via require)
 *   • <DocumentDir>/models/<assetName>  (bundled: false → push the file at runtime)
 *
 * The specs below (input size, channel order, normalization, layout) come from each
 * model's published preprocessing. The MobileFaceNet row is verified against rnbaby's
 * production constants. ⚠️ For the fetched models, CONFIRM the normalization against the
 * source repo before trusting a comparison — a wrong mean/std silently produces garbage
 * embeddings (low AUC) that look like "this model is bad" when it's really mis-fed.
 *
 * We always L2-normalize the model output ourselves (see pipeline), so `l2normalized`
 * is informational only — re-normalizing an already-unit vector is a no-op.
 */
import type { ModelSpec } from '../types';

export const MODEL_REGISTRY: ModelSpec[] = [
  {
    id: 'mobilefacenet',
    label: 'MobileFaceNet (baseline)',
    family: 'MobileFaceNet',
    runtime: 'onnx',
    assetName: 'mobilefacenet.onnx',
    bundled: true,
    input: { width: 112, height: 112, layout: 'NCHW', channels: 'RGB' },
    norm: { mean: 127.5, std: 128.0 },
    output: { dim: 512, l2normalized: false, inputName: 'input.1', outputName: '516' },
    align: 'arcface',
    license: 'InsightFace (non-commercial / research)',
    notes: 'rnbaby production model (InsightFace w600k_mbf). The bar to beat.',
    enabled: true,
  },
  {
    id: 'facenet',
    label: 'FaceNet (InceptionResnetV1)',
    family: 'FaceNet',
    runtime: 'onnx',
    assetName: 'facenet.onnx',
    bundled: true,
    // Verified from the .onnx: Keras FaceNet — input "image_input" [n,160,160,3] (NHWC),
    // output "Bottleneck_BatchNorm" 128-D, per-image prewhiten standardization.
    input: { width: 160, height: 160, layout: 'NHWC', channels: 'RGB' },
    norm: { mean: 0, std: 1, prewhiten: true },
    output: { dim: 128, l2normalized: false, inputName: 'image_input', outputName: 'Bottleneck_BatchNorm' },
    align: 'bbox',
    license: 'MIT (keras-facenet); VGGFace2 weights research-use',
    notes:
      'From rnbaby2 (89 MB — the largest bundled model). 160² NHWC, 128-D, prewhiten. Reference point — older/heavier than the ArcFace family.',
    enabled: true,
  },
  {
    id: 'sface',
    label: 'SFace (OpenCV Zoo)',
    family: 'SFace',
    runtime: 'onnx',
    assetName: 'sface_2021dec.onnx',
    bundled: true,
    input: { width: 112, height: 112, layout: 'NCHW', channels: 'BGR' },
    // SFace consumes raw 0–255 BGR (the model scales internally). mean 0 / std 1.
    norm: { mean: 0, std: 1 },
    // Verified from the .onnx: input "data" [1,3,112,112] → output "fc1" [1,128].
    output: { dim: 128, l2normalized: false, inputName: 'data', outputName: 'fc1' },
    align: 'arcface',
    license: 'Apache-2.0 (commercial-friendly)',
    notes: 'OpenCV Zoo (2021dec). BGR + raw 0–255, 128-D. Apache-2.0 — cleanest licence here.',
    downloadUrl:
      'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx',
    enabled: true,
  },
  {
    id: 'edgeface_s',
    label: 'EdgeFace-S (γ=0.5)',
    family: 'EdgeFace',
    runtime: 'onnx',
    assetName: 'edgeface_s_gamma_05.onnx',
    bundled: true,
    input: { width: 112, height: 112, layout: 'NCHW', channels: 'RGB' },
    // EdgeFace: /255 then (x-0.5)/0.5  ≡  (x-127.5)/127.5 on 0–255 input.
    norm: { mean: 127.5, std: 127.5 },
    // Verified from the exported .onnx: input "input" [b,3,112,112] → output "embedding" 512-D.
    output: { dim: 512, l2normalized: false, inputName: 'input', outputName: 'embedding' },
    align: 'arcface',
    license: 'Idiap research licence (verify commercial terms)',
    notes:
      'Exported from the official .pt (Idiap/EdgeFace-S-GAMMA) via scripts/export_edgeface.py and verified. RGB, (x−127.5)/127.5.',
    downloadUrl: 'https://huggingface.co/Idiap/EdgeFace-S-GAMMA',
    enabled: true,
  },
  {
    id: 'ghostfacenet_v2',
    label: 'GhostFaceNet V2',
    family: 'GhostFaceNet',
    runtime: 'onnx',
    assetName: 'ghostfacenet_v2.onnx',
    bundled: true,
    input: { width: 112, height: 112, layout: 'NCHW', channels: 'RGB' },
    norm: { mean: 127.5, std: 127.5 },
    // Verified from the .onnx: input "input" [b,3,112,112] → output "embedding" (512-D,
    // self-normalized via an in-graph Div, so re-L2-norm is a no-op).
    output: { dim: 512, l2normalized: true, inputName: 'input', outputName: 'embedding' },
    align: 'arcface',
    license: 'MIT (code)',
    notes:
      'Community ONNX (HF raj0120/edge-face-pipeline). RGB, (x−127.5)/127.5. If matches look poor, the export may expect /255 or BGR — flip channels/norm here and rescan.',
    downloadUrl: 'https://huggingface.co/raj0120/edge-face-pipeline',
    enabled: true,
  },
  {
    id: 'facelivt',
    label: 'FaceLiVT v2-S (ICIP 2025)',
    family: 'FaceLiVT',
    runtime: 'onnx',
    assetName: 'facelivt_v2_s.onnx',
    bundled: true,
    input: { width: 112, height: 112, layout: 'NCHW', channels: 'RGB' },
    norm: { mean: 127.5, std: 127.5 },
    // Exported from the official .pt via scripts/export_facelivt.py and verified:
    // input "data" [b,3,112,112] → output "embedding" 512-D. (The community ONNX
    // rifatrahman378/facelivt-onnx is broken — it outputs a 1284-D intermediate.)
    output: { dim: 512, l2normalized: false, inputName: 'data', outputName: 'embedding' },
    align: 'arcface',
    license: 'novendrastywn/FaceLiVT (research; InsightFace-based)',
    notes:
      'Hybrid linear-ViT, newest/fastest on paper. Exported from the official PyTorch weights (FaceLiVTv2-Small, 4.62M params). RGB, (x−127.5)/127.5.',
    downloadUrl: 'https://github.com/novendrastywn/FaceLiVT',
    enabled: true,
  },
];

export function getModel(id: string): ModelSpec | undefined {
  return MODEL_REGISTRY.find(m => m.id === id);
}
