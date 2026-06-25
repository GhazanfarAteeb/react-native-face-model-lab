/**
 * modelAssets — resolve a ModelSpec to a loadable file path on disk.
 *
 * Mirrors rnbaby's ensureModelFile: bundled models live in the iOS app bundle
 * (MainBundlePath) / Android assets and are copied once to DocumentDir/models;
 * fetched models are placed directly in DocumentDir/models by the user.
 *
 * Every model — bundled or fetched — ends up at the same canonical path, so the
 * runtime has a single load path.
 */
import RNFS from 'react-native-fs';
import { Platform } from 'react-native';
import type { ModelSpec } from '../types';

export const MODELS_DIR = `${RNFS.DocumentDirectoryPath}/models`;

export function modelPath(spec: ModelSpec): string {
  return `${MODELS_DIR}/${spec.assetName}`;
}

async function ensureDir(): Promise<void> {
  const exists = await RNFS.exists(MODELS_DIR);
  if (!exists) await RNFS.mkdir(MODELS_DIR);
}

type RNFSWithAssets = typeof RNFS & {
  copyFileAssets?: (src: string, dest: string) => Promise<void>;
  existsAssets?: (path: string) => Promise<boolean>;
};

/** A file dropped at the Documents root (iOS Files app / adb push) instead of in
 *  models/. We migrate it into models/ on first use so both flows work. */
function rootCandidate(spec: ModelSpec): string {
  return `${RNFS.DocumentDirectoryPath}/${spec.assetName}`;
}

/** True when the model's weight file can be loaded right now (bundled, already in the
 *  models dir, or sitting at the Documents root awaiting migration). */
export async function isModelAvailable(spec: ModelSpec): Promise<boolean> {
  await ensureDir();
  if (await RNFS.exists(modelPath(spec))) return true;
  if (await RNFS.exists(rootCandidate(spec))) return true;
  if (!spec.bundled) return false;
  if (Platform.OS === 'android') {
    const rnfs = RNFS as RNFSWithAssets;
    if (rnfs.existsAssets) return rnfs.existsAssets(spec.assetName);
    return true; // assume present in assets; copy will surface the real error
  }
  return RNFS.exists(`${RNFS.MainBundlePath}/${spec.assetName}`);
}

/**
 * Ensure the weight file is on disk and return its path (no file:// prefix).
 * Throws a user-readable error when a non-bundled model hasn't been fetched yet.
 */
export async function ensureModelFile(spec: ModelSpec): Promise<string> {
  await ensureDir();
  const dest = modelPath(spec);
  if (await RNFS.exists(dest)) return dest;

  // Migrate a file dropped at the Documents root into models/.
  const root = rootCandidate(spec);
  if (await RNFS.exists(root)) {
    await RNFS.moveFile(root, dest);
    return dest;
  }

  if (!spec.bundled) {
    throw new Error(
      `"${spec.label}" file not found.\nPush it to:\n${dest}\n` +
        (spec.downloadUrl ? `\nSource: ${spec.downloadUrl}` : ''),
    );
  }

  // Bundled: copy once from the platform's bundle location. Idempotent — a lost
  // copy race is fine as long as the file ends up present (matches rnbaby).
  try {
    const rnfs = RNFS as RNFSWithAssets;
    if (Platform.OS === 'android' && typeof rnfs.copyFileAssets === 'function') {
      await rnfs.copyFileAssets(spec.assetName, dest);
    } else {
      const bundlePath = `${RNFS.MainBundlePath}/${spec.assetName}`;
      if (!(await RNFS.exists(bundlePath))) {
        throw new Error(
          `Bundled model "${spec.assetName}" not found in the app bundle.\n` +
            `iOS: add assets/models/${spec.assetName} to the Xcode target (Copy Bundle Resources).`,
        );
      }
      await RNFS.copyFile(bundlePath, dest);
    }
  } catch (err) {
    if (!(await RNFS.exists(dest))) throw err;
  }
  return dest;
}

/** Resolve a bundled asset by bare filename (for non-registry files like the YuNet
 *  detector). Same copy-from-bundle / migrate-from-root logic as ensureModelFile. */
export async function ensureNamedAsset(assetName: string): Promise<string> {
  await ensureDir();
  const dest = `${MODELS_DIR}/${assetName}`;
  if (await RNFS.exists(dest)) return dest;
  const root = `${RNFS.DocumentDirectoryPath}/${assetName}`;
  if (await RNFS.exists(root)) {
    await RNFS.moveFile(root, dest);
    return dest;
  }
  try {
    const rnfs = RNFS as RNFSWithAssets;
    if (Platform.OS === 'android' && typeof rnfs.copyFileAssets === 'function') {
      await rnfs.copyFileAssets(assetName, dest);
    } else {
      const bundlePath = `${RNFS.MainBundlePath}/${assetName}`;
      if (!(await RNFS.exists(bundlePath))) {
        throw new Error(
          `Bundled asset "${assetName}" not found.\n` +
            `iOS: add assets/models/${assetName} to the Xcode target (Copy Bundle Resources).`,
        );
      }
      await RNFS.copyFile(bundlePath, dest);
    }
  } catch (err) {
    if (!(await RNFS.exists(dest))) throw err;
  }
  return dest;
}

export async function modelFileSize(spec: ModelSpec): Promise<number> {
  try {
    const stat = await RNFS.stat(modelPath(spec));
    return Number(stat.size) || 0;
  } catch {
    return 0;
  }
}
