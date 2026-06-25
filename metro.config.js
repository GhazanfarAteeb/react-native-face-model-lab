const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * @type {import('@react-native/metro-config').MetroConfig}
 */
const defaultConfig = getDefaultConfig(__dirname);

const config = {
  resolver: {
    // Let Metro bundle model weights as assets (also lets you require() one if you
    // ever want to). Models are normally loaded from disk, so this is just belt-and-braces.
    assetExts: [...defaultConfig.resolver.assetExts, 'onnx', 'ort', 'tflite', 'bin'],
  },
};

module.exports = mergeConfig(defaultConfig, config);
