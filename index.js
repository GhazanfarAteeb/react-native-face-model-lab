/**
 * @format
 */

// jpeg-js + base64 image decode need a Buffer global in the Hermes runtime.
import { Buffer } from 'buffer';
global.Buffer = global.Buffer || Buffer;

import { AppRegistry } from 'react-native';
import App from './App';
import { name as appName } from './app.json';

AppRegistry.registerComponent(appName, () => App);
