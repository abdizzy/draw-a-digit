// Verifies the JS engine (web/engine.mjs + web/model.mjs) against the
// Rust-trained model on MNIST test images. Run: node web/tests/engine.test.mjs

import { readFileSync } from 'node:fs';
import { Mlp } from '../engine.mjs';
import { parseModel } from '../model.mjs';

const modelBuf = readFileSync(new URL('../model.bin', import.meta.url));
const net = new Mlp(parseModel(modelBuf));

const imgBuf = readFileSync(new URL('../../mnist/t10k-images-idx3-ubyte', import.meta.url));
const labBuf = readFileSync(new URL('../../mnist/t10k-labels-idx1-ubyte', import.meta.url));
const count = imgBuf.readUInt32BE(4);
let correct = 0;
for (let i = 0; i < count; i++) {
  const off = 16 + i * 784;
  const x = new Float32Array(784);
  for (let p = 0; p < 784; p++) x[p] = imgBuf[off + p] / 255;
  if (net.predict(x) === labBuf[8 + i]) correct++;
}
const acc = correct / count;
console.log(`JS engine accuracy on ${count} MNIST test images: ${(acc * 100).toFixed(2)}%`);
if (acc < 0.9) {
  console.error('FAIL: expected accuracy >= 90% — check the model or the engine');
  process.exit(1);
}
console.log('engine test passed');
