import { Mlp } from './engine.mjs';
import { parseModel } from './model.mjs';

const out = document.getElementById('bench');

async function run() {
  const res = await fetch('model.bin');
  if (!res.ok) {
    out.textContent = 'model.bin not found — train with `nn train` first';
    return;
  }
  const net = new Mlp(parseModel(await res.arrayBuffer()));
  const x = new Float32Array(784);
  for (let i = 0; i < x.length; i++) x[i] = Math.random();

  const N = 2000;
  const warmup = 200;
  for (let i = 0; i < warmup; i++) net.predictProbs(x);
  const t0 = performance.now();
  for (let i = 0; i < N; i++) net.predictProbs(x);
  const total = performance.now() - t0;
  const per = total / N;
  out.textContent =
    `${N} predictions on a random input: average ${per.toFixed(3)} ms/prediction (~${Math.round(1000 / per)} predictions/s)`;
}

run().catch((e) => {
  out.textContent = `benchmark failed: ${e.message}`;
});
