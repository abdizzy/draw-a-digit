// Web port of the Rust engine (nn/src/engine.rs): Linear/Mlp forward pass,
// softmax and argmax. Pure JS, no dependencies.

class Linear {
  constructor(inDim, outDim) {
    this.inDim = inDim;
    this.outDim = outDim;
    this.w = new Float32Array(inDim * outDim);
    this.b = new Float32Array(outDim);
  }

  // y = W x + b
  forward(x) {
    const y = this.b.slice();
    for (let o = 0; o < this.outDim; o++) {
      const base = o * this.inDim;
      let s = y[o];
      for (let i = 0; i < this.inDim; i++) {
        s += this.w[base + i] * x[i];
      }
      y[o] = s;
    }
    return y;
  }
}

class Mlp {
  // model: { inputDim, outputDim, hiddenAct, layers: [{ inDim, outDim, w, b }] }
  constructor(model) {
    this.inputDim = model.inputDim;
    this.outputDim = model.outputDim;
    this.hiddenAct = model.hiddenAct;
    this.layers = model.layers.map((l) => {
      const layer = new Linear(l.inDim, l.outDim);
      layer.w.set(l.w);
      layer.b.set(l.b);
      return layer;
    });
  }

  #applyAct(z) {
    const out = new Float32Array(z.length);
    for (let i = 0; i < z.length; i++) {
      const v = z[i];
      out[i] = this.hiddenAct === 'relu' ? (v > 0 ? v : 0) : 1 / (1 + Math.exp(-v));
    }
    return out;
  }

  // Forward pass: returns the logits of the output layer.
  forward(x) {
    const n = this.layers.length;
    let cur = new Float32Array(x);
    for (let l = 0; l < n; l++) {
      const z = this.layers[l].forward(cur);
      if (l + 1 === n) return z;
      cur = this.#applyAct(z);
    }
    throw new Error('unreachable');
  }

  predictProbs(x) {
    return softmax(this.forward(x));
  }

  predict(x) {
    return argmax(this.predictProbs(x));
  }
}

export { Linear, Mlp };

// Stable softmax: subtracts the max before exponentiating.
export function softmax(logits) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > max) max = logits[i];
  }
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.exp(logits[i] - max);
    sum += out[i];
  }
  if (sum > 0) {
    for (let i = 0; i < out.length; i++) out[i] /= sum;
  }
  return out;
}

export function argmax(v) {
  let best = 0;
  for (let i = 1; i < v.length; i++) {
    if (v[i] > v[best]) best = i;
  }
  return best;
}
