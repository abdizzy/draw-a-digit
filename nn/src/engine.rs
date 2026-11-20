//! Tiny feed-forward network engine.
//!
//! A `Linear` layer computes `y = W x + b` with row-major weights. The `Mlp`
//! stacks hidden layers (an activation applied between layers) plus an output
//! layer of raw logits; training uses cross-entropy loss on softmax outputs
//! via the standard backpropagation chain rule.

use crate::rng::Rng;

pub const INPUT_DIM: usize = 784;
pub const OUTPUT_DIM: usize = 10;

/// One sample: a normalized 28x28 image (0..1) and its label.
pub struct Sample {
    pub image: Vec<f32>,
    pub label: u8,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Activation {
    Relu,
    Sigmoid,
}

/// A linear layer `y = W x + b`.
pub struct Linear {
    pub in_dim: usize,
    pub out_dim: usize,
    /// Weights, row-major: `w[o * in_dim + i]`.
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    /// Gradient accumulators, filled by `backward`.
    pub dw: Vec<f32>,
    pub db: Vec<f32>,
}

impl Linear {
    pub fn new(rng: &mut Rng, in_dim: usize, out_dim: usize) -> Self {
        // PyTorch-style default init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        let bound = 1.0 / (in_dim as f32).sqrt();
        let w: Vec<f32> = (0..in_dim * out_dim)
            .map(|_| rng.uniform(-bound, bound))
            .collect();
        Linear {
            in_dim,
            out_dim,
            dw: vec![0.0; w.len()],
            b: vec![0.0; out_dim],
            db: vec![0.0; out_dim],
            w,
        }
    }

    pub fn zero_grads(&mut self) {
        self.dw.fill(0.0);
        self.db.fill(0.0);
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = self.b.clone();
        for o in 0..self.out_dim {
            let row = &self.w[o * self.in_dim..(o + 1) * self.in_dim];
            let mut s = y[o];
            for (i, &xi) in x.iter().enumerate() {
                s += row[i] * xi;
            }
            y[o] = s;
        }
        y
    }

    fn accumulate_grads(&mut self, delta: &[f32], x: &[f32]) {
        for o in 0..self.out_dim {
            self.db[o] += delta[o];
            for i in 0..self.in_dim {
                self.dw[o * self.in_dim + i] += delta[o] * x[i];
            }
        }
    }

    fn step(&mut self, lr: f32) {
        for (w, g) in self.w.iter_mut().zip(self.dw.iter()) {
            *w -= lr * g;
        }
        for (b, g) in self.b.iter_mut().zip(self.db.iter()) {
            *b -= lr * g;
        }
    }
}

/// Values cached during the forward pass, needed by `backward`.
pub struct Cache {
    /// Input vector to each layer (for layer l, the output of layer l-1, or
    /// the sample itself for the first layer).
    pub inputs: Vec<Vec<f32>>,
    /// Pre-activation `z = W x + b` of each layer.
    pub pre: Vec<Vec<f32>>,
    /// Activation output of each layer: `act(z)` for hidden layers, `z` for
    /// the last layer (its output is the logits).
    pub act: Vec<Vec<f32>>,
}

impl Default for Cache {
    fn default() -> Self {
        Cache {
            inputs: Vec::new(),
            pre: Vec::new(),
            act: Vec::new(),
        }
    }
}

pub struct Mlp {
    pub input_dim: usize,
    pub output_dim: usize,
    pub hidden_act: Activation,
    /// Hidden layers followed by the output layer (raw logits).
    pub layers: Vec<Linear>,
}

impl Mlp {
    pub fn new(
        rng: &mut Rng,
        input_dim: usize,
        hidden: &[usize],
        output_dim: usize,
        hidden_act: Activation,
    ) -> Self {
        let mut layers = Vec::new();
        let mut prev = input_dim;
        for &d in hidden {
            layers.push(Linear::new(rng, prev, d));
            prev = d;
        }
        layers.push(Linear::new(rng, prev, output_dim));
        Mlp {
            input_dim,
            output_dim,
            hidden_act,
            layers,
        }
    }

    pub fn describe(&self) -> String {
        let mut s = String::new();
        for (i, l) in self.layers.iter().enumerate() {
            if i == 0 {
                s.push_str(&self.input_dim.to_string());
            }
            s.push_str(" -> ");
            s.push_str(&l.out_dim.to_string());
        }
        s
    }

    pub fn zero_grads(&mut self) {
        for l in &mut self.layers {
            l.zero_grads();
        }
    }

    pub fn step(&mut self, lr: f32) {
        for l in &mut self.layers {
            l.step(lr);
        }
    }

    /// Forward pass; returns the logits of the output layer.
    pub fn forward_cached(&self, x: &[f32], cache: &mut Cache) -> Vec<f32> {
        cache.inputs.clear();
        cache.pre.clear();
        cache.act.clear();
        let n = self.layers.len();
        let mut cur = x.to_vec();
        for l in 0..n {
            cache.inputs.push(cur);
            let z = self.layers[l].forward(cache.inputs[l].as_slice());
            cache.pre.push(z.clone());
            let next = if l + 1 == n {
                z
            } else {
                apply_activation(&z, self.hidden_act)
            };
            cache.act.push(next.clone());
            cur = next;
        }
        cur
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut cache = Cache::default();
        self.forward_cached(x, &mut cache)
    }

    /// Forward + backward pass for one sample: accumulates the cross-entropy
    /// gradient into `dw`/`db` (no weight update). Returns the sample's loss.
    pub fn forward_backward(&mut self, sample: &Sample, cache: &mut Cache) -> f32 {
        let logits = self.forward_cached(&sample.image, cache);
        let target = sample.label as usize;
        let probs = softmax(&logits);
        let loss = -probs[target].max(1e-9).ln();
        let mut delta = probs;
        delta[target] -= 1.0;
        self.backward(cache, &delta);
        loss
    }

    /// Scales all gradient accumulators (e.g. by 1/batch_len to average over
    /// a mini-batch).
    pub fn scale_grads(&mut self, f: f32) {
        for l in &mut self.layers {
            for g in l.dw.iter_mut() {
                *g *= f;
            }
            for g in l.db.iter_mut() {
                *g *= f;
            }
        }
    }

    /// Accumulates `dL/dz` of every layer into their gradients.
    /// `d_out` is `d(loss)/d(logits)` of the last layer.
    pub fn backward(&mut self, cache: &Cache, d_out: &[f32]) {
        let n = self.layers.len();
        let mut delta = d_out.to_vec();
        for l in (0..n).rev() {
            let layer = &mut self.layers[l];
            layer.accumulate_grads(&delta, &cache.inputs[l]);
            let mut d_in = vec![0.0; layer.in_dim];
            for i in 0..layer.in_dim {
                let mut s = 0.0;
                for o in 0..layer.out_dim {
                    s += layer.w[o * layer.in_dim + i] * delta[o];
                }
                d_in[i] = s;
            }
            if l > 0 {
                // The input of layer l is act(pre[l - 1]).
                let pre = &cache.pre[l - 1];
                for i in 0..d_in.len() {
                    d_in[i] *= activation_deriv(&pre[i], self.hidden_act);
                }
            }
            delta = d_in;
        }
    }

    pub fn predict_probs(&self, image: &[f32]) -> Vec<f32> {
        softmax(&self.forward(image))
    }

    pub fn predict(&self, image: &[f32]) -> usize {
        argmax(&self.predict_probs(image))
    }

    /// `(correct, total)` over `samples`.
    pub fn accuracy_stats(&self, samples: &[Sample]) -> (usize, usize) {
        let total = samples.len();
        let correct = samples
            .iter()
            .filter(|s| self.predict(&s.image) == s.label as usize)
            .count();
        (correct, total)
    }

    pub fn accuracy(&self, samples: &[Sample]) -> f32 {
        let (c, t) = self.accuracy_stats(samples);
        if t == 0 {
            0.0
        } else {
            c as f32 / t as f32
        }
    }
}

/// One training epoch with mini-batches: per batch, the gradients of its
/// samples are accumulated, scaled by `1/batch_len` (the average gradient),
/// and one SGD step of `lr` is applied. Returns the mean per-batch loss.
pub fn train_epoch(
    net: &mut Mlp,
    samples: &[Sample],
    lr: f32,
    batch_size: usize,
    cache: &mut Cache,
) -> f32 {
    let mut total = 0.0f32;
    let mut batches = 0usize;
    for chunk in samples.chunks(batch_size.max(1)) {
        net.zero_grads();
        let mut loss_sum = 0.0f32;
        for s in chunk {
            loss_sum += net.forward_backward(s, cache);
        }
        net.scale_grads(1.0 / chunk.len() as f32);
        net.step(lr);
        total += loss_sum;
        batches += 1;
    }
    total / batches.max(1) as f32
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|z| (z - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

fn apply_activation(z: &[f32], act: Activation) -> Vec<f32> {
    z.iter()
        .map(|&x| match act {
            Activation::Relu => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        })
        .collect()
}

fn activation_deriv(z: &f32, act: Activation) -> f32 {
    match act {
        Activation::Relu => {
            if *z > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Activation::Sigmoid => {
            let s = 1.0 / (1.0 + (-z).exp());
            s * (1.0 - s)
        }
    }
}
