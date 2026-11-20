# Handwritten digit recognizer

A small neural network (784 → 128 → 10) trained on MNIST, written from scratch
with no ML frameworks: numeric kernels in C, training in Rust, and live
inference in the browser in JavaScript.

**[Live demo](https://abdizzyy.github.io/draw-a-digit/web/index.html)**: draw a
digit and the network predicts it as you draw.

![Drawing digits in the browser demo while the network predicts them live](docs/demo.gif)

## Results
- **Test accuracy: 97.40%** (9740/10000) on the MNIST test set, with a
  784 → 128 → 10 ReLU network trained for 25 epochs (lr 0.5, batch 32, seed 42).
  The Rust and JS engines score the same.
- **Inference time per prediction:** Rust ~45 µs · JS ~75 µs (Zen/Firefox), on
  an Apple M4. Both use the same benchmark: a random input, 200 warmup runs,
  then the average of 2000 predictions (`nn bench` and `web/benchmark.html`).

## How the pieces fit together
| Part | Language | Role |
|---|---|---|
| `numeric_kernels.*` | C | matmul, activations (ReLU, sigmoid, tanh, GELU…), stable softmax |
| `nn/` | Rust | data loading, forward/backward pass, training, CLI |
| `web/engine.mjs` | JS | port of the Rust forward pass, runs the trained model in the browser |

The Rust engine does not call the C kernels; it has its own implementation.
The C file is a standalone kernel library with its own tests. Rust writes the
trained weights to `model.bin` (format documented in `nn/src/model.rs`), and
`web/model.mjs` reads that same file in the browser.

## Quick start
Requires a C compiler, Rust (cargo), Node.js and Python 3.

```sh
# 1. Download MNIST into mnist/ (ossci-datasets mirror)
mkdir -p mnist && cd mnist
for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
         t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte; do
  curl -fLO "https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz" && gunzip "$f.gz"
done
cd ..

# 2. Build, train and evaluate (a trained model.bin is already included)
cd nn
cargo build --release
./target/release/nn train --data ../mnist --epochs 25 --lr 0.5 --batch 32 --save model.bin
./target/release/nn eval  --data ../mnist     # test accuracy
./target/release/nn bench                     # time per prediction
cp model.bin ../web/model.bin
cd ..

# 3. Run the demo, then open http://localhost:8000/web/index.html
python3 -m http.server 8000 --bind 127.0.0.1
```

The demo links to a benchmark page (`web/benchmark.html`) that times
predictions in your browser; its **Back** button returns to the demo.

`nn predict < features.txt` classifies one image given as 784 floats on stdin.

## Testing
```sh
# Backprop: compares analytic gradients against numeric ones
(cd nn && ./target/release/nn gradcheck)

# JS port: accuracy on the full MNIST test set (fails below 90%)
node web/tests/engine.test.mjs

# C kernels
cc -std=c11 -O2 -I. -o test_numeric_kernels test_numeric_kernels.c numeric_kernels.c -lm
./test_numeric_kernels
```

## What I learned / what broke
- **The learning rate killed training until I added mini-batches.** With
  single-sample SGD, lr 0.5 (and also 0.2 and 0.3) made the network collapse:
  ReLU units died, the output became constant, and accuracy sat at chance
  (~9.75%) for the whole run, with no NaNs to warn me. lr 0.1 trained, but
  noisily (test accuracy ~75%). Switching to batches of 32 with averaged
  gradients let lr 0.5 reach 97.4%.
- **A suspicious gradient was actually correct.** In a backprop trace,
  `dw0[0]` was exactly 0 while `db0[0]` was not. It looked like a bug, but the
  top-left pixel of every MNIST image is 0 (digits have margins), so the
  gradient for that weight, `delta * x[0]`, is always 0. `gradcheck` confirmed
  analytic and numeric gradients agree to ~1e-5.
- **Drawings have to look like MNIST.** At first the browser passed the raw
  canvas, downsampled to 28×28, straight into the network, at whatever size and
  position the strokes happened to be. MNIST digits are scaled to fit a 20×20
  box and centered by center of mass, so `centerDigit` in `web/main.js` now does
  the same before every prediction.
- **Two languages, one model file.** Writing the weight format as a spec
  (magic bytes, version, little-endian dims, row-major weights) made the JS port
  straightforward. The parity test shows identical 97.40% accuracy even though
  JS sums in f64 and Rust in f32.

## Limitations
Only digits 0–9, trained on MNIST, so very unusual handwriting can be wrong or
overconfident. The network always picks some digit, even for input that isn't
one: an empty canvas comes out as "5" with 95% confidence.
