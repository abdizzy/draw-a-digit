# Handwritten digit recognizer

A tiny handwritten digit recognizer built from scratch. a small MLP
(784 -> 128 -> 10) trained on MNIST, in C, Rust and JavaScript.

## C kernels — `numeric_kernels.h/.c`

Matrix multiply (`nk_matmul`), activation functions (`nk_relu`,
`nk_leaky_relu`, `nk_sigmoid`, `nk_tanh`, `nk_gelu`, `nk_gelu_tanh`) and a
numerically stable softmax (`nk_softmax`, `nk_softmax_inplace`,
`nk_softmax_rows`).

## Rust engine — `nn/`

```sh
cd nn
cargo build --release
./target/release/nn train --epochs 25 --lr 0.5 --batch 32 --save model.bin
./target/release/nn eval                    # test accuracy on MNIST
./target/release/nn predict < features.txt  # 784 floats on stdin
./target/release/nn gradcheck               # numeric-gradient check of backprop
```

MNIST IDX files live in `mnist/` (downloaded from the standard
`ossci-datasets` mirror); pass `--data <dir>` for another copy. Running from
`nn/` works as is: the `nn/mnist` symlink points at the root copy.

## Web frontend `web/`

`web/engine.mjs` is a port of the Rust engine; `web/model.mjs` parses the
model that the Rust trainer writes.

```sh
cp nn/model.bin web/model.bin      # (already done if you trained to web/)
python3 -m http.server -d web      # serve the folder; open http://localhost:8000/index.html
node web/tests/engine.test.mjs     # JS engine accuracy on the MNIST test set
```

- `index.html` — draw a digit on the canvas; the network predicts live.
- `benchmark.html` — average time per prediction through the same engine.
