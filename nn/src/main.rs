//! `nn` — command-line interface for the handwritten-digit network.
//!
//! Usage:
//!   nn train   [--data <dir>] [--epochs N] [--lr F] [--batch N] [--hidden 128,64] [--seed N] [--save f] [--limit N]
//!   nn eval    [--data <dir>] [--model f] [--limit N]
//!   nn predict [--model f]              # reads 784 floats on stdin
//!
//! Defaults: data=mnist, epochs=20, lr=0.5, batch=32, hidden=128, seed=42, save=model.bin

mod engine;
mod mnist;
mod model;
mod rng;

use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

use engine::{train_epoch, Activation, Cache, Mlp, INPUT_DIM, OUTPUT_DIM};

const USAGE: &str = "Usage:
  nn train   [--data <dir>] [--epochs N] [--lr F] [--batch N] [--hidden 128,64] [--seed N] [--save f] [--limit N]
  nn eval    [--data <dir>] [--model f] [--limit N]
  nn predict [--model f]              # reads 784 whitespace-separated floats on stdin
Defaults: data=mnist, epochs=20, lr=0.5, batch=32, hidden=128, seed=42, save=model.bin";

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let code = match argv.split_first() {
        Some((cmd, rest)) => match cmd.as_str() {
            "train" => cmd_train(rest),
            "eval" => cmd_eval(rest),
            "predict" => cmd_predict(rest),
            "gradcheck" => cmd_gradcheck(rest),
            "help" | "--help" | "-h" => {
                println!("{USAGE}");
                0
            }
            other => {
                eprintln!("error: unknown command '{other}'\n\n{USAGE}");
                2
            }
        },
        None => {
            eprintln!("{USAGE}");
            2
        }
    };
    std::process::exit(code);
}

/// Reads one `--flag value` pair, or an error message naming the flag.
fn next_value(args: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    let v = match args.get(*i) {
        Some(v) => v.clone(),
        None => return Err(format!("missing value for {flag}")),
    };
    *i += 1;
    Ok(v)
}

/// Numerical-gradient check for the backprop implementation (debug tool).
fn cmd_gradcheck(_args: &[String]) -> i32 {
    let mut rng = rng::Rng::new(42);
    let mut net = Mlp::new(&mut rng, INPUT_DIM, &[8], 3, Activation::Relu);
    let mut cache = Cache::default();
    let image: Vec<f32> = (0..INPUT_DIM)
        .map(|i| ((i as f32) * 0.37).sin().abs() * 0.5)
        .collect();
    let sample = engine::Sample { image, label: 2 };

    net.zero_grads();
    let logits = net.forward_cached(&sample.image, &mut cache);
    let probs = engine::softmax(&logits);
    let mut delta = probs;
    delta[2] -= 1.0;
    net.backward(&cache, &delta);

    // Hidden units with a positive pre-activation (relu' = 1): the alive set.
    let pre0: Vec<f32> = cache.pre[0].clone();
    let act0: Vec<f32> = cache.act[0].clone();
    let alive: Vec<usize> = pre0
        .iter()
        .enumerate()
        .filter(|(_, &z)| z > 0.0)
        .map(|(i, _)| i)
        .collect();
    println!(
        "hidden units alive (z_0 > 0): {}/{}, act0 sum: {:.4}",
        alive.len(),
        pre0.len(),
        act0.iter().sum::<f32>()
    );

    let loss_of = |net: &Mlp, s: &engine::Sample| -> f32 {
        let probs = engine::softmax(&net.forward(&s.image));
        -probs[s.label as usize].max(1e-9).ln()
    };
    let numeric =
        |net: &mut Mlp, layer: usize, idx: usize, eps: f32, s: &engine::Sample| -> f32 {
            let orig = net.layers[layer].w[idx];
            net.layers[layer].w[idx] = orig + eps;
            let lp = loss_of(net, s);
            net.layers[layer].w[idx] = orig - eps;
            let lm = loss_of(net, s);
            net.layers[layer].w[idx] = orig;
            (lp - lm) / (2.0 * eps)
        };

    let mut ok = true;
    // Layer-0 weights of the alive units (a few columns each).
    let mut checked = 0;
    for &u in alive.iter().take(4) {
        for col in [3usize, 5, 7, 11] {
            if checked >= 6 {
                break;
            }
            let idx = u * INPUT_DIM + col;
            let a = net.layers[0].dw[idx];
            let n = numeric(&mut net, 0, idx, 1e-3, &sample);
            println!("layer0.dw[{idx}]: analytic {a:.6} numeric {n:.6}");
            if (a - n).abs() > 1e-3 {
                ok = false;
            }
            checked += 1;
        }
    }
    for idx in [0usize, 3, 6, 9, 12, 15, 18, 21] {
        let a = net.layers[1].dw[idx];
        let n = numeric(&mut net, 1, idx, 1e-3, &sample);
        println!("layer1.dw[{idx}]: analytic {a:.6} numeric {n:.6}");
        if (a - n).abs() > 1e-3 {
            ok = false;
        }
    }
    println!("{}", if ok { "gradcheck passed" } else { "gradcheck FAILED" });
    if ok {
        0
    } else {
        1
    }
}

struct TrainOpts {
    data: PathBuf,
    epochs: usize,
    lr: f32,
    batch: usize,
    hidden: Vec<usize>,
    seed: u64,
    save: PathBuf,
    limit: Option<usize>,
}

impl TrainOpts {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut o = TrainOpts {
            data: PathBuf::from("mnist"),
            epochs: 20,
            lr: 0.5,
            batch: 32,
            hidden: vec![128],
            seed: 42,
            save: PathBuf::from("model.bin"),
            limit: None,
        };
        let mut i = 0;
        while i < args.len() {
            let flag = args[i].clone();
            i += 1;
            let value = next_value(args, &mut i, &flag)?;
            match flag.as_str() {
                "--data" => o.data = PathBuf::from(value),
                "--epochs" => o.epochs = value.parse().map_err(|_| format!("bad --epochs '{value}'"))?,
                "--lr" => o.lr = value.parse().map_err(|_| format!("bad --lr '{value}'"))?,
                "--batch" => o.batch = value.parse().map_err(|_| format!("bad --batch '{value}'"))?,
                "--hidden" => o.hidden = parse_dims(&value)?,
                "--seed" => o.seed = value.parse().map_err(|_| format!("bad --seed '{value}'"))?,
                "--save" => o.save = PathBuf::from(value),
                "--limit" => {
                    o.limit = Some(value.parse().map_err(|_| format!("bad --limit '{value}'"))?)
                }
                other => return Err(format!("unknown option for train: {other}")),
            }
        }
        Ok(o)
    }
}

fn parse_dims(s: &str) -> Result<Vec<usize>, String> {
    let dims: Result<Vec<usize>, _> = s
        .split(',')
        .map(|p| p.parse::<usize>())
        .collect();
    let dims = dims.map_err(|e| format!("bad --hidden '{s}': {e}"))?;
    if dims.is_empty() || dims.iter().any(|&d| d == 0) {
        return Err(format!("--hidden must contain positive dimensions: '{s}'"));
    }
    Ok(dims)
}

fn cmd_train(args: &[String]) -> i32 {
    let o = match TrainOpts::parse(args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            return 2;
        }
    };
    let data = match mnist::load_mnist(&o.data) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    };
    let limit = o.limit.unwrap_or(usize::MAX);
    let train = &data.train[..data.train.len().min(limit)];
    let test = &data.test[..data.test.len().min(limit)];

    let mut rng = rng::Rng::new(o.seed);
    let mut net = Mlp::new(
        &mut rng,
        INPUT_DIM,
        &o.hidden,
        OUTPUT_DIM,
        Activation::Relu,
    );
    let mut cache = Cache::default();

    println!("architecture: {}", net.describe());
    println!("train samples: {}, test samples: {}", train.len(), test.len());
    println!("epochs: {}, lr: {}, batch: {}, seed: {}", o.epochs, o.lr, o.batch, o.seed);

    for epoch in 1..=o.epochs {
        let t0 = Instant::now();
        let avg_loss = train_epoch(&mut net, train, o.lr, o.batch, &mut cache);
        let elapsed = t0.elapsed().as_secs_f32();
        let train_acc = net.accuracy(train);
        let test_part = if epoch == o.epochs || epoch % 5 == 0 {
            format!(" test_acc {:.4}", net.accuracy(test))
        } else {
            String::new()
        };
        println!(
            "epoch {epoch}: loss {avg_loss:.4} train_acc {train_acc:.4}{test_part} [{elapsed:.1}s]"
        );
    }

    if let Err(e) = model::save_model(&o.save, &net) {
        eprintln!("error: {e}");
        return 1;
    }
    println!("saved model to {}", o.save.display());
    0
}

struct EvalOpts {
    data: PathBuf,
    model: PathBuf,
    limit: Option<usize>,
}

impl EvalOpts {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut o = EvalOpts {
            data: PathBuf::from("mnist"),
            model: PathBuf::from("model.bin"),
            limit: None,
        };
        let mut i = 0;
        while i < args.len() {
            let flag = args[i].clone();
            i += 1;
            let value = next_value(args, &mut i, &flag)?;
            match flag.as_str() {
                "--data" => o.data = PathBuf::from(value),
                "--model" => o.model = PathBuf::from(value),
                "--limit" => o.limit = Some(value.parse().map_err(|_| format!("bad --limit '{value}'"))?),
                other => return Err(format!("unknown option for eval: {other}")),
            }
        }
        Ok(o)
    }
}

fn cmd_eval(args: &[String]) -> i32 {
    let o = match EvalOpts::parse(args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            return 2;
        }
    };
    let data = match mnist::load_mnist(&o.data) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    };
    let net = match model::load_model(&o.model) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    };
    let limit = o.limit.unwrap_or(usize::MAX);
    let test = &data.test[..data.test.len().min(limit)];
    let (correct, total) = net.accuracy_stats(test);
    println!("model: {}", o.model.display());
    println!(
        "test accuracy: {correct}/{total} = {:.2}%",
        100.0 * correct as f32 / total.max(1) as f32
    );
    0
}

struct PredictOpts {
    model: PathBuf,
}

impl PredictOpts {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut o = PredictOpts {
            model: PathBuf::from("model.bin"),
        };
        let mut i = 0;
        while i < args.len() {
            let flag = args[i].clone();
            i += 1;
            let value = next_value(args, &mut i, &flag)?;
            match flag.as_str() {
                "--model" => o.model = PathBuf::from(value),
                other => return Err(format!("unknown option for predict: {other}")),
            }
        }
        Ok(o)
    }
}

fn cmd_predict(args: &[String]) -> i32 {
    let o = match PredictOpts::parse(args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            return 2;
        }
    };
    let net = match model::load_model(&o.model) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    };
    let mut input = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut input) {
        eprintln!("error: {e}");
        return 1;
    }
    let mut vals: Vec<f32> = Vec::new();
    for tok in input.split_whitespace() {
        match tok.parse::<f32>() {
            Ok(v) => vals.push(v),
            Err(_) => {
                eprintln!("error: cannot parse '{tok}' as a float");
                return 1;
            }
        }
    }
    if vals.len() != INPUT_DIM {
        eprintln!("error: expected {INPUT_DIM} input values, got {}", vals.len());
        return 1;
    }
    let probs = net.predict_probs(&vals);
    let best = engine::argmax(&probs);
    let line: Vec<String> = probs
        .iter()
        .enumerate()
        .map(|(d, p)| format!("{d}:{p:.4}"))
        .collect();
    println!("{}", line.join(" "));
    println!("prediction: {best} (confidence {:.4})", probs[best]);
    0
}
