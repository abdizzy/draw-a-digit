//! Save/load for the network weights.
//!
//! Binary format (little-endian, shared with `web/model.mjs`):
//!
//! ```text
//! magic "DGNN", version u32, input_dim u32, output_dim u32,
//! activation u8 (0 = relu, 1 = sigmoid), layer count u32,
//! then per layer: in_dim u32, out_dim u32, weights f32[in*out], bias f32[out]
//! ```

use std::fs;
use std::io::{self, Read};
use std::path::Path;

use crate::engine::{Activation, Linear, Mlp};

const MAGIC: &[u8; 4] = b"DGNN";
const VERSION: u32 = 1;

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn push_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn activation_id(a: Activation) -> u8 {
    match a {
        Activation::Relu => 0,
        Activation::Sigmoid => 1,
    }
}

fn activation_from_id(id: u8) -> io::Result<Activation> {
    match id {
        0 => Ok(Activation::Relu),
        1 => Ok(Activation::Sigmoid),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad activation id {other}"),
        )),
    }
}

fn read_u32(r: &mut io::Cursor<Vec<u8>>) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_f32s(r: &mut impl Read, n: usize) -> io::Result<Vec<f32>> {
    let mut bytes = vec![0u8; n * 4];
    r.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn save_model(path: &Path, net: &Mlp) -> io::Result<()> {
    let mut out = Vec::with_capacity(64);
    out.extend_from_slice(MAGIC);
    push_u32(&mut out, VERSION);
    push_u32(&mut out, net.input_dim as u32);
    push_u32(&mut out, net.output_dim as u32);
    out.push(activation_id(net.hidden_act));
    push_u32(&mut out, net.layers.len() as u32);
    for l in &net.layers {
        push_u32(&mut out, l.in_dim as u32);
        push_u32(&mut out, l.out_dim as u32);
        for &w in &l.w {
            push_f32(&mut out, w);
        }
        for &b in &l.b {
            push_f32(&mut out, b);
        }
    }
    fs::write(path, out)
}

pub fn load_model(path: &Path) -> io::Result<Mlp> {
    let mut file = fs::File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    let mut r = io::Cursor::new(buf);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bad model magic",
        ));
    }
    let version = read_u32(&mut r)?;
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported model version {version}"),
        ));
    }
    let input_dim = read_u32(&mut r)? as usize;
    let output_dim = read_u32(&mut r)? as usize;
    let mut act_byte = [0u8; 1];
    r.read_exact(&mut act_byte)?;
    let act = activation_from_id(act_byte[0])?;
    let n_layers = read_u32(&mut r)? as usize;

    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let in_dim = read_u32(&mut r)? as usize;
        let out_dim = read_u32(&mut r)? as usize;
        let w = read_f32s(&mut r, in_dim * out_dim)?;
        let b = read_f32s(&mut r, out_dim)?;
        let dw = vec![0.0; w.len()];
        let db = vec![0.0; out_dim];
        layers.push(Linear {
            in_dim,
            out_dim,
            w,
            b,
            dw,
            db,
        });
    }
    Ok(Mlp {
        input_dim,
        output_dim,
        hidden_act: act,
        layers,
    })
}
