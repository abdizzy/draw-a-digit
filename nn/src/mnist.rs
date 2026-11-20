//! Loads the standard MNIST IDX files into `Sample`s (pixels normalized to 0..1).

use std::fs;
use std::io::{self, Read};
use std::path::Path;

use crate::engine::Sample;

pub struct Mnist {
    pub train: Vec<Sample>,
    pub test: Vec<Sample>,
}

fn be_u32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

fn bad(path: &Path, msg: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("{}: {msg}", path.display()),
    )
}

fn read_idx_images(path: &Path) -> io::Result<Vec<u8>> {
    let mut data = Vec::new();
    fs::File::open(path)?.read_to_end(&mut data)?;
    if data.len() < 16 {
        return Err(bad(path, "too short"));
    }
    if [data[0], data[1], data[2], data[3]] != [0, 0, 8, 3] {
        return Err(bad(path, "bad magic (expected 0x00000803)"));
    }
    let rows = be_u32(&data[8..12]);
    let cols = be_u32(&data[12..16]);
    if rows != 28 || cols != 28 {
        return Err(bad(path, "unexpected image size"));
    }
    let n = be_u32(&data[4..8]) as usize * 784;
    if data.len() < 16 + n {
        return Err(bad(path, "truncated pixel data"));
    }
    Ok(data[16..16 + n].to_vec())
}

fn read_idx_labels(path: &Path) -> io::Result<Vec<u8>> {
    let mut data = Vec::new();
    fs::File::open(path)?.read_to_end(&mut data)?;
    if data.len() < 8 {
        return Err(bad(path, "too short"));
    }
    if [data[0], data[1], data[2], data[3]] != [0, 0, 8, 1] {
        return Err(bad(path, "bad magic (expected 0x00000801)"));
    }
    let count = be_u32(&data[4..8]) as usize;
    if data.len() < 8 + count {
        return Err(bad(path, "truncated label data"));
    }
    Ok(data[8..8 + count].to_vec())
}

fn to_samples(images: &[u8], labels: &[u8]) -> io::Result<Vec<Sample>> {
    if images.len() != labels.len() * 784 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "image/label count mismatch",
        ));
    }
    let mut out = Vec::with_capacity(labels.len());
    for (img, &label) in images.chunks_exact(784).zip(labels) {
        if label > 9 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("label {label} out of range"),
            ));
        }
        out.push(Sample {
            image: img.iter().map(|&p| p as f32 / 255.0).collect(),
            label,
        });
    }
    Ok(out)
}

pub fn load_mnist(dir: &Path) -> io::Result<Mnist> {
    let train_img = read_idx_images(&dir.join("train-images-idx3-ubyte"))?;
    let train_lab = read_idx_labels(&dir.join("train-labels-idx1-ubyte"))?;
    let test_img = read_idx_images(&dir.join("t10k-images-idx3-ubyte"))?;
    let test_lab = read_idx_labels(&dir.join("t10k-labels-idx1-ubyte"))?;
    Ok(Mnist {
        train: to_samples(&train_img, &train_lab)?,
        test: to_samples(&test_img, &test_lab)?,
    })
}
