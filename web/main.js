// Live prediction: draw a digit on the canvas — the network (web/engine.mjs,
// a port of the Rust engine) predicts as you draw. The input convention
// matches the Rust engine's MNIST data: bright strokes on a black background,
// values in [0, 1].

import { Mlp, argmax } from './engine.mjs';
import { parseModel } from './model.mjs';

const GRID = 28;
const EMPTY_TEXT = 'Draw a digit on the canvas — predictions appear as you draw.';
const NOT_SURE_TEXT = 'Not sure — draw the digit more clearly';
const NOT_SURE_CONFIDENCE = 0.6;
const INK_THRESHOLD = 0.01;

const canvas = document.getElementById('draw-canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const previewCanvas = document.getElementById('preview-canvas');
const previewCtx = previewCanvas.getContext('2d');
const barsEl = document.getElementById('bars');
const predictionEl = document.getElementById('prediction');
const clearBtn = document.getElementById('clear');

let net = null;
let drawing = false;
let lastPt = null;

ctx.fillStyle = '#fff';
ctx.strokeStyle = '#fff';

function brushAt(x, y) {
  // Stroke from the previous point (round caps) plus a filled round brush
  // at the current point — so fast movement leaves no gaps.
  ctx.beginPath();
  if (lastPt) {
    ctx.moveTo(lastPt.x, lastPt.y);
    ctx.lineTo(x, y);
    ctx.lineCap = 'round';
    ctx.lineWidth = 24;
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.arc(x, y, 12, 0, Math.PI * 2);
  ctx.fill();
  lastPt = { x, y };
}

function pointerPos(e) {
  const rect = canvas.getBoundingClientRect();
  const pt = e.touches ? e.touches[0] : e;
  // Map client coordinates onto canvas pixels: the CSS size of the canvas
  // may differ from its pixel size (resized screens, phones, ...).
  const sx = rect.width > 0 ? canvas.width / rect.width : 1;
  const sy = rect.height > 0 ? canvas.height / rect.height : 1;
  return { x: (pt.clientX - rect.left) * sx, y: (pt.clientY - rect.top) * sy };
}

function onPointer(e) {
  const { x, y } = pointerPos(e);
  brushAt(x, y);
  updatePrediction();
}

canvas.addEventListener('mousedown', (e) => {
  drawing = true;
  lastPt = null;
  onPointer(e);
});
canvas.addEventListener('mousemove', (e) => {
  if (drawing) onPointer(e);
});
window.addEventListener('mouseup', () => {
  drawing = false;
});

canvas.addEventListener('touchstart', (e) => {
  e.preventDefault();
  drawing = true;
  lastPt = null;
  onPointer(e);
});
canvas.addEventListener('touchmove', (e) => {
  e.preventDefault();
  if (drawing) onPointer(e);
});
canvas.addEventListener('touchend', () => {
  drawing = false;
});

// Averages the 28x28 pixel blocks of the 280x280 canvas into the 28x28
// network input (pixel/255), matching MNIST.
function extractDigit() {
  const img = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const step = canvas.width / GRID;
  const v = new Float32Array(GRID * GRID);
  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      let sum = 0;
      for (let y = 0; y < step; y++) {
        for (let x = 0; x < step; x++) {
          const px = Math.min(canvas.width - 1, Math.floor(gx * step + x));
          const py = Math.min(canvas.height - 1, Math.floor(gy * step + y));
          sum += img[(py * canvas.width + px) * 4];
        }
      }
      v[gy * GRID + gx] = sum / (step * step * 255);
    }
  }
  return v;
}

// The extraction is "empty" when the ink is below the threshold — nothing
// was drawn yet.
function digitEmpty(v) {
  let busy = 0;
  for (const x of v) busy += x;
  return busy < INK_THRESHOLD;
}

// Bilinear sample of the source grid at continuous coordinates (x, y);
// returns 0 outside the grid.
function sampleBilinear(v, x, y) {
  const W = GRID;
  const H = GRID;
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const x1 = x0 + 1, y1 = y0 + 1;
  const fx = x - x0, fy = y - y0;
  const at = (px, py) =>
    px >= 0 && px < W && py >= 0 && py < H ? v[py * W + px] : 0;
  const top = at(x0, y0) * (1 - fx) + at(x1, y0) * fx;
  const bottom = at(x0, y1) * (1 - fx) + at(x1, y1) * fx;
  return top * (1 - fy) + bottom * fy;
}

// Centers and scales the extracted digit the way MNIST looks: the bounding
// box of the ink is scaled so its longest side is 20 (in the 28x28 grid)
// and pasted so its center of mass sits at (14, 14). Inverse mapping with
// bilinear sampling, so strokes stay solid when the digit is small.
function centerDigit(v) {
  const W = GRID;
  const H = GRID;
  let minX = W, maxX = -1, minY = H, maxY = -1;
  let sumX = 0, sumY = 0, n = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const val = v[y * W + x];
      if (val > INK_THRESHOLD) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        sumX += x;
        sumY += y;
        n++;
      }
    }
  }
  if (n === 0) return v;
  const cmX = sumX / n;
  const cmY = sumY / n;
  const scale = 20 / Math.max(maxX - minX + 1, maxY - minY + 1);
  const out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      // Inverse of the forward map dest = (src - cm) * scale + 14.
      const srcX = cmX + (x - 14) / scale;
      const srcY = cmY + (y - 14) / scale;
      out[y * W + x] = sampleBilinear(v, srcX, srcY);
    }
  }
  return out;
}

// Draws the 28x28 preview of what the model actually receives.
function drawPreview(v) {
  previewCtx.clearRect(0, 0, GRID, GRID);
  previewCtx.fillStyle = '#000';
  previewCtx.fillRect(0, 0, GRID, GRID);
  for (let y = 0; y < GRID; y++) {
    for (let x = 0; x < GRID; x++) {
      const val = v[y * GRID + x];
      if (val > 0) {
        previewCtx.fillStyle = `rgba(255, 255, 255, ${val})`;
        previewCtx.fillRect(x, y, 1, 1);
      }
    }
  }
}

function updatePrediction() {
  const v = extractDigit();
  if (!net || digitEmpty(v)) {
    predictionEl.textContent = EMPTY_TEXT;
    barsEl.innerHTML = '';
    drawPreview(v);
    return;
  }
  const centered = centerDigit(v);
  drawPreview(centered);
  const probs = net.predictProbs(centered);
  const best = argmax(probs);
  const conf = probs[best];
  predictionEl.textContent =
    conf < NOT_SURE_CONFIDENCE
      ? NOT_SURE_TEXT
      : `Prediction: ${best} (confidence ${(conf * 100).toFixed(1)}%)`;
  barsEl.innerHTML = '';
  for (let d = 0; d < 10; d++) {
    const row = document.createElement('div');
    row.className = 'bar';
    const label = document.createElement('span');
    label.className = 'bar-label';
    label.textContent = String(d);
    const track = document.createElement('div');
    track.className = 'bar-track';
    const fill = document.createElement('div');
    fill.className = 'bar-fill';
    fill.style.width = `${(probs[d] * 100).toFixed(1)}%`;
    const pct = document.createElement('span');
    pct.className = 'bar-pct';
    pct.textContent = `${(probs[d] * 100).toFixed(1)}%`;
    track.appendChild(fill);
    row.append(label, track, pct);
    barsEl.appendChild(row);
  }
}

// Fills the canvas black so the red channel carries real grayscale:
// anti-aliased stroke edges then read as intermediate values instead of
// full brightness.
function resetCanvas() {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#fff';
  lastPt = null;
}

async function init() {
  const res = await fetch('model.bin');
  if (!res.ok) {
    predictionEl.textContent =
      'model.bin not found — train with `nn train --save web/model.bin` first';
    return;
  }
  net = new Mlp(parseModel(await res.arrayBuffer()));
  resetCanvas();
  updatePrediction();
}

clearBtn.addEventListener('click', () => {
  resetCanvas();
  updatePrediction();
});

init();
