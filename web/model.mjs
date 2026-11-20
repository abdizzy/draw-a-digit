// Parses the binary model written by `nn train` (nn/src/model.rs).
// Layout: magic "DGNN", version u32, input_dim u32, output_dim u32,
// activation u8 (0 = relu, 1 = sigmoid), layer count u32, then per layer:
// in_dim u32, out_dim u32, weights f32[in_dim*out_dim] LE, bias f32[out_dim] LE.
export function parseModel(buf) {
  if (!(buf instanceof ArrayBuffer)) {
    // Node Buffer / typed array view: copy into a fresh aligned ArrayBuffer.
    const copy = new Uint8Array(buf.byteLength);
    copy.set(buf);
    buf = copy.buffer;
  }
  const dv = new DataView(buf);
  let off = 0;

  const MAGIC = [68, 71, 78, 78]; // "DGNN"
  for (let i = 0; i < 4; i++) {
    if (dv.getUint8(i) !== MAGIC[i]) throw new Error('bad model magic');
  }
  off = 4;

  const u32 = () => {
    const v = dv.getUint32(off, true);
    off += 4;
    return v;
  };

  const version = u32();
  if (version !== 1) throw new Error('unsupported model version ' + version);
  const inputDim = u32();
  const outputDim = u32();
  const actId = dv.getUint8(off);
  off += 1;
  if (actId !== 0 && actId !== 1) throw new Error('bad activation id ' + actId);
  const hiddenAct = actId === 0 ? 'relu' : 'sigmoid';
  const layerCount = u32();

  const layers = [];
  for (let i = 0; i < layerCount; i++) {
    const inDim = u32();
    const outDim = u32();
    const w = new Float32Array(buf.slice(off, off + 4 * inDim * outDim));
    off += 4 * inDim * outDim;
    const b = new Float32Array(buf.slice(off, off + 4 * outDim));
    off += 4 * outDim;
    layers.push({ inDim, outDim, w, b });
  }
  if (off !== buf.byteLength) throw new Error('trailing bytes in model');
  return { inputDim, outputDim, hiddenAct, layers };
}
