// Depth Anything V2 Small による単眼深度推定。
// 推論は描画ループと独立に回し、最新の結果（相対逆深度: 大きいほど近い）だけを保持する。

const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5';
const MODEL_ID = 'onnx-community/depth-anything-v2-small';
// ViT のパッチが 14px なので入力の縦横は 14 の倍数。縦 252px でおよそ 600 パッチ
const INPUT_H = 252;
const PATCH = 14;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
// 正規化範囲をフレーム間で平滑化し、明るさの揺れで等高線が跳ばないようにする
const RANGE_BLEND = 0.2;

function halfToFloat(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

export class DepthEstimator {
  constructor() {
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.running = false;
    this.loading = null;
    this.result = null;   // { data: Float32Array, w, h, version }
    this.lo = 0;
    this.hi = 1;
    this.fps = 0;
    this.device = '';
  }

  load() {
    this.loading ??= (async () => {
      const T = await import(TRANSFORMERS_URL);
      this.Tensor = T.Tensor;
      this.device = navigator.gpu ? 'webgpu' : 'wasm';
      this.model = await T.AutoModel.from_pretrained(MODEL_ID, {
        device: this.device,
        dtype: this.device === 'webgpu' ? 'fp16' : 'q8',
      });
    })();
    return this.loading;
  }

  async run(video) {
    this.running = true;
    await this.load();
    let version = 0;
    while (this.running) {
      if (!video.videoWidth) { await new Promise((r) => setTimeout(r, 50)); continue; }
      const t0 = performance.now();
      const out = await this.infer(video);
      if (!this.running) break;
      this.updateRange(out.data);
      this.result = { ...out, version: ++version };
      this.fps = 1000 / (performance.now() - t0);
    }
  }

  stop() {
    this.running = false;
    this.result = null;
  }

  async infer(video) {
    const h = INPUT_H;
    const w = Math.max(PATCH, Math.round((h * video.videoWidth) / video.videoHeight / PATCH) * PATCH);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.ctx.drawImage(video, 0, 0, w, h);
    const px = this.ctx.getImageData(0, 0, w, h).data;
    const n = w * h;
    const input = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < 3; c++) input[c * n + i] = (px[i * 4 + c] / 255 - MEAN[c]) / STD[c];
    }
    const { predicted_depth: d } = await this.model({ pixel_values: new this.Tensor('float32', input, [1, 3, h, w]) });
    const [oh, ow] = d.dims.slice(-2);
    // fp16 出力は Float16Array の無い環境では生のビット列（Uint16Array）で返る
    const data = d.data instanceof Uint16Array ? Float32Array.from(d.data, halfToFloat) : Float32Array.from(d.data);
    return { data, w: ow, h: oh };
  }

  // 外れ値に引っ張られないよう、2% と 98% の分位点を範囲にする
  updateRange(data) {
    const step = Math.max(1, Math.floor(data.length / 4096));
    const sample = [];
    for (let i = 0; i < data.length; i += step) sample.push(data[i]);
    sample.sort((a, b) => a - b);
    const lo = sample[Math.floor(sample.length * 0.02)];
    const hi = sample[Math.floor(sample.length * 0.98)];
    if (this.result === null) { this.lo = lo; this.hi = hi; return; }
    this.lo += (lo - this.lo) * RANGE_BLEND;
    this.hi += (hi - this.hi) * RANGE_BLEND;
  }
}
