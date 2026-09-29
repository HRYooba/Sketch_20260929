// Webカメラ映像の輪郭を線分列に変換する。
// 縮小 → ぼかし → Sobel → 非極大抑制 → ヒステリシスで辿って鎖にする → Douglas-Peucker で折れ線化。

const GRID_H = 120;
const SIMPLIFY_EPS = 1.2;
const MIN_SEG_PX = 5;
// 勾配の上位この割合を「確実なエッジ」とみなす
const STRONG_RATIO = 0.06;
const MIN_STRONG = 0.08;

export class EdgeCamera {
  constructor() {
    this.video = document.createElement('video');
    this.video.playsInline = true;
    this.video.muted = true;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.stream = null;
  }

  get on() { return this.stream !== null; }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play();
  }

  stop() {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
  }

  // 画面座標系（高さ 1、中心原点、y 上向き）の線分を長い順に最大 maxSegs 本返す
  detect(aspect, maxSegs) {
    const v = this.video;
    if (!this.on || !v.videoWidth) return null;
    const GH = GRID_H, GW = Math.round(GRID_H * aspect);
    if (this.canvas.width !== GW || this.canvas.height !== GH) {
      this.canvas.width = GW;
      this.canvas.height = GH;
    }
    // 画面を覆うように切り出し、鏡像にする
    const s = Math.max(GW / v.videoWidth, GH / v.videoHeight);
    const dw = v.videoWidth * s, dh = v.videoHeight * s;
    const ctx = this.ctx;
    ctx.setTransform(-1, 0, 0, 1, GW, 0);
    ctx.drawImage(v, (GW - dw) / 2, (GH - dh) / 2, dw, dh);
    const px = ctx.getImageData(0, 0, GW, GH).data;

    const N = GW * GH;
    let g = new Float32Array(N);
    for (let i = 0; i < N; i++) g[i] = (px[i * 4] * 0.299 + px[i * 4 + 1] * 0.587 + px[i * 4 + 2] * 0.114) / 255;
    g = blur(blur(g, GW, GH), GW, GH);

    const mag = new Float32Array(N);
    const gx = new Float32Array(N), gy = new Float32Array(N);
    for (let y = 1; y < GH - 1; y++) {
      for (let x = 1; x < GW - 1; x++) {
        const i = y * GW + x;
        const a = g[i - GW - 1], b = g[i - GW], c = g[i - GW + 1];
        const d = g[i - 1], f = g[i + 1];
        const h = g[i + GW - 1], k = g[i + GW], l = g[i + GW + 1];
        gx[i] = c + 2 * f + l - a - 2 * d - h;
        gy[i] = h + 2 * k + l - a - 2 * b - c;
        mag[i] = Math.hypot(gx[i], gy[i]);
      }
    }

    // 勾配方向に沿って極大の画素だけ残す
    const thin = new Float32Array(N);
    for (let y = 1; y < GH - 1; y++) {
      for (let x = 1; x < GW - 1; x++) {
        const i = y * GW + x, m = mag[i];
        if (m === 0) continue;
        const ang = Math.atan2(gy[i], gx[i]);
        const ox = Math.round(Math.cos(ang)), oy = Math.round(Math.sin(ang));
        if (m >= mag[i + oy * GW + ox] && m >= mag[i - oy * GW - ox]) thin[i] = m;
      }
    }

    const hi = Math.max(percentile(thin, 1 - STRONG_RATIO), MIN_STRONG);
    const lo = hi * 0.45;

    const visited = new Uint8Array(N);
    const chains = [];
    for (let i = 0; i < N; i++) {
      if (thin[i] < hi || visited[i]) continue;
      visited[i] = 1;
      const fwd = walk(i, thin, visited, lo, GW, GH);
      const bwd = walk(i, thin, visited, lo, GW, GH);
      const chain = bwd.reverse().concat([i], fwd).map((j) => [j % GW, (j / GW) | 0]);
      if (chain.length >= MIN_SEG_PX) chains.push(chain);
    }

    const segs = [];
    for (const chain of chains) {
      const pts = simplify(chain, SIMPLIFY_EPS);
      for (let k = 0; k + 1 < pts.length; k++) {
        const [ax, ay] = pts[k], [bx, by] = pts[k + 1];
        const len = Math.hypot(bx - ax, by - ay);
        if (len >= MIN_SEG_PX) segs.push({ ax, ay, bx, by, len });
      }
    }
    segs.sort((p, q) => q.len - p.len);
    const toX = (x) => ((x + 0.5) / GW - 0.5) * aspect;
    const toY = (y) => 0.5 - (y + 0.5) / GH;
    return segs.slice(0, maxSegs).map((s) => ({ ax: toX(s.ax), ay: toY(s.ay), bx: toX(s.bx), by: toY(s.by) }));
  }
}

function blur(src, W, H) {
  const tmp = new Float32Array(src.length), out = new Float32Array(src.length);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      tmp[i] = (src[i - (x > 0)] + 2 * src[i] + src[i + (x < W - 1)]) / 4;
    }
  }
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      out[i] = (tmp[i - (y > 0) * W] + 2 * tmp[i] + tmp[i + (y < H - 1) * W]) / 4;
    }
  }
  return out;
}

function percentile(arr, q) {
  let max = 0;
  for (const v of arr) if (v > max) max = v;
  if (max === 0) return Infinity;
  const BINS = 256, hist = new Uint32Array(BINS);
  let count = 0;
  for (const v of arr) if (v > 0) { hist[Math.min(BINS - 1, ((v / max) * BINS) | 0)]++; count++; }
  let acc = 0;
  for (let b = 0; b < BINS; b++) {
    acc += hist[b];
    if (acc >= count * q) return ((b + 1) / BINS) * max;
  }
  return max;
}

const NB = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];

function walk(start, thin, visited, lo, W, H) {
  const out = [];
  let cur = start;
  for (;;) {
    const x = cur % W, y = (cur / W) | 0;
    let next = -1;
    for (const [dx, dy] of NB) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
      const j = ny * W + nx;
      if (!visited[j] && thin[j] >= lo) { next = j; break; }
    }
    if (next < 0) return out;
    visited[next] = 1;
    out.push(next);
    cur = next;
  }
}

function simplify(pts, eps) {
  if (pts.length < 3) return pts;
  const [ax, ay] = pts[0], [bx, by] = pts[pts.length - 1];
  const dx = bx - ax, dy = by - ay, len = Math.hypot(dx, dy) || 1;
  let maxD = 0, idx = 0;
  for (let i = 1; i < pts.length - 1; i++) {
    const d = Math.abs((pts[i][0] - ax) * dy - (pts[i][1] - ay) * dx) / len;
    if (d > maxD) { maxD = d; idx = i; }
  }
  if (maxD <= eps) return [pts[0], pts[pts.length - 1]];
  return simplify(pts.slice(0, idx + 1), eps).slice(0, -1).concat(simplify(pts.slice(idx), eps));
}
