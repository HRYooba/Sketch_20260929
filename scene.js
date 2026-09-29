import { ENV_ORDER, MAX_LIGHTS, MAX_SEGS } from './renderer.js';

const TAU = Math.PI * 2;

function mulberry32(a) {
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const PALETTE = [
  [1.0, 0.45, 0.12], // orange
  [1.0, 0.3, 0.2],   // red
  [0.5, 0.6, 1.0],   // blue
  [0.7, 0.55, 1.0],  // violet
  [1.0, 0.68, 0.76], // pink
  [1.0, 0.9, 0.78],  // cream
  [0.75, 0.88, 1.0], // ice
  [0.6, 1.0, 0.78],  // mint
  [0.4, 0.9, 0.9],   // teal
];

// 環境光は「基底 + 方向つきの色ローブ」で定義し、フーリエ係数へ射影して GPU に渡す
function envCoeffs(base, lobes) {
  const N = 256;
  const A = Array.from({ length: ENV_ORDER + 1 }, () => [0, 0, 0]);
  const B = Array.from({ length: ENV_ORDER + 1 }, () => [0, 0, 0]);
  for (let k = 0; k < N; k++) {
    const t = (k / N) * TAU;
    const L = base.slice();
    for (const lb of lobes) {
      const w = lb.I * Math.pow(Math.max(0, Math.cos(t - lb.dir)), lb.k);
      for (let c = 0; c < 3; c++) L[c] += lb.col[c] * w;
    }
    for (let n = 0; n <= ENV_ORDER; n++) {
      const cn = Math.cos(n * t), sn = Math.sin(n * t);
      const f = n === 0 ? 1 / N : 2 / N;
      for (let c = 0; c < 3; c++) { A[n][c] += L[c] * cn * f; B[n][c] += L[c] * sn * f; }
    }
  }
  return { A: new Float32Array(A.flat()), B: new Float32Array(B.flat()) };
}

export function randomScene(seed, aspect) {
  const R = mulberry32(seed);
  const pick = (arr) => arr[Math.floor(R() * arr.length)];

  const segs = [];
  let prev = null;
  for (let i = 0, n = 5 + Math.floor(R() * 14); i < n; i++) {
    let x, y;
    if (prev && R() < 0.35) { x = prev.bx; y = prev.by; }
    else { x = (R() - 0.5) * aspect * 1.15; y = (R() - 0.5) * 1.15; }
    const a = R() < 0.5 ? Math.floor(R() * 8) * (Math.PI / 4) : R() * TAU;
    const len = 0.04 + Math.pow(R(), 1.6) * 0.45;
    prev = { ax: x, ay: y, bx: x + Math.cos(a) * len, by: y + Math.sin(a) * len, ph: R() * 10 };
    segs.push(prev);
  }

  const b = 0.004 + Math.pow(R(), 2) * 0.09;
  const lobes = [];
  for (let i = 0, n = 1 + Math.floor(R() * 3); i < n; i++) {
    lobes.push({ dir: R() * TAU, k: 1 + R() * 7, I: 0.2 + R() * 2.3, col: pick(PALETTE) });
  }
  const env = envCoeffs([b * (0.9 + R() * 0.2), b * (0.9 + R() * 0.2), b * (0.9 + R() * 0.2)], lobes);

  const lights = [];
  for (let i = 0, n = 2 + Math.floor(R() * 5); i < n; i++) {
    const l = { r: 0.003 + R() * 0.01, I: 6 + R() * 30, col: pick(PALETTE), ph: R() * 10 };
    if (R() < 0.55) {
      // 線分端のすぐ先に置くと、端から扇状に光が回り込む
      l.seg = pick(segs);
      l.end = R() < 0.5;
      l.side = (R() - 0.5) * 0.01;
    } else {
      l.x0 = (R() - 0.5) * aspect;
      l.y0 = R() - 0.5;
      l.amp = 0.02 + R() * 0.13;
      l.speed = 0.05 + R() * 0.25;
    }
    lights.push(l);
  }

  return {
    segs: segs.slice(0, MAX_SEGS),
    lights: lights.slice(0, MAX_LIGHTS - 1),
    env,
    rot0: R() * TAU,
  };
}

// 各線分を中点まわりにわずかに揺らした姿を返す
export function segsAt(scene, t) {
  return scene.segs.map((s) => {
    const cx = (s.ax + s.bx) / 2, cy = (s.ay + s.by) / 2;
    const a = 0.03 * Math.sin(t * 0.21 + s.ph);
    const c = Math.cos(a), sn = Math.sin(a);
    const hx = (s.bx - s.ax) / 2, hy = (s.by - s.ay) / 2;
    const rx = c * hx - sn * hy, ry = sn * hx + c * hy;
    return { ax: cx - rx, ay: cy - ry, bx: cx + rx, by: cy + ry };
  });
}

// segs は segsAt の結果（光源を揺れた線分端へ追従させるため）
export function lightsAt(scene, segs, t) {
  return scene.lights.map((l) => {
    const I = l.I * (1 + 0.12 * Math.sin(t * 0.6 + l.ph));
    if (l.seg) {
      const s = segs[scene.segs.indexOf(l.seg)];
      const dx = s.bx - s.ax, dy = s.by - s.ay, len = Math.hypot(dx, dy);
      const ux = dx / len, uy = dy / len;
      const off = (l.end ? 1 : -1) * (l.r + 0.004);
      const side = l.side + 0.004 * Math.sin(t * 0.5 + l.ph);
      const tx = l.end ? s.bx : s.ax, ty = l.end ? s.by : s.ay;
      return { x: tx + ux * off - uy * side, y: ty + uy * off + ux * side, r: l.r, I, col: l.col };
    }
    return {
      x: l.x0 + l.amp * Math.sin(t * l.speed + l.ph),
      y: l.y0 + l.amp * Math.cos(t * l.speed * 0.83 + l.ph * 1.7),
      r: l.r, I, col: l.col,
    };
  });
}
