'use strict';

// 遮蔽物は太さゼロの線分。各画素から見て線分が塞ぐ角度区間を解析的に求め、
// 色つきの環境光（角度のフーリエ級数）と円盤光源を、塞がれていない角度だけ積分する。
// サンプリングを使わないのでノイズも時間方向の蓄積も無い。

const MAX_SEGS = 24;
const MAX_LIGHTS = 8;
const ENV_ORDER = 4;

const canvas = document.getElementById('c');
const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
if (!gl) {
  document.body.innerHTML = '<p style="color:#aaa;font:14px sans-serif;padding:2em">WebGL2 が必要です</p>';
  throw new Error('WebGL2 unsupported');
}

const params = new URLSearchParams(location.search);
const RES_SCALE = Number(params.get('res')) || 1;

const VERT = `#version 300 es
void main(){
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
#define MAX_SEGS ${MAX_SEGS}
#define MAX_LIGHTS ${MAX_LIGHTS}
#define ENV_ORDER ${ENV_ORDER}
#define MAXI ${MAX_SEGS * 2}
#define PI 3.14159265
#define TAU 6.2831853

uniform vec4 uSegs[MAX_SEGS];        // ax, ay, bx, by
uniform int uNumSegs;
uniform vec4 uLights[MAX_LIGHTS];    // x, y, radius, intensity
uniform vec3 uLightCols[MAX_LIGHTS];
uniform int uNumLights;
uniform vec3 uEnvA[ENV_ORDER + 1];   // L(θ) = A0 + Σ An cos nθ' + Bn sin nθ'
uniform vec3 uEnvB[ENV_ORDER + 1];
uniform float uEnvRot;
uniform float uBlocked;              // 塞がれた方向から返ってくる光の割合（簡易バウンス）
uniform float uExposure;
uniform float uLineAlpha;
uniform float uFalloff;
uniform vec2 uRes;
uniform float uAspect;
uniform uint uFrame;
out vec4 fragColor;

float gS[MAXI];
float gE[MAXI];
int gN;

void push(float s, float e){
  if (e > s && gN < MAXI) { gS[gN] = s; gE[gN] = e; gN++; }
}

void sortMerge(){
  for (int i = 1; i < MAXI; i++) {
    if (i >= gN) break;
    float s = gS[i], e = gE[i];
    int j = i - 1;
    while (j >= 0 && gS[j] > s) { gS[j + 1] = gS[j]; gE[j + 1] = gE[j]; j--; }
    gS[j + 1] = s; gE[j + 1] = e;
  }
  int m = 0;
  for (int i = 0; i < MAXI; i++) {
    if (i >= gN) break;
    if (m > 0 && gS[i] <= gE[m - 1]) gE[m - 1] = max(gE[m - 1], gE[i]);
    else { gS[m] = gS[i]; gE[m] = gE[i]; m++; }
  }
  gN = m;
}

float wrapPi(float a){ return a - TAU * floor((a + PI) / TAU); }
float ang(vec2 v){ float a = atan(v.y, v.x); return a < 0. ? a + TAU : a; }

vec3 envInt(float s, float e){
  vec3 r = uEnvA[0] * (e - s);
  for (int n = 1; n <= ENV_ORDER; n++) {
    float fn = float(n);
    float ce = fn * (e - uEnvRot), cs = fn * (s - uEnvRot);
    r += (uEnvA[n] * (sin(ce) - sin(cs)) - uEnvB[n] * (cos(ce) - cos(cs))) / fn;
  }
  return r;
}

// 区間 [s, s+len] のうち [lo, hi] に入る長さ（±TAU 巻き戻した分も数える）
float overlap(float s, float len, float lo, float hi){
  float o = 0.;
  for (int k = -1; k <= 1; k++) {
    float a = s + float(k) * TAU;
    o += max(0., min(a + len, hi) - max(a, lo));
  }
  return o;
}

float sdSeg(vec2 p, vec2 a, vec2 b){
  vec2 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0., 1.);
  return length(pa - ba * h);
}

vec3 aces(vec3 x){ return clamp((x * (2.51 * x + .03)) / (x * (2.43 * x + .59) + .14), 0., 1.); }
uint pcg(uint v){ uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }

vec3 shade(vec2 p){
  // 環境光: 線分が塞ぐ角度区間の和集合を除いて積分する
  gN = 0;
  for (int i = 0; i < MAX_SEGS; i++) {
    if (i >= uNumSegs) break;
    vec4 sg = uSegs[i];
    float a = ang(sg.xy - p);
    float d = wrapPi(ang(sg.zw - p) - a);
    float s = d > 0. ? a : a + d;
    if (s < 0.) s += TAU;
    float e = s + abs(d);
    if (e > TAU) { push(s, TAU); push(0., e - TAU); }
    else push(s, e);
  }
  sortMerge();
  vec3 total = uEnvA[0] * TAU;
  vec3 blocked = vec3(0.);
  for (int i = 0; i < MAXI; i++) {
    if (i >= gN) break;
    blocked += envInt(gS[i], gE[i]);
  }
  vec3 E = (total - (1. - uBlocked) * blocked) / TAU;

  // 円盤光源: 光源より手前にある線分の部分だけが遮る。
  // 見込み角が小さいので遮蔽の重なりは和で近似し、見込み角で頭打ちにする
  for (int l = 0; l < MAX_LIGHTS; l++) {
    if (l >= uNumLights) break;
    vec4 L = uLights[l];
    vec2 toL = L.xy - p;
    float dist = length(toL);
    float half_ = asin(min(L.z / max(dist, 1e-5), 1.));
    float phi = atan(toL.y, toL.x);
    float cov = 0.;
    for (int i = 0; i < MAX_SEGS; i++) {
      if (i >= uNumSegs) break;
      vec4 sg = uSegs[i];
      vec2 a = sg.xy - p, dv = sg.zw - sg.xy;
      float A = dot(dv, dv), B = dot(a, dv), C = dot(a, a) - dist * dist;
      float disc = B * B - A * C;
      if (disc <= 0.) continue;
      float sq = sqrt(disc);
      float t0 = max((-B - sq) / A, 0.), t1 = min((-B + sq) / A, 1.);
      if (t0 >= t1) continue;
      float ra = wrapPi(atan(a.y + t0 * dv.y, a.x + t0 * dv.x) - phi);
      float rb = wrapPi(atan(a.y + t1 * dv.y, a.x + t1 * dv.x) - phi);
      float d = wrapPi(rb - ra);
      cov += overlap(d > 0. ? ra : ra + d, abs(d), -half_, half_);
    }
    float vis = max(2. * half_ - cov, 0.);
    // 見込み角による 1/d に加え、光源が画面全体を照らしすぎないよう減衰をかける
    E += uLightCols[l] * L.w * vis / TAU / (1. + uFalloff * dist);
  }
  return E;
}

void main(){
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 p = (uv - .5) * vec2(uAspect, 1.);
  float px = 1. / uRes.y;

  vec3 col = shade(p) * uExposure;

  float dl = 1e5;
  for (int i = 0; i < MAX_SEGS; i++) {
    if (i >= uNumSegs) break;
    dl = min(dl, sdSeg(p, uSegs[i].xy, uSegs[i].zw));
  }
  col *= 1. - uLineAlpha * smoothstep(1.5 * px, 0., dl);

  for (int l = 0; l < MAX_LIGHTS; l++) {
    if (l >= uNumLights) break;
    vec4 L = uLights[l];
    float d = length(p - L.xy);
    col += uLightCols[l] * L.w * 2.5e-6 / (d * d + 5e-5);
  }

  col = pow(aces(col), vec3(1. / 2.2));
  uint s = pcg(uint(gl_FragCoord.x) * 7919u + uint(gl_FragCoord.y) * 104729u + uFrame * 31u);
  col += (float(s) / 4294967295. - .5) * (2. / 255.);
  fragColor = vec4(col, 1.);
}`;

function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
const prog = gl.createProgram();
gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
const loc = {};
for (let i = 0, n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS); i < n; i++) {
  const name = gl.getActiveUniform(prog, i).name.replace(/\[0\]$/, '');
  loc[name] = gl.getUniformLocation(prog, name);
}
gl.useProgram(prog);
gl.bindVertexArray(gl.createVertexArray());

let W = 0, H = 0;
function resize() {
  // 画素あたりの計算が重いので DPR は 1.5 で頭打ちにする
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5) * RES_SCALE;
  W = Math.floor(innerWidth * dpr);
  H = Math.floor(innerHeight * dpr);
  canvas.width = W;
  canvas.height = H;
}

// --- scene -------------------------------------------------------------

function mulberry32(a) {
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const COLORS = {
  orange: [1.0, 0.45, 0.12],
  blue: [0.5, 0.6, 1.0],
  pink: [1.0, 0.68, 0.76],
  cream: [1.0, 0.9, 0.78],
  ice: [0.75, 0.88, 1.0],
  mint: [0.6, 1.0, 0.78],
};
const pick = (R, arr) => arr[Math.floor(R() * arr.length)];

// 環境光は「基底 + 方向つきの色ローブ」で定義し、フーリエ係数へ射影して GPU に渡す
function envCoeffs(base, lobes) {
  const N = 256;
  const A = Array.from({ length: ENV_ORDER + 1 }, () => [0, 0, 0]);
  const B = Array.from({ length: ENV_ORDER + 1 }, () => [0, 0, 0]);
  for (let k = 0; k < N; k++) {
    const t = (k / N) * Math.PI * 2;
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

let scene;

function makeScene(seed) {
  const R = mulberry32(seed);
  const aspect = innerWidth / innerHeight;
  const segs = [];
  const lights = [];
  let env;
  const mode = params.get('mode') !== null ? Number(params.get('mode')) : Math.floor(R() * 3);

  // 光源は線分端のすぐ先に置く（端から扇状に光が回り込む）
  const tipLight = (s, end, col, I) => {
    const r = 0.004 + R() * 0.006;
    lights.push({ seg: s, end, off: (end ? 1 : -1) * (r + 0.004), side: (R() - 0.5) * 0.01, r, I, col, ph: R() * 10 });
  };

  if (mode === 0) {
    // 45° の折れ線（V 字・菱形の一部）
    const n = 4 + Math.floor(R() * 4);
    for (let i = 0; i < n; i++) {
      let x = (R() - 0.5) * aspect * 1.1, y = (R() - 0.5) * 1.1;
      let dir = Math.floor(R() * 4) * 2 + 1;
      const k = 1 + Math.floor(R() * 3);
      for (let j = 0; j < k; j++) {
        const len = 0.12 + R() * 0.3;
        const a = (dir * Math.PI) / 4;
        const nx = x + Math.cos(a) * len, ny = y + Math.sin(a) * len;
        segs.push({ ax: x, ay: y, bx: nx, by: ny });
        x = nx; y = ny;
        dir = (dir + (R() < 0.5 ? 2 : 6)) % 8;
      }
    }
    env = envCoeffs([0.06, 0.06, 0.07], [
      { dir: Math.PI * 0.3, k: 4, I: 2.2, col: COLORS.orange },
      { dir: Math.PI * 1.1, k: 3, I: 1.4, col: COLORS.blue },
      { dir: Math.PI * 1.6, k: 2, I: 0.8, col: COLORS.cream },
    ]);
    for (let i = 0; i < 4; i++) tipLight(pick(R, segs), R() < 0.5, pick(R, [COLORS.ice, COLORS.cream, COLORS.blue]), 14 + R() * 12);
  } else if (mode === 1) {
    // 暗い空間に水平の棚が浮く
    const n = 2 + Math.floor(R() * 3);
    for (let i = 0; i < n; i++) {
      const y = ((i + 0.5) / n - 0.5) * 0.8 + (R() - 0.5) * 0.05;
      const x = (R() - 0.5) * 0.3;
      const w = 0.08 + R() * 0.12;
      segs.push({ ax: x - w, ay: y, bx: x + w, by: y });
    }
    env = envCoeffs([0.01, 0.01, 0.012], [
      { dir: Math.PI * 0.5, k: 8, I: 0.25, col: COLORS.cream },
      { dir: Math.PI * 1.5, k: 8, I: 0.2, col: COLORS.pink },
      { dir: 0, k: 2, I: 0.08, col: COLORS.mint },
    ]);
    segs.forEach((s) => {
      tipLight(s, false, pick(R, [COLORS.pink, COLORS.cream]), 18 + R() * 14);
      if (R() < 0.5) tipLight(s, true, pick(R, [COLORS.mint, COLORS.ice]), 8 + R() * 8);
    });
  } else {
    // 上下に並ぶ短い縦の壁
    const cols = 3 + Math.floor(R() * 3);
    const h = 0.1 + R() * 0.1;
    for (const row of [-1, 1]) {
      for (let c = 0; c < cols; c++) {
        const x = ((c + 0.5) / cols - 0.5) * aspect * 0.95;
        segs.push({ ax: x, ay: row * 0.5, bx: x, by: row * (0.5 - h) });
      }
    }
    env = envCoeffs([0.04, 0.03, 0.03], [
      { dir: 0, k: 2, I: 0.3, col: COLORS.pink },
      { dir: Math.PI, k: 2, I: 0.3, col: COLORS.pink },
    ]);
    segs.forEach((s) => {
      if (R() < 0.6) tipLight(s, true, s.ay < 0 ? COLORS.orange : pick(R, [COLORS.cream, COLORS.pink]), 12 + R() * 14);
    });
  }

  segs.forEach((s) => { s.ph = R() * 10; s.ax0 = s.ax; s.ay0 = s.ay; s.bx0 = s.bx; s.by0 = s.by; });
  scene = { segs: segs.slice(0, MAX_SEGS), lights: lights.slice(0, MAX_LIGHTS - 1), env, rot0: R() * Math.PI * 2 };
}

const mouse = { x: 0, y: 0, on: false };
let paused = false;
let frame = 0;
let time = 0;
let last = performance.now();

const segData = new Float32Array(MAX_SEGS * 4);
const lightData = new Float32Array(MAX_LIGHTS * 4);
const lightColData = new Float32Array(MAX_LIGHTS * 3);

function update(t) {
  const { segs, lights } = scene;
  segs.forEach((s, i) => {
    // 各線分を中点まわりにわずかに揺らす
    const cx = (s.ax0 + s.bx0) / 2, cy = (s.ay0 + s.by0) / 2;
    const a = 0.03 * Math.sin(t * 0.21 + s.ph);
    const c = Math.cos(a), sn = Math.sin(a);
    const rot = (x, y) => [cx + c * (x - cx) - sn * (y - cy), cy + sn * (x - cx) + c * (y - cy)];
    [s.ax, s.ay] = rot(s.ax0, s.ay0);
    [s.bx, s.by] = rot(s.bx0, s.by0);
    segData.set([s.ax, s.ay, s.bx, s.by], i * 4);
  });
  let n = 0;
  for (const l of lights) {
    const s = l.seg;
    const dx = s.bx - s.ax, dy = s.by - s.ay, len = Math.hypot(dx, dy);
    const ux = dx / len, uy = dy / len;
    const tx = l.end ? s.bx : s.ax, ty = l.end ? s.by : s.ay;
    const side = l.side + 0.004 * Math.sin(t * 0.5 + l.ph);
    lightData.set([tx + ux * l.off - uy * side, ty + uy * l.off + ux * side, l.r, l.I * (1 + 0.12 * Math.sin(t * 0.6 + l.ph))], n * 4);
    lightColData.set(l.col, n * 3);
    n++;
  }
  if (mouse.on) {
    lightData.set([mouse.x, mouse.y, 0.008, 22], n * 4);
    lightColData.set([1, 0.95, 0.9], n * 3);
    n++;
  }
  return n;
}

function render(now) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  if (!paused) time += dt;
  const numLights = update(time);

  gl.viewport(0, 0, W, H);
  gl.uniform4fv(loc.uSegs, segData);
  gl.uniform1i(loc.uNumSegs, scene.segs.length);
  gl.uniform4fv(loc.uLights, lightData);
  gl.uniform3fv(loc.uLightCols, lightColData);
  gl.uniform1i(loc.uNumLights, numLights);
  gl.uniform3fv(loc.uEnvA, scene.env.A);
  gl.uniform3fv(loc.uEnvB, scene.env.B);
  gl.uniform1f(loc.uEnvRot, scene.rot0 + time * 0.03);
  gl.uniform1f(loc.uBlocked, 0.04);
  gl.uniform1f(loc.uFalloff, 8.0);
  gl.uniform1f(loc.uExposure, 0.7);
  gl.uniform1f(loc.uLineAlpha, 0.25);
  gl.uniform2f(loc.uRes, W, H);
  gl.uniform1f(loc.uAspect, W / H);
  gl.uniform1ui(loc.uFrame, frame);
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  frame++;
  requestAnimationFrame(render);
}

// --- input -------------------------------------------------------------

canvas.addEventListener('pointermove', (e) => {
  if (e.pointerType !== 'mouse') return;
  mouse.x = (e.clientX / innerWidth - 0.5) * (innerWidth / innerHeight);
  mouse.y = 0.5 - e.clientY / innerHeight;
  mouse.on = true;
});
canvas.addEventListener('pointerleave', () => { mouse.on = false; });
canvas.addEventListener('click', () => makeScene((Math.random() * 2 ** 31) | 0));
addEventListener('keydown', (e) => {
  if (e.code === 'Space') { paused = !paused; e.preventDefault(); }
  if (e.code === 'KeyS') {
    const a = document.createElement('a');
    a.download = `gi-${Date.now()}.png`;
    a.href = canvas.toDataURL('image/png');
    a.click();
  }
});
addEventListener('resize', resize);

resize();
makeScene(params.get('seed') ? Number(params.get('seed')) : (Math.random() * 2 ** 31) | 0);
requestAnimationFrame(render);
