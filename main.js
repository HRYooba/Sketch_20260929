'use strict';

const MAX_BOXES = 16;
const MAX_LIGHTS = 8;
// GI はコストが高いので表示解像度より低く回し、表示パスで拡大する
const GI_SCALE = 0.5;
const DIRECT_STEPS = 64;
const INDIRECT_RAYS = 6;

const canvas = document.getElementById('c');
const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
if (!gl || !gl.getExtension('EXT_color_buffer_float')) {
  document.body.innerHTML = '<p style="color:#aaa;font:14px sans-serif;padding:2em">WebGL2 + float render target が必要です</p>';
  throw new Error('WebGL2 unsupported');
}

const VERT = `#version 300 es
void main(){
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const COMMON = `#version 300 es
precision highp float;
#define MAX_BOXES ${MAX_BOXES}
#define MAX_LIGHTS ${MAX_LIGHTS}
#define PI 3.14159265
#define TAU 6.2831853
uniform vec4 uBoxes[MAX_BOXES];   // cx, cy, hx, hy
uniform float uAngles[MAX_BOXES];
uniform int uNumBoxes;
uniform vec4 uLights[MAX_LIGHTS]; // x, y, radius, intensity
uniform vec3 uLightCols[MAX_LIGHTS];
uniform int uNumLights;
uniform float uAspect;
uniform vec2 uRes;
out vec4 fragColor;

float sdBox(vec2 p, vec2 b){ vec2 d = abs(p) - b; return length(max(d, 0.)) + min(max(d.x, d.y), 0.); }
float sdOcc(vec2 p){
  float d = 1e5;
  for (int i = 0; i < MAX_BOXES; i++) {
    if (i >= uNumBoxes) break;
    vec4 b = uBoxes[i];
    float c = cos(uAngles[i]), s = sin(uAngles[i]);
    vec2 q = p - b.xy;
    q = vec2(c * q.x + s * q.y, -s * q.x + c * q.y);
    d = min(d, sdBox(q, b.zw));
  }
  return d;
}
vec2 toWorld(vec2 uv){ return (uv - .5) * vec2(uAspect, 1.); }
vec2 toUV(vec2 p){ return p / vec2(uAspect, 1.) + .5; }
uint pcg(uint v){ uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
float rnd(inout uint s){ s = pcg(s); return float(s) / 4294967295.; }
`;

const SDF_FRAG = COMMON + `
void main(){
  vec2 p = toWorld(gl_FragCoord.xy / uRes);
  fragColor = vec4(sdOcc(p), 0., 0., 1.);
}`;

const GI_FRAG = COMMON + `
uniform sampler2D uSdf;
uniform sampler2D uPrev;
uniform uint uFrame;
uniform float uBlend;
uniform float uBounce;
uniform vec3 uAmbient;

float occAt(vec2 p){ return texture(uSdf, toUV(p)).x; }

// 遮蔽物に当たった距離を返す。maxT まで何も無ければ -1
float march(vec2 o, vec2 dir, float maxT, float px){
  float t = 0.;
  // 遮蔽物の内側（と境界から 1px 以内）から出るレイは、その箱を抜けるまで遮蔽とみなさない
  bool inside = occAt(o) < px;
  for (int i = 0; i < ${DIRECT_STEPS}; i++) {
    if (t >= maxT) return -1.;
    vec2 q = o + dir * t;
    vec2 uv = toUV(q);
    if (uv.x < 0. || uv.y < 0. || uv.x > 1. || uv.y > 1.) return -1.;
    float d = occAt(q);
    if (inside) { if (d > px) inside = false; t += max(abs(d), px); continue; }
    if (d < px * .5) return t;
    t += max(d, px * .5);
  }
  return -1.;
}

void main(){
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 p = toWorld(uv);
  float px = 1. / uRes.y;
  uint seed = pcg(uint(gl_FragCoord.x) * 1973u + uint(gl_FragCoord.y) * 9277u + uFrame * 26699u);

  vec3 E = vec3(0.);
  // 直接光: 光源円盤上の点へシャドウレイを飛ばし、見込み角で重み付けする
  for (int i = 0; i < MAX_LIGHTS; i++) {
    if (i >= uNumLights) break;
    vec4 L = uLights[i];
    vec2 toL = L.xy - p;
    float d = length(toL);
    vec2 dir = toL / max(d, 1e-5);
    vec2 n = vec2(-dir.y, dir.x);
    vec2 tgt = L.xy + n * (rnd(seed) * 2. - 1.) * L.z;
    vec2 td = tgt - p;
    float dd = length(td);
    if (march(p, td / max(dd, 1e-5), dd, px) < 0.) {
      float ang = d > L.z ? 2. * asin(L.z / d) : PI;
      E += uLightCols[i] * L.w * ang / TAU;
    }
  }
  // 間接光: 当たった面の手前の前フレーム照度を拾う（多重バウンスがフレームをまたいで積もる）
  float jitter = rnd(seed);
  for (int k = 0; k < ${INDIRECT_RAYS}; k++) {
    float a = (float(k) + jitter) / float(${INDIRECT_RAYS}) * TAU;
    vec2 dir = vec2(cos(a), sin(a));
    float h = march(p, dir, 3., px);
    vec3 r = h < 0. ? uAmbient : texture(uPrev, toUV(p + dir * max(h - 2. * px, 0.))).rgb * uBounce;
    E += r / float(${INDIRECT_RAYS});
  }

  vec3 prev = texture(uPrev, uv).rgb;
  fragColor = vec4(mix(prev, E, uBlend), 1.);
}`;

const DISPLAY_FRAG = COMMON + `
uniform sampler2D uGI;
uniform float uExposure;
uniform uint uFrame;

vec3 aces(vec3 x){ return clamp((x * (2.51 * x + .03)) / (x * (2.43 * x + .59) + .14), 0., 1.); }

void main(){
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 p = toWorld(uv);
  float px = 1. / uRes.y;
  vec3 E = texture(uGI, uv).rgb;

  float sd = sdOcc(p);
  float inside = smoothstep(px, -px, sd);
  vec3 alb = mix(vec3(.92, .9, .88), vec3(.78, .77, .8), inside);
  vec3 col = alb * E * uExposure;

  for (int i = 0; i < MAX_LIGHTS; i++) {
    if (i >= uNumLights) break;
    vec4 L = uLights[i];
    float d = length(p - L.xy);
    col += uLightCols[i] * L.w * 6e-6 / (d * d + 1e-4);
  }

  col = pow(aces(col), vec3(1. / 2.2));
  uint seed = pcg(uint(gl_FragCoord.x) * 7919u + uint(gl_FragCoord.y) * 104729u + uFrame * 31u);
  col += (rnd(seed) - .5) * (3. / 255.);
  fragColor = vec4(col, 1.);
}`;

function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function program(frag) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, VERT));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, frag));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const loc = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const name = gl.getActiveUniform(p, i).name.replace(/\[0\]$/, '');
    loc[name] = gl.getUniformLocation(p, name);
  }
  return { p, loc };
}

const progSdf = program(SDF_FRAG);
const progGi = program(GI_FRAG);
const progDisplay = program(DISPLAY_FRAG);
gl.bindVertexArray(gl.createVertexArray());

function target(w, h) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  return { tex, fb, w, h };
}
function freeTarget(t) { if (t) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fb); } }

let W = 0, H = 0, sdfT, giA, giB;
function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  W = Math.floor(innerWidth * dpr);
  H = Math.floor(innerHeight * dpr);
  canvas.width = W;
  canvas.height = H;
  const gw = Math.max(1, Math.floor(W * GI_SCALE)), gh = Math.max(1, Math.floor(H * GI_SCALE));
  [sdfT, giA, giB].forEach(freeTarget);
  sdfT = target(gw, gh);
  giA = target(gw, gh);
  giB = target(gw, gh);
  accum = 0;
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

const PALETTE = [
  [1.0, 0.42, 0.1],
  [0.55, 0.62, 1.0],
  [1.0, 0.7, 0.78],
  [1.0, 0.9, 0.78],
  [0.7, 0.88, 1.0],
  [0.6, 1.0, 0.75],
];

let scene;
function makeScene(seed) {
  const R = mulberry32(seed);
  const aspect = innerWidth / innerHeight;
  const boxes = [];
  const mode = Math.floor(R() * 3);
  if (mode === 0) {
    // 45° に傾いた大きな板
    const n = 4 + Math.floor(R() * 5);
    for (let i = 0; i < n; i++) {
      const h = 0.07 + R() * 0.16;
      boxes.push({
        x: (R() - 0.5) * aspect * 1.1, y: (R() - 0.5) * 1.1,
        hx: h, hy: h * (0.5 + R() * 1.0),
        a: Math.PI / 4 + (R() < 0.3 ? Math.PI / 2 : 0),
      });
    }
  } else if (mode === 1) {
    // 上下の列に並ぶ壁
    const cols = 3 + Math.floor(R() * 3);
    const hx = 0.02 + R() * 0.05;
    const hy = 0.1 + R() * 0.12;
    for (const row of [-1, 1]) {
      for (let c = 0; c < cols; c++) {
        boxes.push({
          x: ((c + 0.5) / cols - 0.5) * aspect,
          y: row * (0.5 - hy * 0.6),
          hx, hy, a: 0,
        });
      }
    }
  } else {
    // 任意角の細長い破片
    const n = 5 + Math.floor(R() * 6);
    for (let i = 0; i < n; i++) {
      boxes.push({
        x: (R() - 0.5) * aspect * 1.1, y: (R() - 0.5) * 1.1,
        hx: 0.03 + R() * 0.2, hy: 0.01 + R() * 0.05,
        a: R() * Math.PI,
      });
    }
  }
  boxes.forEach((b) => { b.a0 = b.a; b.ph = R() * 10; });

  // 光源は箱の角のすぐ外側に置き、箱と一緒に動かす
  const lights = [];
  const nL = Math.min(MAX_LIGHTS - 1, 3 + Math.floor(R() * 3));
  for (let i = 0; i < nL; i++) {
    lights.push({
      box: Math.floor(R() * boxes.length),
      sx: R() < 0.5 ? -1 : 1, sy: R() < 0.5 ? -1 : 1,
      r: 0.006 + R() * 0.01,
      I: 10 + R() * 25,
      col: PALETTE[Math.floor(R() * PALETTE.length)],
      ph: R() * 10,
    });
  }
  scene = { boxes: boxes.slice(0, MAX_BOXES), lights };
  accum = 0;
}

const mouse = { x: 0, y: 0, on: false };
let paused = false;
let accum = 0;
let frame = 0;
let time = 0;
let last = performance.now();

const boxData = new Float32Array(MAX_BOXES * 4);
const angData = new Float32Array(MAX_BOXES);
const lightData = new Float32Array(MAX_LIGHTS * 4);
const lightColData = new Float32Array(MAX_LIGHTS * 3);
let numLights = 0;

function updateScene(t) {
  const { boxes, lights } = scene;
  boxes.forEach((b, i) => {
    b.a = b.a0 + 0.06 * Math.sin(t * 0.17 + b.ph);
    boxData.set([b.x, b.y, b.hx, b.hy], i * 4);
    angData[i] = b.a;
  });
  numLights = 0;
  for (const l of lights) {
    const b = boxes[l.box];
    const c = Math.cos(b.a), s = Math.sin(b.a);
    const off = l.r + 0.006;
    const lx = l.sx * (b.hx + off), ly = l.sy * (b.hy + off);
    const flicker = 1 + 0.15 * Math.sin(t * 0.7 + l.ph);
    lightData.set([b.x + c * lx - s * ly, b.y + s * lx + c * ly, l.r, l.I * flicker], numLights * 4);
    lightColData.set(l.col, numLights * 3);
    numLights++;
  }
  if (mouse.on) {
    lightData.set([mouse.x, mouse.y, 0.012, 20], numLights * 4);
    lightColData.set([1, 0.95, 0.9], numLights * 3);
    numLights++;
  }
}

function setCommon(prog, w, h) {
  const L = prog.loc;
  gl.uniform4fv(L.uBoxes, boxData);
  gl.uniform1fv(L.uAngles, angData);
  gl.uniform1i(L.uNumBoxes, scene.boxes.length);
  gl.uniform4fv(L.uLights, lightData);
  gl.uniform3fv(L.uLightCols, lightColData);
  gl.uniform1i(L.uNumLights, numLights);
  gl.uniform1f(L.uAspect, W / H);
  gl.uniform2f(L.uRes, w, h);
}

function render(now) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  if (!paused) time += dt;
  updateScene(time);

  gl.bindFramebuffer(gl.FRAMEBUFFER, sdfT.fb);
  gl.viewport(0, 0, sdfT.w, sdfT.h);
  gl.useProgram(progSdf.p);
  setCommon(progSdf, sdfT.w, sdfT.h);
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  // 停止中は累積平均で収束させ、動いている間は指数移動平均で追従させる
  accum++;
  const blend = paused && !mouse.on ? Math.max(1 / accum, 0.002) : Math.max(1 / accum, 0.06);

  gl.bindFramebuffer(gl.FRAMEBUFFER, giB.fb);
  gl.viewport(0, 0, giB.w, giB.h);
  gl.useProgram(progGi.p);
  setCommon(progGi, giB.w, giB.h);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, sdfT.tex);
  gl.uniform1i(progGi.loc.uSdf, 0);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, giA.tex);
  gl.uniform1i(progGi.loc.uPrev, 1);
  gl.uniform1ui(progGi.loc.uFrame, frame);
  gl.uniform1f(progGi.loc.uBlend, blend);
  gl.uniform1f(progGi.loc.uBounce, 0.72);
  gl.uniform3f(progGi.loc.uAmbient, 0.04, 0.042, 0.055);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  [giA, giB] = [giB, giA];

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, W, H);
  gl.useProgram(progDisplay.p);
  setCommon(progDisplay, W, H);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, giA.tex);
  gl.uniform1i(progDisplay.loc.uGI, 0);
  gl.uniform1f(progDisplay.loc.uExposure, 0.75);
  gl.uniform1ui(progDisplay.loc.uFrame, frame);
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  frame++;
  requestAnimationFrame(render);
}

// --- input -------------------------------------------------------------

function toWorld(e) {
  const aspect = innerWidth / innerHeight;
  return { x: (e.clientX / innerWidth - 0.5) * aspect, y: (0.5 - e.clientY / innerHeight) };
}
canvas.addEventListener('pointermove', (e) => {
  if (e.pointerType !== 'mouse') return;
  Object.assign(mouse, toWorld(e), { on: true });
});
canvas.addEventListener('pointerleave', () => { mouse.on = false; accum = 0; });
canvas.addEventListener('click', () => makeScene((Math.random() * 2 ** 31) | 0));
addEventListener('keydown', (e) => {
  if (e.code === 'Space') { paused = !paused; accum = 0; e.preventDefault(); }
  if (e.code === 'KeyS') {
    const a = document.createElement('a');
    a.download = `gi-${Date.now()}.png`;
    a.href = canvas.toDataURL('image/png');
    a.click();
  }
});
addEventListener('resize', resize);

const seedParam = new URLSearchParams(location.search).get('seed');
resize();
makeScene(seedParam ? Number(seedParam) : (Math.random() * 2 ** 31) | 0);
requestAnimationFrame(render);
