// 解析的な 2D 照明（少数の線分向け）。
// 遮蔽物は太さゼロの線分。各画素から見て線分が塞ぐ角度区間を解析的に求め、
// 色つきの環境光（角度のフーリエ級数）と円盤光源を、塞がれていない角度だけ積分する。
// 表示解像度で厳密に解くので影の境目が最もシャープに出るが、線分の多いカメラ入力には使えない。
import { ENV_ORDER, MAX_LIGHTS } from './renderer.js';

// 画素ごとに線分の角度区間を挿入ソートする。コストは線分数の 2 乗で増え、
// 作業配列（線分数の 2 倍）が大きいとレジスタから溢れて急に遅くなる
export const MAX_SEGS = 24;

export function createAnalyticRenderer(canvas) {
  const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
  if (!gl) throw new Error('WebGL2 unsupported');

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

  const segData = new Float32Array(MAX_SEGS * 4);
  const lightData = new Float32Array(MAX_LIGHTS * 4);
  const lightColData = new Float32Array(MAX_LIGHTS * 3);

  return {
    // scale: 表示解像度に対する描画解像度の比
    resize(cssW, cssH, scale) {
      const dpr = Math.min(devicePixelRatio || 1, 1.5) * scale;
      canvas.width = Math.floor(cssW * dpr);
      canvas.height = Math.floor(cssH * dpr);
    },
    // segs, lights は画面座標（高さ 1、中心原点、y 上向き）
    draw(f) {
      const segs = f.segs.slice(0, MAX_SEGS);
      segs.forEach((s, i) => segData.set([s.ax, s.ay, s.bx, s.by], i * 4));
      const lights = f.lights.slice(0, MAX_LIGHTS);
      lights.forEach((l, i) => {
        lightData.set([l.x, l.y, l.r, l.I], i * 4);
        lightColData.set(l.col, i * 3);
      });
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.uniform4fv(loc.uSegs, segData);
      gl.uniform1i(loc.uNumSegs, segs.length);
      gl.uniform4fv(loc.uLights, lightData);
      gl.uniform3fv(loc.uLightCols, lightColData);
      gl.uniform1i(loc.uNumLights, lights.length);
      gl.uniform3fv(loc.uEnvA, f.env.A);
      gl.uniform3fv(loc.uEnvB, f.env.B);
      gl.uniform1f(loc.uEnvRot, f.envRot);
      gl.uniform1f(loc.uBlocked, 0.04);
      gl.uniform1f(loc.uFalloff, 8.0);
      gl.uniform1f(loc.uExposure, 0.7);
      gl.uniform1f(loc.uLineAlpha, 0.25);
      gl.uniform2f(loc.uRes, canvas.width, canvas.height);
      gl.uniform1f(loc.uAspect, canvas.width / canvas.height);
      gl.uniform1ui(loc.uFrame, f.frame);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    },
  };
}
