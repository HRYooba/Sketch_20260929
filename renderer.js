// 遮蔽物は太さゼロの線分。各画素から見て線分が塞ぐ角度区間を解析的に求め、
// 色つきの環境光（角度のフーリエ級数）と円盤光源を、塞がれていない角度だけ積分する。
// サンプリングを使わないのでノイズも時間方向の蓄積も無い。
//
// GI（間接光）: 各線分の両面に点を並べて「壁が受けた光 × 反射率」を求め（壁パス）、
// 低解像度の画素から多数の光線を飛ばして最初に当たった壁面の明るさを集める（間接光パス）。
// 壁パスは前フレームの壁の明るさも光源として拾うので、フレームを重ねるごとに多重反射へ収束する。

// 画素ごとに線分の角度区間を挿入ソートする。コストは線分数の 2 乗で増え、
// 作業配列（線分数の 2 倍）が大きいとレジスタから溢れて急に遅くなる
export const MAX_SEGS = 24;
export const MAX_LIGHTS = 8;
export const ENV_ORDER = 4;
// 壁 1 面あたりの標本点数、間接光の光線数、間接光バッファの解像度比
const WALL_SAMPLES = 32;
const GATHER_RAYS = 64;
const INDIRECT_SCALE = 1 / 3;

export function createRenderer(canvas) {
  const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
  if (!gl) throw new Error('WebGL2 unsupported');

  const VERT = `#version 300 es
  void main(){
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
  }`;

  const COMMON = `#version 300 es
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
  uniform sampler2D uWalls;            // x: 面上の位置, y: 線分 i の面 side（2i+side 行目）
  uniform float uGI;
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

  // 平らな壁の上の点は自分の壁を見られないので、壁パスでは自分の線分を除く
  int gSelf = -1;

  // p から dir へ飛ばした光線が最初に当たる壁面の放射輝度（当たらなければ 0）
  vec3 wallHit(vec2 p, vec2 dir){
    float best = 1e9;
    vec2 hit = vec2(-1.);
    for (int i = 0; i < MAX_SEGS; i++) {
      if (i >= uNumSegs) break;
      if (i == gSelf) continue;
      vec2 a = uSegs[i].xy, e = uSegs[i].zw - a;
      float den = dir.x * e.y - dir.y * e.x;
      if (abs(den) < 1e-9) continue;
      vec2 ap = a - p;
      float t = (ap.x * e.y - ap.y * e.x) / den;
      float u = (ap.x * dir.y - ap.y * dir.x) / den;
      if (t > 1e-4 && t < best && u >= 0. && u <= 1.) {
        best = t;
        float side = (e.x * (p.y - a.y) - e.y * (p.x - a.x)) > 0. ? 0. : 1.;
        hit = vec2(u, (float(i) * 2. + side + .5) / float(MAX_SEGS * 2));
      }
    }
    return hit.x < 0. ? vec3(0.) : texture(uWalls, hit).rgb;
  }

  // 等角に並べた光線を画素ごとに回転させて、方向の刻みを縞ではなく細かいノイズにする
  vec3 gather(vec2 p, float rot){
    vec3 acc = vec3(0.);
    for (int k = 0; k < ${GATHER_RAYS}; k++) {
      float a = (float(k) + rot) / float(${GATHER_RAYS}) * TAU;
      acc += wallHit(p, vec2(cos(a), sin(a)));
    }
    return acc / float(${GATHER_RAYS});
  }

  float ign(vec2 px){ return fract(52.9829189 * fract(dot(px, vec2(.06711056, .00583715)))); }

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

  `;

  // 壁パス: 出力 (x, y) = 線分 y/2 の面 y%2 の位置 x。面からわずかに離した点の照度に反射率を掛ける。
  // 2D のランバート面は半円の照度を 2 倍したものが放射輝度になる（∫cos = 2 に対し照度は角度平均のため）
  const WALL_FRAG = COMMON + `
  uniform float uAlbedo;
  void main(){
    int i = int(gl_FragCoord.y) / 2;
    float side = mod(floor(gl_FragCoord.y), 2.);
    if (i >= uNumSegs) { fragColor = vec4(0.); return; }
    vec2 a = uSegs[i].xy, e = uSegs[i].zw - a;
    vec2 n = normalize(vec2(-e.y, e.x)) * (side < .5 ? 1. : -1.);
    vec2 p = a + e * (gl_FragCoord.x / float(${WALL_SAMPLES})) + n * 1e-3;
    gSelf = i;
    vec3 E = shade(p) + gather(p, ign(gl_FragCoord.xy));
    fragColor = vec4(uAlbedo * 2. * E, 1.);
  }`;

  const INDIRECT_FRAG = COMMON + `
  void main(){
    vec2 p = (gl_FragCoord.xy / uRes - .5) * vec2(uAspect, 1.);
    fragColor = vec4(gather(p, ign(gl_FragCoord.xy)), 1.);
  }`;

  const FRAG = COMMON + `
  uniform sampler2D uIndirect;
  uniform vec2 uIndirectRes;

  bool crosses(vec2 p, vec2 q){
    for (int i = 0; i < MAX_SEGS; i++) {
      if (i >= uNumSegs) break;
      vec2 a = uSegs[i].xy, b = uSegs[i].zw;
      vec2 r = q - p, e = b - a;
      float den = r.x * e.y - r.y * e.x;
      if (abs(den) < 1e-9) continue;
      vec2 ap = a - p;
      float t = (ap.x * e.y - ap.y * e.x) / den;
      float u = (ap.x * r.y - ap.y * r.x) / den;
      if (t > 0. && t < 1. && u >= 0. && u <= 1.) return true;
    }
    return false;
  }

  // 低解像度の間接光を、線分をまたがない近傍 4x4 テクセルだけで重み付き平均して拡大する
  vec3 indirectAt(vec2 uv, vec2 p){
    vec2 g = uv * uIndirectRes - .5;
    vec2 g0 = floor(g);
    vec3 acc = vec3(0.);
    float ws = 0.;
    for (int y = -1; y <= 2; y++) {
      for (int x = -1; x <= 2; x++) {
        vec2 t = g0 + vec2(x, y);
        vec2 tuv = (t + .5) / uIndirectRes;
        vec2 tp = (tuv - .5) * vec2(uAspect, 1.);
        vec2 d = g - t;
        float w = exp(-dot(d, d) * .5);
        if (crosses(p, tp)) continue;
        acc += texture(uIndirect, tuv).rgb * w;
        ws += w;
      }
    }
    return ws > 0. ? acc / ws : vec3(0.);
  }

  void main(){
    vec2 uv = gl_FragCoord.xy / uRes;
    vec2 p = (uv - .5) * vec2(uAspect, 1.);
    float px = 1. / uRes.y;

    vec3 E = shade(p);
    if (uGI > .5) E += indirectAt(uv, p);
    vec3 col = E * uExposure;

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
  function program(frag) {
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, frag));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    const loc = {};
    for (let i = 0, n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS); i < n; i++) {
      const name = gl.getActiveUniform(prog, i).name.replace(/\[0\]$/, '');
      loc[name] = gl.getUniformLocation(prog, name);
    }
    return { prog, loc };
  }
  const P = { main: program(FRAG), wall: program(WALL_FRAG), indirect: program(INDIRECT_FRAG) };
  gl.bindVertexArray(gl.createVertexArray());

  if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('float render target が必要です');
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
  const free = (t) => { if (t) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fb); } };
  let wallA = target(WALL_SAMPLES, MAX_SEGS * 2), wallB = target(WALL_SAMPLES, MAX_SEGS * 2);
  let indirect = null;

  const segData = new Float32Array(MAX_SEGS * 4);
  const lightData = new Float32Array(MAX_LIGHTS * 4);
  const lightColData = new Float32Array(MAX_LIGHTS * 3);

  return {
    // scale: 表示解像度に対する描画解像度の比
    resize(cssW, cssH, scale) {
      const dpr = Math.min(devicePixelRatio || 1, 1.5) * scale;
      canvas.width = Math.floor(cssW * dpr);
      canvas.height = Math.floor(cssH * dpr);
      free(indirect);
      indirect = target(Math.max(1, Math.round(canvas.width * INDIRECT_SCALE)), Math.max(1, Math.round(canvas.height * INDIRECT_SCALE)));
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
      const common = (loc, w, h) => {
        gl.uniform4fv(loc.uSegs, segData);
        gl.uniform1i(loc.uNumSegs, segs.length);
        gl.uniform4fv(loc.uLights, lightData);
        gl.uniform3fv(loc.uLightCols, lightColData);
        gl.uniform1i(loc.uNumLights, lights.length);
        gl.uniform3fv(loc.uEnvA, f.env.A);
        gl.uniform3fv(loc.uEnvB, f.env.B);
        gl.uniform1f(loc.uEnvRot, f.envRot);
        // GI 有効時は塞がれた方向の底上げをやめ、壁からの反射だけにする
        gl.uniform1f(loc.uBlocked, f.gi ? 0 : 0.04);
        gl.uniform1f(loc.uFalloff, 8.0);
        gl.uniform1f(loc.uExposure, 0.7);
        gl.uniform1f(loc.uLineAlpha, 0.25);
        gl.uniform2f(loc.uRes, w, h);
        gl.uniform1f(loc.uAspect, canvas.width / canvas.height);
        gl.uniform1ui(loc.uFrame, f.frame);
        gl.uniform1f(loc.uGI, f.gi ? 1 : 0);
      };
      const bindTex = (unit, loc, tex) => {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.uniform1i(loc, unit);
      };

      if (f.gi) {
        gl.useProgram(P.wall.prog);
        gl.bindFramebuffer(gl.FRAMEBUFFER, wallB.fb);
        gl.viewport(0, 0, wallB.w, wallB.h);
        common(P.wall.loc, wallB.w, wallB.h);
        bindTex(0, P.wall.loc.uWalls, wallA.tex);
        gl.uniform1f(P.wall.loc.uAlbedo, Number(new URLSearchParams(location.search).get("albedo")) || 0.5);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        [wallA, wallB] = [wallB, wallA];

        gl.useProgram(P.indirect.prog);
        gl.bindFramebuffer(gl.FRAMEBUFFER, indirect.fb);
        gl.viewport(0, 0, indirect.w, indirect.h);
        common(P.indirect.loc, indirect.w, indirect.h);
        bindTex(0, P.indirect.loc.uWalls, wallA.tex);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }

      const M = P.main.loc;
      gl.useProgram(P.main.prog);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, canvas.width, canvas.height);
      common(M, canvas.width, canvas.height);
      bindTex(0, M.uWalls, wallA.tex);
      bindTex(1, M.uIndirect, indirect.tex);
      gl.uniform2f(M.uIndirectRes, indirect.w, indirect.h);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    },
  };
}
