// 2D Radiance Cascades（多数の遮蔽物向け。カメラのエッジに使う）。
// 遮蔽物と光源を低解像度のグリッドへラスタライズし、JFA で距離場を作り、
// 角度分解能と区間長を段ごとに 4 倍にしたプローブ群で光を集めて上の段から合成する。
// コストは遮蔽物の数に依らずグリッド解像度だけで決まる。

export const MAX_SEGS = 64;
export const MAX_LIGHTS = 8;
export const ENV_ORDER = 4;

// 上位段のプローブ数を整数にするため、グリッドの縦横は 2^段数 の倍数に丸める
const MAX_CASCADES = 7;
const MARCH_STEPS = 32;

const VERT = `#version 300 es
void main(){
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const HEAD = `#version 300 es
precision highp float;
precision highp int;
#define TAU 6.2831853
#define MAX_SEGS ${MAX_SEGS}
#define MAX_LIGHTS ${MAX_LIGHTS}
#define ENV_ORDER ${ENV_ORDER}
out vec4 o;
`;

// 光源はグリッド座標（テクセル単位）の円盤
const LIGHTS = `
uniform vec4 uLights[MAX_LIGHTS];    // x, y, radius, intensity
uniform vec3 uLightCols[MAX_LIGHTS];
uniform int uNumLights;
vec4 lightAt(vec2 p){
  for (int l = 0; l < MAX_LIGHTS; l++) {
    if (l >= uNumLights) break;
    if (length(p - uLights[l].xy) < uLights[l].z) return vec4(uLightCols[l] * uLights[l].w, 1.);
  }
  return vec4(0.);
}
`;

// 線（遮蔽物）は表示解像度のマスクとして持つ。GI グリッドの遮蔽物はこのマスクを縮小して作り、
// 表示パスではこのマスクで線をまたぐ補間を切る
const SD_SEG = `
float sdSeg(vec2 p, vec2 a, vec2 b){
  vec2 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0., 1.);
  return length(pa - ba * h);
}
`;

// uSegs は表示範囲を [0,1] とした座標
const MASK_SEGS = HEAD + SD_SEG + `
uniform vec4 uSegs[MAX_SEGS];
uniform int uNumSegs;
uniform vec2 uMaskSize;
uniform float uLineWidth;
void main(){
  vec2 p = gl_FragCoord.xy;
  float d = 1e9;
  for (int i = 0; i < MAX_SEGS; i++) {
    if (i >= uNumSegs) break;
    d = min(d, sdSeg(p, uSegs[i].xy * uMaskSize, uSegs[i].zw * uMaskSize));
  }
  o = vec4(clamp(uLineWidth + .5 - d, 0., 1.), 0., 0., 1.);
}`;

// カメラの輝度を時間方向に平滑化して、エッジのちらつきを抑える
const LUMA = HEAD + `
uniform sampler2D uVideo;
uniform sampler2D uPrev;
uniform vec2 uMaskSize;
uniform vec2 uCover;
uniform float uBlend;
void main(){
  vec2 uv = gl_FragCoord.xy / uMaskSize;
  vec2 v = (uv - .5) * uCover + .5;
  v.x = 1. - v.x;
  float l = dot(texture(uVideo, v).rgb, vec3(.299, .587, .114));
  float prev = texelFetch(uPrev, ivec2(gl_FragCoord.xy), 0).r;
  o = vec4(mix(prev, l, uBlend), 0., 0., 1.);
}`;

const MASK_EDGES = HEAD + `
uniform sampler2D uLuma;
uniform float uThresh;
float lum(ivec2 p){ return texelFetch(uLuma, clamp(p, ivec2(0), textureSize(uLuma, 0) - 1), 0).r; }
void main(){
  ivec2 q = ivec2(gl_FragCoord.xy);
  float a = lum(q + ivec2(-1, 1)), b = lum(q + ivec2(0, 1)), c = lum(q + ivec2(1, 1));
  float d = lum(q + ivec2(-1, 0)), f = lum(q + ivec2(1, 0));
  float g = lum(q + ivec2(-1, -1)), h = lum(q + ivec2(0, -1)), k = lum(q + ivec2(1, -1));
  float gx = c + 2. * f + k - a - 2. * d - g;
  float gy = a + 2. * b + c - g - 2. * h - k;
  float m = length(vec2(gx, gy));
  o = vec4(smoothstep(uThresh * .8, uThresh * 1.25, m), 0., 0., 1.);
}`;

// グリッド 1 テクセルが覆うマスクの範囲に線が少しでもあれば遮蔽物にする（光漏れ防止のため保守的に太らせる）
const SCENE = HEAD + LIGHTS + `
uniform sampler2D uMask;
uniform vec2 uView;
void main(){
  vec2 p = gl_FragCoord.xy;
  vec4 L = lightAt(p);
  if (L.a > 0.) { o = L; return; }
  if (p.x > uView.x + 1. || p.y > uView.y + 1.) { o = vec4(0.); return; }
  float m = 0.;
  for (int y = 0; y < 3; y++) {
    for (int x = 0; x < 3; x++) {
      vec2 q = floor(p) + (vec2(x, y) + .5) / 3.;
      m = max(m, texture(uMask, q / uView).r);
    }
  }
  o = vec4(0., 0., 0., m > .3 ? 1. : 0.);
}`;

const JFA_SEED = HEAD + `
uniform sampler2D uScene;
void main(){
  o = texelFetch(uScene, ivec2(gl_FragCoord.xy), 0).a > .5 ? vec4(gl_FragCoord.xy, 0., 1.) : vec4(-1e4, -1e4, 0., 1.);
}`;

const JFA_STEP = HEAD + `
uniform sampler2D uSrc;
uniform int uStep;
void main(){
  ivec2 ip = ivec2(gl_FragCoord.xy);
  ivec2 size = textureSize(uSrc, 0);
  vec2 p = gl_FragCoord.xy;
  vec2 best = vec2(-1e4);
  float bd = 1e9;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      ivec2 q = ip + ivec2(x, y) * uStep;
      if (q.x < 0 || q.y < 0 || q.x >= size.x || q.y >= size.y) continue;
      vec2 c = texelFetch(uSrc, q, 0).xy;
      if (c.x < -1e3) continue;
      float d = distance(c, p);
      if (d < bd) { bd = d; best = c; }
    }
  }
  o = vec4(best, 0., 1.);
}`;

const JFA_DIST = HEAD + `
uniform sampler2D uSrc;
void main(){
  vec2 c = texelFetch(uSrc, ivec2(gl_FragCoord.xy), 0).xy;
  o = vec4(c.x < -1e3 ? 1e4 : distance(c, gl_FragCoord.xy), 0., 0., 1.);
}`;

// 段 i: プローブ間隔 2^i、方向数 4^(i+1)（2^(i+1) 四方のタイルに格納）、
// 区間 [r0(4^i-1)/3, r0(4^(i+1)-1)/3]。遮られなかった光線には上の段の 4 方向を双線形で足す
const CASCADE = HEAD + `
uniform sampler2D uScene;
uniform sampler2D uSdf;
uniform sampler2D uUpper;
uniform int uIndex;
uniform int uCount;
uniform vec2 uGrid;
uniform float uR0;
uniform vec3 uEnvA[ENV_ORDER + 1];
uniform vec3 uEnvB[ENV_ORDER + 1];
uniform float uEnvRot;

vec3 sky(float a){
  vec3 r = uEnvA[0];
  for (int n = 1; n <= ENV_ORDER; n++) {
    float t = float(n) * (a - uEnvRot);
    r += uEnvA[n] * cos(t) + uEnvB[n] * sin(t);
  }
  return max(r, 0.);
}

// rgb: 当たった面の放射輝度, a: 透過（1 なら何にも当たらず区間を抜けた）
vec4 march(vec2 o0, vec2 rd, float len){
  float t = 0.;
  for (int s = 0; s < ${MARCH_STEPS}; s++) {
    vec2 q = o0 + rd * t;
    if (q.x < 0. || q.y < 0. || q.x >= uGrid.x || q.y >= uGrid.y) return vec4(0., 0., 0., 1.);
    float d = texelFetch(uSdf, ivec2(q), 0).r;
    if (d < .75) return vec4(texelFetch(uScene, ivec2(q), 0).rgb, 0.);
    t += d - .5;
    if (t >= len) return vec4(0., 0., 0., 1.);
  }
  return vec4(0., 0., 0., 1.);
}

void main(){
  ivec2 tc = ivec2(gl_FragCoord.xy);
  int T = 2 << uIndex;
  int D = T * T;
  ivec2 probe = tc / T;
  ivec2 sub = tc - probe * T;
  int dir = sub.y * T + sub.x;
  float sp = float(1 << uIndex);
  vec2 origin = (vec2(probe) + .5) * sp;

  float ang = (float(dir) + .5) / float(D) * TAU;
  vec2 rd = vec2(cos(ang), sin(ang));
  float p4 = float(1 << (2 * uIndex));
  float start = uR0 * (p4 - 1.) / 3.;
  float len = uR0 * p4;
  vec2 a = origin + rd * start;

  if (uIndex == uCount - 1) {
    vec4 r = march(a, rd, len);
    vec3 up = vec3(0.);
    for (int k = 0; k < 4; k++) up += sky((float(dir * 4 + k) + .5) / float(D * 4) * TAU);
    o = vec4(r.rgb + r.a * up * .25, 1.);
    return;
  }

  // bilinear fix: 上の段の 4 プローブそれぞれへ向けて区間を張り直してから補間する。
  // 自分の区間端と上の段の区間始点がずれると、光漏れとブロック状のムラになる
  int Tu = T * 2;
  float spu = sp * 2.;
  ivec2 nUp = ivec2(uGrid / spu);
  vec2 g = origin / spu - .5;
  ivec2 g0 = ivec2(floor(g));
  vec2 f = g - vec2(g0);
  vec3 acc = vec3(0.);
  for (int c = 0; c < 4; c++) {
    ivec2 off = ivec2(c & 1, c >> 1);
    ivec2 pr = clamp(g0 + off, ivec2(0), nUp - 1);
    vec2 w2 = mix(1. - f, f, vec2(off));
    vec2 b = (vec2(pr) + .5) * spu + rd * (start + len);
    vec2 seg = b - a;
    float sl = length(seg);
    vec4 r = march(a, seg / max(sl, 1e-4), sl);
    vec3 s = vec3(0.);
    if (r.a > 0.) {
      for (int k = 0; k < 4; k++) {
        int cd = dir * 4 + k;
        s += texelFetch(uUpper, pr * Tu + ivec2(cd % Tu, cd / Tu), 0).rgb;
      }
    }
    acc += (r.rgb + r.a * s * .25) * w2.x * w2.y;
  }
  o = vec4(acc, 1.);
}`;

// 段 0 の 4 方向平均が照度。遮蔽物の画素は自分の中から光を見られないので、周囲の照度で埋める
const IRRADIANCE = HEAD + `
uniform sampler2D uC0;
uniform sampler2D uScene;
vec3 probeE(ivec2 p){
  ivec2 b = p * 2;
  return (texelFetch(uC0, b, 0).rgb + texelFetch(uC0, b + ivec2(1, 0), 0).rgb
        + texelFetch(uC0, b + ivec2(0, 1), 0).rgb + texelFetch(uC0, b + ivec2(1, 1), 0).rgb) * .25;
}
bool isOccluder(ivec2 p){
  vec4 s = texelFetch(uScene, p, 0);
  return s.a > .5 && dot(s.rgb, vec3(1.)) == 0.;
}
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  if (!isOccluder(p)) { o = vec4(probeE(p), 1.); return; }
  ivec2 size = textureSize(uScene, 0);
  vec3 acc = vec3(0.);
  float n = 0.;
  for (int y = -2; y <= 2; y++) {
    for (int x = -2; x <= 2; x++) {
      ivec2 q = p + ivec2(x, y);
      if (q.x < 0 || q.y < 0 || q.x >= size.x || q.y >= size.y || isOccluder(q)) continue;
      acc += probeE(q);
      n += 1.;
    }
  }
  o = vec4(n > 0. ? acc / n : vec3(0.), 1.);
}`;

// 低解像度の照度を、線をまたがない近傍テクセルだけで補間して表示解像度へ拡大する
const DISPLAY = HEAD + `
uniform sampler2D uIrr;
uniform sampler2D uScene;
uniform sampler2D uMask;
uniform vec2 uRes;
uniform vec2 uView;
uniform vec2 uGrid;
uniform float uExposure;
uniform float uLineAlpha;
uniform int uFrame;
uniform vec4 uLights[MAX_LIGHTS];
uniform vec3 uLightCols[MAX_LIGHTS];
uniform int uNumLights;

vec3 aces(vec3 x){ return clamp((x * (2.51 * x + .03)) / (x * (2.43 * x + .59) + .14), 0., 1.); }
uint pcg(uint v){ uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }

bool occluderTexel(ivec2 t){
  vec4 s = texelFetch(uScene, t, 0);
  return s.a > .5 && dot(s.rgb, vec3(1.)) == 0.;
}

void main(){
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 bt = uv * uView;
  float line = texture(uMask, uv).r;

  vec2 g = bt - .5;
  ivec2 g0 = ivec2(floor(g));
  vec2 f = g - vec2(g0);
  vec3 acc = vec3(0.);
  float wsum = 0.;
  for (int c = 0; c < 4; c++) {
    ivec2 off = ivec2(c & 1, c >> 1);
    ivec2 t = clamp(g0 + off, ivec2(0), ivec2(uGrid) - 1);
    vec2 w2 = mix(1. - f, f, vec2(off));
    float w = w2.x * w2.y + 1e-4;
    if (occluderTexel(t)) continue;
    vec2 tc = (vec2(t) + .5) / uView;
    bool blocked = false;
    for (int s = 1; s <= 4; s++) {
      if (texture(uMask, mix(uv, tc, float(s) / 5.)).r > .5) { blocked = true; break; }
    }
    if (blocked) continue;
    acc += texelFetch(uIrr, t, 0).rgb * w;
    wsum += w;
  }
  vec3 E = wsum > 0. ? acc / wsum : texture(uIrr, bt / uGrid).rgb;

  vec3 col = E * uExposure;
  col *= 1. - uLineAlpha * line;

  for (int l = 0; l < MAX_LIGHTS; l++) {
    if (l >= uNumLights) break;
    float d = length(bt - uLights[l].xy) / uView.y;
    col += uLightCols[l] * uLights[l].w * 2.5e-6 / (d * d + 5e-5);
  }

  col = pow(aces(col), vec3(1. / 2.2));
  uint s = pcg(uint(gl_FragCoord.x) * 7919u + uint(gl_FragCoord.y) * 104729u + uint(uFrame) * 31u);
  col += (float(s) / 4294967295. - .5) * (2. / 255.);
  o = vec4(col, 1.);
}`;

export function createCascadeRenderer(canvas) {
  const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
  if (!gl || !gl.getExtension('EXT_color_buffer_float')) throw new Error('WebGL2 + float render target が必要です');

  const SETTERS = {
    [gl.FLOAT]: 'uniform1fv', [gl.FLOAT_VEC2]: 'uniform2fv', [gl.FLOAT_VEC3]: 'uniform3fv', [gl.FLOAT_VEC4]: 'uniform4fv',
    [gl.INT]: 'uniform1iv', [gl.BOOL]: 'uniform1iv',
  };

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
    const u = {};
    for (let i = 0, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i < n; i++) {
      const info = gl.getActiveUniform(p, i);
      const name = info.name.replace(/\[0\]$/, '');
      u[name] = { loc: gl.getUniformLocation(p, name), setter: SETTERS[info.type], sampler: info.type === gl.SAMPLER_2D };
    }
    return { p, u };
  }

  const P = {
    maskSegs: program(MASK_SEGS),
    luma: program(LUMA),
    maskEdges: program(MASK_EDGES),
    scene: program(SCENE),
    jfaSeed: program(JFA_SEED),
    jfaStep: program(JFA_STEP),
    jfaDist: program(JFA_DIST),
    cascade: program(CASCADE),
    irradiance: program(IRRADIANCE),
    display: program(DISPLAY),
  };
  gl.bindVertexArray(gl.createVertexArray());

  const FMT = {
    rgba: [gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT],
    // 段テクスチャは帯域が支配的なので、アルファ無しの 32bit 形式にする
    rgb: [gl.R11F_G11F_B10F, gl.RGB, gl.HALF_FLOAT],
    r: [gl.R16F, gl.RED, gl.HALF_FLOAT],
    mask: [gl.R8, gl.RED, gl.UNSIGNED_BYTE],
  };
  function target(w, h, filter = gl.NEAREST, fmt = FMT.rgba) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, fmt[0], w, h, 0, fmt[1], fmt[2], null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
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

  const videoTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, videoTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  function run(prog, dst, uniforms) {
    gl.useProgram(prog.p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, dst ? dst.fb : null);
    gl.viewport(0, 0, dst ? dst.w : canvas.width, dst ? dst.h : canvas.height);
    let unit = 0;
    for (const [name, val] of Object.entries(uniforms)) {
      const u = prog.u[name];
      if (!u) continue;
      if (u.sampler) {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, val.tex ?? val);
        gl.uniform1i(u.loc, unit++);
      } else {
        gl[u.setter](u.loc, typeof val === 'number' ? [val] : val);
      }
    }
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // view: 画面に対応するグリッドの範囲（テクセル）。grid: 2^段数 の倍数へ切り上げた全体。
  // mask: 線の表示解像度マスク（CSS ピクセル）
  let view = [0, 0], grid = [0, 0], mask = [0, 0], count = 1, aspect = 1;
  let T = {};

  function resize(cssW, cssH, scale) {
    canvas.width = Math.floor(cssW * Math.min(devicePixelRatio || 1, 2));
    canvas.height = Math.floor(cssH * Math.min(devicePixelRatio || 1, 2));
    aspect = cssW / cssH;
    view = [Math.max(16, Math.round(cssW * scale)), Math.max(16, Math.round(cssH * scale))];
    mask = [Math.round(cssW), Math.round(cssH)];
    // 最上段の区間の終端が対角線を超える段数
    const diag = Math.hypot(view[0], view[1]);
    count = Math.min(MAX_CASCADES, Math.max(2, Math.ceil(Math.log(diag * 3 + 1) / Math.log(4))));
    const m = 1 << count;
    grid = [Math.ceil(view[0] / m) * m, Math.ceil(view[1] / m) * m];
    Object.values(T).forEach(free);
    const [gw, gh] = grid;
    T = {
      mask: target(mask[0], mask[1], gl.LINEAR, FMT.mask),
      lumaA: target(mask[0], mask[1], gl.NEAREST, FMT.r), lumaB: target(mask[0], mask[1], gl.NEAREST, FMT.r),
      scene: target(gw, gh),
      jfaA: target(gw, gh), jfaB: target(gw, gh),
      sdf: target(gw, gh, gl.NEAREST, FMT.r),
      casA: target(gw * 2, gh * 2, gl.NEAREST, FMT.rgb), casB: target(gw * 2, gh * 2, gl.NEAREST, FMT.rgb),
      irr: target(gw, gh, gl.LINEAR, FMT.rgb),
    };
  }

  // 画面座標（高さ 1、中心原点、y 上向き）→ グリッド座標
  const toGridX = (x) => (x / aspect + 0.5) * view[0];
  const toGridY = (y) => (y + 0.5) * view[1];

  const segData = new Float32Array(MAX_SEGS * 4);
  const lightData = new Float32Array(MAX_LIGHTS * 4);
  const lightColData = new Float32Array(MAX_LIGHTS * 3);

  function draw(f) {
    const segs = f.segs.slice(0, MAX_SEGS);
    const u = (x) => x / aspect + 0.5, v = (y) => y + 0.5;
    segs.forEach((s, i) => segData.set([u(s.ax), v(s.ay), u(s.bx), v(s.by)], i * 4));
    const lights = f.lights.slice(0, MAX_LIGHTS);
    lights.forEach((l, i) => {
      lightData.set([toGridX(l.x), toGridY(l.y), Math.max(1.5, l.r * view[1]), l.I], i * 4);
      lightColData.set(l.col, i * 3);
    });
    const lightU = { uLights: lightData, uLightCols: lightColData, uNumLights: lights.length };

    if (f.video) {
      const v = f.video;
      gl.bindTexture(gl.TEXTURE_2D, videoTex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, v);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      const va = v.videoWidth / v.videoHeight, a = mask[0] / mask[1];
      const cover = a > va ? [1, va / a] : [a / va, 1];
      run(P.luma, T.lumaB, { uVideo: videoTex, uPrev: T.lumaA, uMaskSize: mask, uCover: cover, uBlend: 0.5 });
      [T.lumaA, T.lumaB] = [T.lumaB, T.lumaA];
      run(P.maskEdges, T.mask, { uLuma: T.lumaA, uThresh: f.edgeThresh });
    } else {
      run(P.maskSegs, T.mask, { uSegs: segData, uNumSegs: segs.length, uMaskSize: mask, uLineWidth: 0.6 });
    }
    run(P.scene, T.scene, { ...lightU, uMask: T.mask, uView: view });

    run(P.jfaSeed, T.jfaA, { uScene: T.scene });
    for (let step = 1 << Math.floor(Math.log2(Math.max(...grid))); step >= 1; step >>= 1) {
      run(P.jfaStep, T.jfaB, { uSrc: T.jfaA, uStep: step });
      [T.jfaA, T.jfaB] = [T.jfaB, T.jfaA];
    }
    run(P.jfaDist, T.sdf, { uSrc: T.jfaA });

    for (let i = count - 1; i >= 0; i--) {
      run(P.cascade, T.casB, {
        uScene: T.scene, uSdf: T.sdf, uUpper: T.casA,
        uIndex: i, uCount: count, uGrid: grid, uR0: 1,
        uEnvA: f.env.A, uEnvB: f.env.B, uEnvRot: f.envRot,
      });
      [T.casA, T.casB] = [T.casB, T.casA];
    }
    run(P.irradiance, T.irr, { uC0: T.casA, uScene: T.scene });

    run(P.display, null, {
      ...lightU, uIrr: T.irr, uScene: T.scene, uMask: T.mask,
      uRes: [canvas.width, canvas.height], uView: view, uGrid: grid,
      uExposure: 0.7, uLineAlpha: 0.25, uFrame: f.frame,
    });
  }

  return { resize, draw, get view() { return view; } };
}
