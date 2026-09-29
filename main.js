import { createCascadeRenderer } from './renderer.js';
import { createAnalyticRenderer } from './analytic.js';
import { randomScene, segsAt, lightsAt } from './scene.js';
import { Camera } from './camera.js';

const camButton = document.getElementById('cam');
const preview = document.getElementById('preview');
const params = new URLSearchParams(location.search);

// 線分シーンは表示解像度で厳密に解く解析版、カメラのエッジは遮蔽物数に依らない Radiance Cascades で描く。
// 1 つの canvas に 2 つの WebGL コンテキストは持てないので、canvas を分けて表示を切り替える。
// scale はフレーム時間に合わせて [min, max] で自動調整する（?res= で固定）
const FIXED_SCALE = Number(params.get('res')) || 0;
const modes = {};
try {
  modes.analytic = { canvas: document.getElementById('c'), scale: 1, min: 0.6, max: 1 };
  modes.analytic.renderer = createAnalyticRenderer(modes.analytic.canvas);
  modes.cascade = { canvas: document.getElementById('c2'), scale: 0.5, min: 0.35, max: 0.6 };
  modes.cascade.renderer = createCascadeRenderer(modes.cascade.canvas);
} catch (e) {
  document.body.innerHTML = `<p style="color:#aaa;font:14px sans-serif;padding:2em">${e.message}</p>`;
  throw e;
}
if (FIXED_SCALE) Object.values(modes).forEach((m) => { m.scale = FIXED_SCALE; });

const aspect = () => innerWidth / innerHeight;
const resize = (m) => m.renderer.resize(innerWidth, innerHeight, m.scale);

const camera = new Camera(preview);
let edgeThresh = Number(params.get('edge')) || 0.35;
let scene;
// カメラモードは線分を持たない専用シーン（光源と環境光だけ）で、遮蔽物はカメラのエッジだけ。
// カメラの許可待ち・起動中もこのシーンを出すため、モードはカメラの状態と別に持つ
let cameraMode = false;
const active = () => (cameraMode ? modes.cascade : modes.analytic);
const newSeed = () => (Math.random() * 2 ** 31) | 0;
function newScene(seed = newSeed()) {
  scene = randomScene(seed, aspect(), { withSegs: !cameraMode });
}

const mouse = { x: 0, y: 0, on: false };
let paused = false;
let frame = 0;
let time = 0;
let last = performance.now();
let slowFrames = 0, fastFrames = 0;

function adapt(m, dt) {
  if (FIXED_SCALE) return;
  if (dt > 1 / 30) { slowFrames++; fastFrames = 0; }
  else if (dt < 1 / 55) { fastFrames++; slowFrames = 0; }
  if (slowFrames > 30 && m.scale > m.min) { m.scale = Math.max(m.min, m.scale * 0.85); slowFrames = 0; resize(m); }
  if (fastFrames > 120 && m.scale < m.max) { m.scale = Math.min(m.max, m.scale * 1.1); fastFrames = 0; resize(m); }
}

function render(now) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  if (!paused) time += dt;
  const m = active();
  adapt(m, dt);

  const segs = segsAt(scene, time);
  const lights = lightsAt(scene, segs, time);
  if (mouse.on) lights.push({ x: mouse.x, y: mouse.y, r: 0.008, I: 22, col: [1, 0.95, 0.9] });

  m.renderer.draw({
    segs, lights,
    video: cameraMode && camera.ready ? camera.video : null,
    edgeThresh,
    env: scene.env, envRot: scene.rot0 + time * 0.03,
    frame,
  });

  frame++;
  requestAnimationFrame(render);
}

function setCameraMode(on) {
  cameraMode = on;
  modes.analytic.canvas.hidden = on;
  modes.cascade.canvas.hidden = !on;
  camButton.classList.toggle('on', on);
  preview.hidden = !on;
  slowFrames = fastFrames = 0;
  newScene();
}

async function toggleCamera() {
  if (cameraMode) {
    camera.stop();
    setCameraMode(false);
    return;
  }
  setCameraMode(true);
  try {
    await camera.start();
  } catch (e) {
    console.error(e);
    camera.stop();
    setCameraMode(false);
    alert(`カメラを開けませんでした: ${e.message}`);
  }
}

for (const m of Object.values(modes)) {
  m.canvas.addEventListener('pointermove', (e) => {
    if (e.pointerType !== 'mouse') return;
    mouse.x = (e.clientX / innerWidth - 0.5) * aspect();
    mouse.y = 0.5 - e.clientY / innerHeight;
    mouse.on = true;
  });
  m.canvas.addEventListener('pointerleave', () => { mouse.on = false; });
  m.canvas.addEventListener('click', () => newScene());
}
camButton.addEventListener('click', toggleCamera);
addEventListener('keydown', (e) => {
  if (e.code === 'Space') { paused = !paused; e.preventDefault(); }
  if (e.code === 'KeyC') toggleCamera();
  if (e.code === 'BracketLeft') edgeThresh = Math.max(0.05, edgeThresh * 0.85);
  if (e.code === 'BracketRight') edgeThresh = Math.min(3, edgeThresh / 0.85);
  if (e.code === 'KeyS') {
    const a = document.createElement('a');
    a.download = `gi-${Date.now()}.png`;
    a.href = active().canvas.toDataURL('image/png');
    a.click();
  }
});
addEventListener('resize', () => Object.values(modes).forEach(resize));

Object.values(modes).forEach(resize);
newScene(params.get('seed') ? Number(params.get('seed')) : undefined);
if (params.has('camera')) toggleCamera();
requestAnimationFrame(render);
