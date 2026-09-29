import { createRenderer } from './renderer.js';
import { randomScene, segsAt, lightsAt } from './scene.js';
import { Camera } from './camera.js';

const canvas = document.getElementById('c');
const camButton = document.getElementById('cam');
const preview = document.getElementById('preview');
const params = new URLSearchParams(location.search);

let renderer;
try {
  renderer = createRenderer(canvas);
} catch (e) {
  document.body.innerHTML = `<p style="color:#aaa;font:14px sans-serif;padding:2em">${e.message}</p>`;
  throw e;
}

// GI グリッドの解像度（CSS ピクセル比）。?res= で固定しない限り、フレーム時間に合わせて自動で上下させる
const FIXED_SCALE = Number(params.get('res')) || 0;
const SCALE_MIN = 0.3, SCALE_MAX = 0.6;
let scale = FIXED_SCALE || 0.5;
const aspect = () => innerWidth / innerHeight;
const resize = () => renderer.resize(innerWidth, innerHeight, scale);

const camera = new Camera(preview);
let edgeThresh = Number(params.get('edge')) || 0.35;
let scene;
const newSeed = () => (Math.random() * 2 ** 31) | 0;
// カメラモードは線分を持たない専用シーン（光源と環境光だけ）を作り、遮蔽物はカメラのエッジだけにする
function newScene(seed = newSeed()) {
  scene = randomScene(seed, aspect(), { withSegs: !camera.on });
}

const mouse = { x: 0, y: 0, on: false };
let paused = false;
let frame = 0;
let time = 0;
let last = performance.now();
let slowFrames = 0, fastFrames = 0;

function adapt(dt) {
  if (FIXED_SCALE) return;
  if (dt > 1 / 30) { slowFrames++; fastFrames = 0; }
  else if (dt < 1 / 55) { fastFrames++; slowFrames = 0; }
  if (slowFrames > 30 && scale > SCALE_MIN) { scale = Math.max(SCALE_MIN, scale * 0.85); slowFrames = 0; resize(); }
  if (fastFrames > 120 && scale < SCALE_MAX) { scale = Math.min(SCALE_MAX, scale * 1.1); fastFrames = 0; resize(); }
}

function render(now) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  if (!paused) time += dt;
  adapt(dt);

  // 線分の有無はシーン側が持つ（カメラ用シーンは線分を持たない）。カメラ起動とシーン切替の間のフレームでも整合する
  const segs = segsAt(scene, time);
  const lights = lightsAt(scene, segs, time);
  if (mouse.on) lights.push({ x: mouse.x, y: mouse.y, r: 0.008, I: 22, col: [1, 0.95, 0.9] });

  renderer.draw({
    segs, lights,
    video: scene.segs.length === 0 && camera.ready ? camera.video : null,
    edgeThresh,
    env: scene.env, envRot: scene.rot0 + time * 0.03,
    frame,
  });

  frame++;
  requestAnimationFrame(render);
}

async function toggleCamera() {
  if (camera.on) {
    camera.stop();
  } else {
    try {
      await camera.start();
    } catch (e) {
      console.error(e);
      alert(`カメラを開けませんでした: ${e.message}`);
      return;
    }
  }
  camButton.classList.toggle('on', camera.on);
  preview.hidden = !camera.on;
  newScene();
}

canvas.addEventListener('pointermove', (e) => {
  if (e.pointerType !== 'mouse') return;
  mouse.x = (e.clientX / innerWidth - 0.5) * aspect();
  mouse.y = 0.5 - e.clientY / innerHeight;
  mouse.on = true;
});
canvas.addEventListener('pointerleave', () => { mouse.on = false; });
canvas.addEventListener('click', () => newScene());
camButton.addEventListener('click', toggleCamera);
addEventListener('keydown', (e) => {
  if (e.code === 'Space') { paused = !paused; e.preventDefault(); }
  if (e.code === 'KeyC') toggleCamera();
  if (e.code === 'BracketLeft') edgeThresh = Math.max(0.05, edgeThresh * 0.85);
  if (e.code === 'BracketRight') edgeThresh = Math.min(3, edgeThresh / 0.85);
  if (e.code === 'KeyS') {
    const a = document.createElement('a');
    a.download = `gi-${Date.now()}.png`;
    a.href = canvas.toDataURL('image/png');
    a.click();
  }
});
addEventListener('resize', resize);

resize();
newScene(params.get('seed') ? Number(params.get('seed')) : undefined);
if (params.has('camera')) toggleCamera();
requestAnimationFrame(render);
