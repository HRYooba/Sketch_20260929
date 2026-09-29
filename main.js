import { createRenderer, MAX_LIGHTS, MAX_SEGS } from './renderer.js';
import { randomScene, segsAt, lightsAt } from './scene.js';
import { EdgeCamera } from './camera.js';

const canvas = document.getElementById('c');
const camButton = document.getElementById('cam');
const params = new URLSearchParams(location.search);
const RES_SCALE = Number(params.get('res')) || 1;

let renderer;
try {
  renderer = createRenderer(canvas);
} catch (e) {
  document.body.innerHTML = '<p style="color:#aaa;font:14px sans-serif;padding:2em">WebGL2 が必要です</p>';
  throw e;
}

function resize() {
  // 画素あたりの計算が重いので DPR は 1.5 で頭打ちにする
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5) * RES_SCALE;
  canvas.width = Math.floor(innerWidth * dpr);
  canvas.height = Math.floor(innerHeight * dpr);
}
const aspect = () => innerWidth / innerHeight;

const camera = new EdgeCamera();
let camSegs = [];
let scene;
const newSeed = () => (Math.random() * 2 ** 31) | 0;
function newScene(seed = newSeed()) {
  scene = randomScene(seed, aspect(), { withSegs: !camera.on });
}

const mouse = { x: 0, y: 0, on: false };
let paused = false;
let frame = 0;
let time = 0;
let last = performance.now();

const segData = new Float32Array(MAX_SEGS * 4);
const lightData = new Float32Array(MAX_LIGHTS * 4);
const lightColData = new Float32Array(MAX_LIGHTS * 3);

function render(now) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  if (!paused) time += dt;

  if (camera.on && frame % 2 === 0) camSegs = camera.detect(aspect(), MAX_SEGS) ?? camSegs;
  const segs = camera.on ? camSegs : segsAt(scene, time);
  const lights = lightsAt(scene, segs, time);
  if (mouse.on) lights.push({ x: mouse.x, y: mouse.y, r: 0.008, I: 22, col: [1, 0.95, 0.9] });

  segs.forEach((s, i) => segData.set([s.ax, s.ay, s.bx, s.by], i * 4));
  lights.forEach((l, i) => {
    lightData.set([l.x, l.y, l.r, l.I], i * 4);
    lightColData.set(l.col, i * 3);
  });

  renderer.draw({
    segs: segData, numSegs: segs.length,
    lights: lightData, lightCols: lightColData, numLights: lights.length,
    env: scene.env, envRot: scene.rot0 + time * 0.03,
    frame,
  });

  frame++;
  requestAnimationFrame(render);
}

async function toggleCamera() {
  if (camera.on) {
    camera.stop();
    camSegs = [];
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
