import { createRenderer } from './renderer.js';
import { randomScene, segsAt, lightsAt } from './scene.js';

const canvas = document.getElementById('c');
const hint = document.getElementById('hint');
const hintText = () => `click: new scene / g: GI ${gi ? 'on' : 'off'} / space: pause / s: save`;
const params = new URLSearchParams(location.search);

let renderer;
try {
  renderer = createRenderer(canvas);
} catch (e) {
  document.body.innerHTML = `<p style="color:#aaa;font:14px sans-serif;padding:2em">${e.message}</p>`;
  throw e;
}

// 描画解像度（表示解像度比）。?res= で固定しない限り、フレーム時間に合わせて自動で上下させる
const FIXED_SCALE = Number(params.get('res')) || 0;
const SCALE_MIN = 0.6, SCALE_MAX = 1;
let scale = FIXED_SCALE || 1;
const aspect = () => innerWidth / innerHeight;
const resize = () => renderer.resize(innerWidth, innerHeight, scale);

// GI は見た目の差が小さい割に重いので既定で切る（G キーか ?gi=1 で有効）
let gi = params.get('gi') === '1';
let scene;
const newScene = (seed = (Math.random() * 2 ** 31) | 0) => { scene = randomScene(seed, aspect()); };

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

  const segs = segsAt(scene, time);
  const lights = lightsAt(scene, segs, time);
  if (mouse.on) lights.push({ x: mouse.x, y: mouse.y, r: 0.008, I: 22, col: [1, 0.95, 0.9] });

  renderer.draw({
    segs, lights,
    env: scene.env, envRot: scene.rot0 + time * 0.03,
    gi, frame,
  });

  frame++;
  requestAnimationFrame(render);
}

canvas.addEventListener('pointermove', (e) => {
  if (e.pointerType !== 'mouse') return;
  mouse.x = (e.clientX / innerWidth - 0.5) * aspect();
  mouse.y = 0.5 - e.clientY / innerHeight;
  mouse.on = true;
});
canvas.addEventListener('pointerleave', () => { mouse.on = false; });
canvas.addEventListener('click', () => newScene());
addEventListener('keydown', (e) => {
  if (e.code === 'Space') { paused = !paused; e.preventDefault(); }
  if (e.code === 'KeyG') { gi = !gi; hint.textContent = hintText(); }
  if (e.code === 'KeyS') {
    const a = document.createElement('a');
    a.download = `gi-${Date.now()}.png`;
    a.href = canvas.toDataURL('image/png');
    a.click();
  }
});
addEventListener('resize', resize);

resize();
hint.textContent = hintText();
newScene(params.get('seed') ? Number(params.get('seed')) : undefined);
requestAnimationFrame(render);
