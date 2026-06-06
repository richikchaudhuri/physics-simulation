import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { initWasm, World, makePositions } from './wasm.js';

await initWasm();

// --- simulation -----------------------------------------------------------
const BODY_COUNT = 200;
const world = new World(0, BODY_COUNT);
const positions = makePositions(world);
const radius = world.radius();
const bounds = world.bounds();
const count = world.count();

// --- renderer / scene -----------------------------------------------------
const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
  55,
  window.innerWidth / window.innerHeight,
  0.1,
  1000,
);
camera.position.set(11, 9, 15);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, bounds * 0.5, 0);

// --- lights ---------------------------------------------------------------
scene.add(new THREE.HemisphereLight(0xffffff, 0x202028, 0.7));
const key = new THREE.DirectionalLight(0xffffff, 1.3);
key.position.set(8, 16, 6);
scene.add(key);

// --- environment: ground grid + open-top box ------------------------------
const grid = new THREE.GridHelper(bounds * 2, 20, 0x444444, 0x222222);
scene.add(grid);

const boxGeo = new THREE.BoxGeometry(bounds * 2, bounds * 2, bounds * 2);
const box = new THREE.LineSegments(
  new THREE.EdgesGeometry(boxGeo),
  new THREE.LineBasicMaterial({ color: 0x333333 }),
);
box.position.y = bounds; // rests on the grid (floor at y = 0)
scene.add(box);

// --- instanced spheres ----------------------------------------------------
const sphereGeo = new THREE.SphereGeometry(radius, 18, 14);
const sphereMat = new THREE.MeshStandardMaterial({
  color: 0xe8e8e8,
  roughness: 0.4,
  metalness: 0.1,
});
const mesh = new THREE.InstancedMesh(sphereGeo, sphereMat, count);
mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
scene.add(mesh);

const dummy = new THREE.Object3D();

// --- ui -------------------------------------------------------------------
const statCount = document.getElementById('stat-count');
const statFps = document.getElementById('stat-fps');
statCount.textContent = count;
document.getElementById('btn-reset').addEventListener('click', () => world.reset());

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- render + fixed-timestep loop -----------------------------------------
const FIXED = 1 / 120;
let last = performance.now();
let acc = 0;
let fpsTime = 0;
let fpsFrames = 0;

function render() {
  const p = positions(); // re-fetched AFTER stepping (step may reallocate)
  for (let i = 0; i < count; i++) {
    dummy.position.set(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
  }
  mesh.instanceMatrix.needsUpdate = true;
  controls.update();
  renderer.render(scene, camera);
}

function frame(now) {
  let dt = (now - last) / 1000;
  last = now;
  if (dt > 0.1) dt = 0.1; // clamp after tab-switch / stalls
  acc += dt;
  while (acc >= FIXED) {
    world.step(FIXED);
    acc -= FIXED;
  }
  render();

  fpsFrames++;
  fpsTime += dt;
  if (fpsTime >= 0.5) {
    statFps.textContent = Math.round(fpsFrames / fpsTime);
    fpsFrames = 0;
    fpsTime = 0;
  }

  schedule();
}

// Prefer requestAnimationFrame (vsync-smooth) when the tab is visible. Some
// embedded/preview webviews report the tab as hidden, which pauses rAF entirely;
// fall back to setTimeout so the scene still animates there.
function schedule() {
  if (document.hidden) {
    setTimeout(() => frame(performance.now()), 1000 / 60);
  } else {
    requestAnimationFrame(frame);
  }
}

render(); // paint the initial frame right away, before the loop ticks
schedule();

document.addEventListener('visibilitychange', () => {
  last = performance.now(); // avoid a huge dt spike when visibility flips
});

// Dev-only: headless previews keep the tab hidden, which pauses rAF. This hook
// lets the harness manually advance the sim and draw a frame for screenshots.
if (import.meta.env.DEV) {
  window.__sandbox = {
    tick(steps = 60) {
      for (let i = 0; i < steps; i++) world.step(FIXED);
      render();
    },
    render,
    sample() {
      const p = positions();
      let minY = Infinity;
      let maxY = -Infinity;
      let maxAbsXZ = 0;
      let minPairDist = Infinity;
      for (let i = 0; i < count; i++) {
        const y = p[i * 3 + 1];
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        maxAbsXZ = Math.max(maxAbsXZ, Math.abs(p[i * 3]), Math.abs(p[i * 3 + 2]));
      }
      for (let i = 0; i < count; i++) {
        for (let j = i + 1; j < count; j++) {
          const dx = p[i * 3] - p[j * 3];
          const dy = p[i * 3 + 1] - p[j * 3 + 1];
          const dz = p[i * 3 + 2] - p[j * 3 + 2];
          const d = Math.hypot(dx, dy, dz);
          if (d < minPairDist) minPairDist = d;
        }
      }
      return { count, radius, bounds, minY, maxY, maxAbsXZ, minPairDist, first: [p[0], p[1], p[2]] };
    },
  };
}
