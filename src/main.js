import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { initWasm, World, makePositions, makeExtra, makeGrid } from './wasm.js';

await initWasm();

// --- sim registry ---------------------------------------------------------
// Each entry describes how a Rust sim (by `kind`) is presented: body-count
// range, environment (box/grid), camera framing, and rendering (per-instance
// scale, speed coloring). The SceneManager (loadSim/buildWorld below) reads
// these to (re)configure the scene without per-sim branching elsewhere.
const SIMS = {
  collisions: {
    kind: 0,
    accent: '#e0a155', // warm amber — energetic bounces
    defaultCount: 200,
    minCount: 2,
    maxCount: 2000, // uniform-grid broad-phase keeps high counts smooth
    boxed: true, // walls + floor box, camera framed on the box
    showGrid: true,
    colorBySpeed: true, // tint balls by speed (hot = fast)
    speedScale: 12,
    centralScale: 1,
    hasGravityParam: true,
    gravLabel: 'Gravity',
    gravMin: 0,
    gravMax: 2, // multiplier on Earth gravity; 0 = weightless gas
    gravStep: 0.1,
    gravDefault: 1,
    grabbable: true,
    rods: false,
    camPos: new THREE.Vector3(11, 9, 15),
    hint: 'Drag a ball to grab & throw · left-drag to orbit · scroll to zoom',
  },
  gravity: {
    kind: 1,
    accent: '#9d8cff', // cosmic violet
    defaultCount: 500,
    minCount: 50,
    maxCount: 1500,
    boxed: false, // open space centered on the origin
    showGrid: false,
    colorBySpeed: true,
    speedScale: 16,
    centralScale: 6, // draw the heavy central body (index 0) larger
    hasGravityParam: true,
    gravLabel: 'Gravity G',
    gravMin: 0,
    gravMax: 3,
    gravStep: 0.1,
    gravDefault: 1,
    grabbable: true,
    rods: false,
    camPos: new THREE.Vector3(0, 15, 22),
    hint: 'Drag a star to grab & throw · left-drag to orbit · scroll to zoom',
  },
  pendulum: {
    kind: 2,
    accent: '#57c8c2', // teal
    defaultCount: 24, // number of pendulums (bobs = 2x this)
    minCount: 1,
    maxCount: 200,
    boxed: false,
    showGrid: false,
    colorBySpeed: false,
    colorByIndex: true, // rainbow by pendulum, so chaotic divergence is visible
    speedScale: 1,
    centralScale: 1,
    rods: true, // draw the two arms of each pendulum
    grabbable: false, // bobs are angle-constrained; left-drag always orbits
    hasGravityParam: true,
    gravLabel: 'Gravity g',
    gravMin: 0,
    gravMax: 30,
    gravStep: 0.5,
    gravDefault: 9.81,
    camPos: new THREE.Vector3(0, -0.5, 8),
    camTarget: new THREE.Vector3(0, -1, 0),
    hint: 'Identical pendulums, tiny angle offsets — watch chaos diverge · left-drag to orbit',
  },
  cloth: {
    kind: 3,
    accent: '#6ca0ff', // blue — matches the sheet
    // Count here is the grid resolution R (Rust builds an R×R sheet); the slider
    // shows it as "Resolution" and the renderer derives R = sqrt(count).
    defaultCount: 30,
    minCount: 8,
    maxCount: 60,
    countName: 'Resolution',
    boxed: false,
    showGrid: false,
    bobs: false, // the sheet is a deformable mesh, not instanced spheres
    rods: false,
    cloth: true,
    grabbable: false,
    colorBySpeed: false,
    centralScale: 1,
    speedScale: 1,
    hasGravityParam: true,
    gravLabel: 'Gravity g',
    gravMin: 0,
    gravMax: 20,
    gravStep: 0.5,
    gravDefault: 9.81,
    hasWind: true,
    windLabel: 'Wind',
    windMin: 0,
    windMax: 30,
    windStep: 0.5,
    windDefault: 6,
    hasPin: true,
    camPos: new THREE.Vector3(5, 1, 11),
    camTarget: new THREE.Vector3(0, -0.3, 0),
    hint: 'A pinned mass-spring cloth billowing in the wind · toggle Pin · left-drag to orbit',
  },
  descent: {
    kind: 4,
    accent: '#7ad1a5', // mint green — the "converged" signal color
    contour: true, // paints a top-down loss-landscape heatmap under the walkers
    colorByLoss: true, // walker dots brighten as their loss drops
    // "Count" is the number of optimizer walkers racing down the surface.
    defaultCount: 12,
    minCount: 1,
    maxCount: 64,
    countName: 'Walkers',
    boxed: false,
    showGrid: false,
    grabbable: false, // walkers are driven by the optimizer; left-drag orbits
    rods: false,
    cloth: false,
    colorBySpeed: false,
    centralScale: 1,
    speedScale: 1,
    // The learning rate rides the shared "gravity" slider (set_param id 0). A
    // per-sim decimals override keeps small rates (0.05) from rounding to "0.1".
    hasGravityParam: true,
    gravLabel: 'Learning rate',
    gravMin: 0.001,
    gravMax: 0.3,
    gravStep: 0.001,
    gravDefault: 0.05,
    gravDecimals: 3,
    // Landscape + optimizer pickers (filled into the .select navs) and the
    // momentum/Adam β slider.
    landscapes: ['Bowl', 'Saddle', 'Rosenbrock', 'Himmelblau', 'Rastrigin'],
    optimizers: ['SGD', 'Momentum', 'RMSProp', 'Adam'],
    defaultLandscape: 0, // Bowl
    defaultOptimizer: 3, // Adam
    hasBeta: true,
    betaDefault: 0.9,
    betaMin: 0,
    betaMax: 0.999,
    betaStep: 0.001,
    // Straight top-down; the tiny x/z offset avoids an OrbitControls gimbal flip.
    camPos: new THREE.Vector3(0.001, 14, 0.001),
    camTarget: new THREE.Vector3(0, 0, 0),
    hint: 'Optimizers race down a loss landscape · pick a function & optimizer · left-drag to orbit',
  },
  boids: {
    kind: 5,
    accent: '#e8718d', // rose — a lively flock against the cool sims
    defaultCount: 180,
    minCount: 30,
    maxCount: 400, // naive O(n^2) neighbour search stays smooth to a few hundred
    countName: 'Boids',
    boxed: false, // open 3D space; boids softly turn back from invisible bounds
    showGrid: false,
    colorBySpeed: true, // tint by speed (warm = fast)
    speedScale: 2.8, // boids settle into a tight cruise band — compress so the ramp still spreads
    centralScale: 1,
    grabbable: false, // boids are autonomous; left-drag always orbits
    rods: false,
    // The shared "gravity" slider drives cohesion strength (set_param id 0): how
    // hard boids steer toward their local flockmates' centre of mass.
    hasGravityParam: true,
    gravLabel: 'Cohesion',
    gravMin: 0,
    gravMax: 3,
    gravStep: 0.1,
    gravDefault: 1,
    camPos: new THREE.Vector3(13, 9, 18),
    hint: 'Emergent flocking from three local rules — separation, alignment, cohesion · left-drag to orbit',
  },
};

let simKey = 'collisions';
let cfg = SIMS[simKey];

// --- simulation state (rebuilt on sim / count change) ---------------------
let world = null;
let positions = null;
let extra = null;
let lossGrid = null; // loss-landscape heatmap view (gradient-descent sim only)
let simRadius = 0.4;
let simBounds = 4;
let count = 0;
let bodyCount = cfg.defaultCount;

// --- renderer / scene -----------------------------------------------------
const canvas = document.getElementById('scene');
// preserveDrawingBuffer keeps the rendered frame readable by canvas.toBlob for
// the screenshot feature (S key), at a negligible cost for this scene.
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
  55,
  window.innerWidth / window.innerHeight,
  0.1,
  1000,
);
camera.position.copy(cfg.camPos);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

// --- lights ---------------------------------------------------------------
scene.add(new THREE.HemisphereLight(0xffffff, 0x202028, 0.7));
const key = new THREE.DirectionalLight(0xffffff, 1.3);
key.position.set(8, 16, 6);
scene.add(key);

// --- environment: ground grid + box (sized to the boxed sim) --------------
const ENV_BOUNDS = 4; // collisions bounds; box/grid only show for that sim
const grid = new THREE.GridHelper(ENV_BOUNDS * 2, 20, 0x2a2a30, 0x161619);
scene.add(grid);

const boxGeo = new THREE.BoxGeometry(ENV_BOUNDS * 2, ENV_BOUNDS * 2, ENV_BOUNDS * 2);
const box = new THREE.LineSegments(
  new THREE.EdgesGeometry(boxGeo),
  new THREE.LineBasicMaterial({ color: 0x2a2a30 }),
);
box.position.y = ENV_BOUNDS; // rests on the grid (floor at y = 0)
scene.add(box);

// --- instanced spheres (unit sphere, scaled per instance) -----------------
const sphereGeo = new THREE.SphereGeometry(1, 16, 12);
const sphereMat = new THREE.MeshStandardMaterial({
  color: 0xffffff, // actual tint comes from per-instance colors
  roughness: 0.4,
  metalness: 0.1,
});
const BASE_COLOR = new THREE.Color(0xe8e8e8);
const HELD_COLOR = new THREE.Color(0x6ca0ff);
const tmpColor = new THREE.Color();
const dummy = new THREE.Object3D();
let mesh = null;

// Rods for the pendulum sim: one THREE.LineSegments holding both arms of every
// pendulum (pivot->bob1, bob1->bob2). Per-vertex colored to match its bobs;
// positions are rewritten each frame from the WASM bob coords.
const rodMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 });
let rodGeo = null;
let rodPos = null; // Float32Array backing rodGeo's position attribute
let rodLines = null;

// Cloth (sim #3): a single deformable triangle mesh whose vertices ARE the WASM
// particle positions. We copy the position buffer in each frame and recompute
// normals so lighting follows the billowing surface. DoubleSide so the back of
// the sheet is lit too when the wind flips it toward the camera.
const clothMat = new THREE.MeshStandardMaterial({
  color: 0x6ca0ff,
  roughness: 0.65,
  metalness: 0.0,
  side: THREE.DoubleSide,
});
let clothGeo = null;
let clothPos = null; // Float32Array backing clothGeo's position attribute
let clothMesh = null;

// Hue ramp by pendulum index (0..n) -> a spread-out rainbow.
function indexToColor(p, total, out) {
  return out.setHSL((p / Math.max(total, 1)) * 0.92, 0.7, 0.56);
}

// (Re)build the pendulum rod geometry for the current pendulum count. Colors are
// fixed here (per pendulum); only positions change per frame.
function buildRods() {
  if (rodLines) {
    scene.remove(rodLines);
    rodGeo.dispose();
    rodLines = null;
  }
  const pend = count / 2; // two bobs per pendulum
  const nVerts = pend * 4; // 2 segments * 2 endpoints
  rodPos = new Float32Array(nVerts * 3);
  const colArr = new Float32Array(nVerts * 3);
  for (let pi = 0; pi < pend; pi++) {
    indexToColor(pi, pend, tmpColor);
    for (let v = 0; v < 4; v++) {
      const c = (pi * 4 + v) * 3;
      colArr[c] = tmpColor.r;
      colArr[c + 1] = tmpColor.g;
      colArr[c + 2] = tmpColor.b;
    }
  }
  rodGeo = new THREE.BufferGeometry();
  rodGeo.setAttribute('position', new THREE.BufferAttribute(rodPos, 3));
  rodGeo.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
  rodLines = new THREE.LineSegments(rodGeo, rodMat);
  rodLines.frustumCulled = false; // bobs can swing past the initial bounds
  scene.add(rodLines);
}

// Rewrite rod endpoints from the bob positions `p` (shared pivot at the origin).
function updateRods(p) {
  const pend = count / 2;
  for (let pi = 0; pi < pend; pi++) {
    const b1 = 2 * pi * 3; // bob1
    const b2 = (2 * pi + 1) * 3; // bob2
    const o = pi * 12; // 4 verts * 3
    // segment: pivot -> bob1
    rodPos[o] = 0;
    rodPos[o + 1] = 0;
    rodPos[o + 2] = 0;
    rodPos[o + 3] = p[b1];
    rodPos[o + 4] = p[b1 + 1];
    rodPos[o + 5] = p[b1 + 2];
    // segment: bob1 -> bob2
    rodPos[o + 6] = p[b1];
    rodPos[o + 7] = p[b1 + 1];
    rodPos[o + 8] = p[b1 + 2];
    rodPos[o + 9] = p[b2];
    rodPos[o + 10] = p[b2 + 1];
    rodPos[o + 11] = p[b2 + 2];
  }
  rodGeo.attributes.position.needsUpdate = true;
}

// (Re)build the cloth mesh for an R×R grid: a flat position buffer (filled each
// frame from WASM) plus a static triangle index (two tris per quad).
function buildCloth(R) {
  if (clothMesh) {
    scene.remove(clothMesh);
    clothGeo.dispose();
    clothMesh = null;
  }
  clothPos = new Float32Array(R * R * 3);
  const index = [];
  for (let row = 0; row < R - 1; row++) {
    for (let col = 0; col < R - 1; col++) {
      const a = row * R + col;
      const b = row * R + col + 1;
      const c = (row + 1) * R + col;
      const d = (row + 1) * R + col + 1;
      index.push(a, c, b, b, c, d); // two triangles per cell
    }
  }
  clothGeo = new THREE.BufferGeometry();
  clothGeo.setAttribute('position', new THREE.BufferAttribute(clothPos, 3));
  clothGeo.setIndex(index);
  clothMesh = new THREE.Mesh(clothGeo, clothMat);
  clothMesh.frustumCulled = false; // the sheet swings outside its initial box
  scene.add(clothMesh);
}

// Copy the WASM particle positions into the cloth vertices and recompute normals.
function updateCloth(p) {
  clothPos.set(p);
  clothGeo.attributes.position.needsUpdate = true;
  clothGeo.computeVertexNormals();
}

// --- gradient-descent contour heatmap (sim #4) ----------------------------
// A flat plane lying in the XZ-plane just under the walkers, textured with the
// loss landscape sampled by Rust. The math domain is centred on the world
// origin and scaled to ±domain_extent, so the walkers (also in world coords)
// land exactly on their loss value.
let contourMesh = null;
let contourGeo = null;
let contourMat = null;
let contourTex = null;
let contourData = null; // Uint8 RGBA backing the DataTexture
let contourDim = 0;
// #7ad1a5 — the basin accent, premultiplied to 0..1 for blending.
const ACCENT_RGB = [0x7a / 255, 0xd1 / 255, 0xa5 / 255];
const ISO_BANDS = 7; // number of baked iso-contour lines

function buildContour() {
  if (contourMesh) {
    scene.remove(contourMesh);
    contourGeo.dispose();
    contourTex.dispose();
    contourMat.dispose();
    contourMesh = null;
  }
  const dim = world.grid_dim();
  const ext = world.domain_extent();
  contourDim = dim;
  contourData = new Uint8Array(dim * dim * 4);
  contourTex = new THREE.DataTexture(contourData, dim, dim, THREE.RGBAFormat, THREE.UnsignedByteType);
  contourTex.minFilter = THREE.LinearFilter;
  contourTex.magFilter = THREE.LinearFilter;
  contourMat = new THREE.MeshBasicMaterial({ map: contourTex }); // unlit: the heatmap IS the color
  contourGeo = new THREE.PlaneGeometry(ext * 2, ext * 2);
  contourMesh = new THREE.Mesh(contourGeo, contourMat);
  contourMesh.rotation.x = -Math.PI / 2; // lie flat in the XZ-plane
  contourMesh.position.y = -0.01; // just below the walkers/trails
  contourMesh.frustumCulled = false;
  scene.add(contourMesh);
  refreshContourTexture();
}

// Repaint the heatmap from the Rust grid. Cheap dynamic-range compression (sqrt)
// keeps high-range landscapes (Rosenbrock/Rastrigin) from washing out to a flat
// square; baked iso-bands and a restrained basin accent add readable structure.
// Rebuilt only on sim load + landscape change, never per frame.
function refreshContourTexture() {
  if (!contourMesh) return;
  const g = lossGrid();
  const dim = contourDim;
  if (g.length < dim * dim) return;
  const lo = world.loss_min();
  const hi = world.loss_max();
  const range = hi > lo ? hi - lo : 1;
  for (let row = 0; row < dim; row++) {
    // Flip V so the math-y axis aligns with world +Z (verified: Rosenbrock's
    // valley lands where the walkers converge).
    const gridRow = dim - 1 - row;
    for (let col = 0; col < dim; col++) {
      const f = g[gridRow * dim + col];
      let t = (f - lo) / range;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      t = Math.sqrt(t); // compress the dynamic range
      let lum = 0.06 + 0.62 * t; // dark valley -> light ridge
      // Iso-band: a thin dark line at each evenly spaced level crossing.
      const frac = t * ISO_BANDS - Math.floor(t * ISO_BANDS);
      if (frac < 0.07 || frac > 0.93) lum *= 0.55;
      // Basin accent: tint the low-loss region toward mint (strong near t=0).
      const basin = 1 - t * 2.0;
      const b = basin < 0 ? 0 : basin;
      const r = lum + (ACCENT_RGB[0] - lum) * 0.3 * b;
      const gc = lum + (ACCENT_RGB[1] - lum) * 0.3 * b;
      const bc = lum + (ACCENT_RGB[2] - lum) * 0.3 * b;
      const o = (row * dim + col) * 4;
      contourData[o] = (r * 255) | 0;
      contourData[o + 1] = (gc * 255) | 0;
      contourData[o + 2] = (bc * 255) | 0;
      contourData[o + 3] = 255;
    }
  }
  contourTex.needsUpdate = true;
}

// Descent trails: a per-walker ring buffer of recent world positions, drawn as
// fading line ribbons so each optimizer's path down the surface is visible.
const TRAIL_LEN = 64;
const TRAIL_Y = 0.005; // lift trails just above the contour plane
let trailGeo = null;
let trailPos = null; // Float32Array backing the LineSegments positions
let trailCol = null; // per-vertex colors (walker hue, faded by age)
let trailLines = null;
let trailHist = null; // ring buffer: trailCount * TRAIL_LEN * 3 world coords
let trailHead = 0; // next write slot in the ring
let trailCount = 0; // walker count the buffers were sized for

function buildTrails() {
  if (trailLines) {
    scene.remove(trailLines);
    trailGeo.dispose();
    trailLines = null;
  }
  trailCount = count;
  const segs = TRAIL_LEN - 1;
  const nVerts = trailCount * segs * 2; // LineSegments: 2 endpoints per segment
  trailHist = new Float32Array(trailCount * TRAIL_LEN * 3);
  trailPos = new Float32Array(nVerts * 3);
  trailCol = new Float32Array(nVerts * 3);
  trailHead = 0;
  trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
  trailGeo.setAttribute('color', new THREE.BufferAttribute(trailCol, 3));
  trailLines = new THREE.LineSegments(trailGeo, rodMat); // reuse the vertex-colored line material
  trailLines.frustumCulled = false;
  scene.add(trailLines);
  primeTrails();
}

// Fill every walker's whole ring with its current position, so a fresh start (or
// a landscape/optimizer switch) doesn't draw a streak from the previous spot.
function primeTrails() {
  if (!trailHist) return;
  const p = positions();
  for (let w = 0; w < trailCount; w++) {
    const px = p[w * 3];
    const pz = p[w * 3 + 2];
    for (let c = 0; c < TRAIL_LEN; c++) {
      const h = (w * TRAIL_LEN + c) * 3;
      trailHist[h] = px;
      trailHist[h + 1] = TRAIL_Y;
      trailHist[h + 2] = pz;
    }
  }
  trailHead = 0;
  rebuildTrailSegments();
}

// Record the current positions into the ring and rebuild the segment buffers.
function updateTrails(p) {
  if (!trailHist) return;
  for (let w = 0; w < trailCount; w++) {
    const h = (w * TRAIL_LEN + trailHead) * 3;
    trailHist[h] = p[w * 3];
    trailHist[h + 1] = TRAIL_Y;
    trailHist[h + 2] = p[w * 3 + 2];
  }
  trailHead = (trailHead + 1) % TRAIL_LEN;
  rebuildTrailSegments();
}

// Walk each ring oldest->newest, emitting line segments whose color is the
// walker's hue dimmed by age (older = fainter). One distinct hue per walker so
// the individual trajectories stay legible as they converge.
function rebuildTrailSegments() {
  const segs = TRAIL_LEN - 1;
  let v = 0;
  for (let w = 0; w < trailCount; w++) {
    indexToColor(w, trailCount, tmpColor);
    for (let c = 0; c < segs; c++) {
      const rOld = (trailHead + c) % TRAIL_LEN;
      const rNew = (trailHead + c + 1) % TRAIL_LEN;
      const hOld = (w * TRAIL_LEN + rOld) * 3;
      const hNew = (w * TRAIL_LEN + rNew) * 3;
      const fade = (c + 1) / segs; // 0 (oldest) -> 1 (newest)
      const o = v * 3;
      trailPos[o] = trailHist[hOld];
      trailPos[o + 1] = trailHist[hOld + 1];
      trailPos[o + 2] = trailHist[hOld + 2];
      trailPos[o + 3] = trailHist[hNew];
      trailPos[o + 4] = trailHist[hNew + 1];
      trailPos[o + 5] = trailHist[hNew + 2];
      const cr = tmpColor.r * fade;
      const cg = tmpColor.g * fade;
      const cb = tmpColor.b * fade;
      trailCol[o] = cr;
      trailCol[o + 1] = cg;
      trailCol[o + 2] = cb;
      trailCol[o + 3] = cr;
      trailCol[o + 4] = cg;
      trailCol[o + 5] = cb;
      v += 2;
    }
  }
  trailGeo.attributes.position.needsUpdate = true;
  trailGeo.attributes.color.needsUpdate = true;
}

// (Re)create the InstancedMesh for the current body count. instanceColor must
// be fully initialized or untouched instances render black. For sims that render
// their own geometry (cloth) the instanced spheres are built but hidden.
function buildMesh() {
  if (mesh) {
    scene.remove(mesh);
    mesh.dispose();
  }
  mesh = new THREE.InstancedMesh(sphereGeo, sphereMat, count);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  for (let i = 0; i < count; i++) mesh.setColorAt(i, BASE_COLOR);
  mesh.instanceColor.needsUpdate = true;
  mesh.visible = cfg.bobs !== false; // cloth hides the spheres
  scene.add(mesh);

  if (cfg.rods) buildRods();
  else if (rodLines) rodLines.visible = false;

  if (cfg.cloth) buildCloth(Math.round(Math.sqrt(count)));
  else if (clothMesh) clothMesh.visible = false;

  if (cfg.contour) {
    buildContour();
    buildTrails();
  } else {
    if (contourMesh) contourMesh.visible = false;
    if (trailLines) trailLines.visible = false;
  }
}

// Rebuild the active sim with a new body count (slider / sim switch).
function buildWorld(n) {
  releaseGrab(); // indices change; drop anything held
  if (world && typeof world.free === 'function') world.free();
  bodyCount = n;
  world = new World(cfg.kind, n);
  positions = makePositions(world);
  extra = makeExtra(world);
  lossGrid = makeGrid(world);
  simRadius = world.radius();
  simBounds = world.bounds();
  count = world.count();
  buildMesh();
  statCount.textContent = count;
  // A fresh World boots at the sim's built-in defaults; re-push the live control
  // state so a count change (or sim load) preserves the user's selections.
  reapplyParams();
}

// Frame the camera on the active sim (used on sim load and by Reset View).
function frameCamera() {
  camera.position.copy(cfg.camPos);
  if (cfg.camTarget) controls.target.copy(cfg.camTarget);
  else controls.target.set(0, cfg.boxed ? simBounds * 0.5 : 0, 0);
  controls.update();
}

// Switch to a different sim: reconfigure environment, controls, UI, then build.
// `restore` (optional) carries persisted control state (URL hash / localStorage);
// each per-sim default below falls back to it so a shared link or a return visit
// reopens with the user's exact settings.
function loadSim(nextKey, restore) {
  if (!SIMS[nextKey]) return;
  simKey = nextKey;
  cfg = SIMS[nextKey];

  // Restrained per-sim signature accent: drives slider fill, active tab, focus.
  document.documentElement.style.setProperty('--accent', cfg.accent || '#6ca0ff');

  grid.visible = cfg.showGrid;
  box.visible = cfg.boxed;

  // Speed is a global multiplier (not per-sim), so only a restore touches it.
  if (restore?.spd != null) {
    speed = clamp(restore.spd, parseFloat(speedSlider.min), parseFloat(speedSlider.max));
    speedSlider.value = String(speed);
    speedLabel.textContent = `${speed.toFixed(1)}×`;
  }

  countName.textContent = cfg.countName || 'Count';
  countSlider.min = String(cfg.minCount);
  countSlider.max = String(cfg.maxCount);
  const initCount = restore?.n != null ? clamp(Math.round(restore.n), cfg.minCount, cfg.maxCount) : cfg.defaultCount;
  countSlider.value = String(initCount);
  countLabel.textContent = String(initCount);

  // The gravity control is shared; each sim relabels/rescales it (and a fresh
  // World starts at the matching default, so no set_param push is needed here).
  gravRow.style.display = cfg.hasGravityParam ? '' : 'none';
  if (cfg.hasGravityParam) {
    gravName.textContent = cfg.gravLabel;
    gravSlider.min = String(cfg.gravMin);
    gravSlider.max = String(cfg.gravMax);
    gravSlider.step = String(cfg.gravStep);
    const g = restore?.g != null ? clamp(restore.g, cfg.gravMin, cfg.gravMax) : cfg.gravDefault;
    gravSlider.value = String(g);
    gravLabel.textContent = g.toFixed(cfg.gravDecimals ?? 1);
  }

  // Wind control (cloth only): same shared-default trick as gravity.
  windRow.style.display = cfg.hasWind ? '' : 'none';
  if (cfg.hasWind) {
    windName.textContent = cfg.windLabel;
    windSlider.min = String(cfg.windMin);
    windSlider.max = String(cfg.windMax);
    windSlider.step = String(cfg.windStep);
    const w = restore?.w != null ? clamp(restore.w, cfg.windMin, cfg.windMax) : cfg.windDefault;
    windSlider.value = String(w);
    windLabel.textContent = w.toFixed(1);
  }

  // Pin toggle (cloth only): defaults to corners-pinned unless a restore overrides.
  pinMode = cfg.hasPin ? (restore?.pin ?? 0) : 0;
  btnPin.style.display = cfg.hasPin ? '' : 'none';
  btnPin.classList.toggle('is-on', pinMode === 1);
  if (cfg.hasPin) btnPin.querySelector('.lbl').textContent = pinMode === 1 ? 'Pin: Top Edge' : 'Pin: Corners';

  // Gradient-descent selector state (read by reapplyParams inside buildWorld, so
  // it must be set BEFORE the build).
  if (cfg.contour) {
    landscapeIdx = restore?.land ?? cfg.defaultLandscape ?? 0;
    optimizerIdx = restore?.opt ?? cfg.defaultOptimizer ?? 3;
    betaVal = restore?.beta ?? cfg.betaDefault ?? 0.9;
  }

  buildWorld(initCount);

  // Landscape / optimizer / beta controls (gradient-descent only). Built AFTER
  // buildWorld so `world` exists for the pick handlers.
  const isContour = !!cfg.contour;
  landscapeRow.style.display = isContour ? '' : 'none';
  optimizerRow.style.display = isContour ? '' : 'none';
  betaRow.style.display = isContour && cfg.hasBeta ? '' : 'none';
  if (isContour) {
    buildSelect(landscapeNav, cfg.landscapes, landscapeIdx, (i) => {
      landscapeIdx = i;
      world.set_param(1, i); // rebuilds the grid + re-seeds walkers in Rust
      refreshContourTexture();
      primeTrails();
      schedulePersist();
    });
    buildSelect(optimizerNav, cfg.optimizers, optimizerIdx, (i) => {
      optimizerIdx = i;
      world.set_param(2, i); // re-seeds for a fair restart
      primeTrails();
      schedulePersist();
    });
    if (cfg.hasBeta) {
      betaSlider.min = String(cfg.betaMin);
      betaSlider.max = String(cfg.betaMax);
      betaSlider.step = String(cfg.betaStep);
      betaSlider.value = String(betaVal);
      betaLabel.textContent = betaVal.toFixed(2);
    }
  }

  frameCamera();

  refreshFills(); // sliders were just re-min/maxed; repaint their accent fills

  if (cfg.hint) hintEl.textContent = cfg.hint;
  for (const b of segBtns) b.classList.toggle('is-active', b.dataset.sim === nextKey);

  schedulePersist();
}

// Fill a `.select` nav with one borderless button per label; clicking re-marks
// the active button and fires `onPick(index)`.
function buildSelect(container, labels, activeIdx, onPick) {
  container.innerHTML = '';
  labels.forEach((label, i) => {
    const b = document.createElement('button');
    b.className = 'select__btn' + (i === activeIdx ? ' is-active' : '');
    b.textContent = label;
    b.addEventListener('click', () => {
      for (const c of container.children) c.classList.remove('is-active');
      b.classList.add('is-active');
      onPick(i);
    });
    container.appendChild(b);
  });
}

// --- ui -------------------------------------------------------------------
const statCount = document.getElementById('stat-count');
const statFps = document.getElementById('stat-fps');
const countSlider = document.getElementById('count-slider');
const countLabel = document.getElementById('count-label');
const countName = document.getElementById('count-name');
const speedSlider = document.getElementById('speed-slider');
const speedLabel = document.getElementById('speed-label');
const gravSlider = document.getElementById('grav-slider');
const gravLabel = document.getElementById('grav-label');
const gravName = document.getElementById('grav-name');
const gravRow = document.getElementById('grav-row');
const windSlider = document.getElementById('wind-slider');
const windLabel = document.getElementById('wind-label');
const windName = document.getElementById('wind-name');
const windRow = document.getElementById('wind-row');
const landscapeRow = document.getElementById('landscape-row');
const landscapeNav = document.getElementById('landscape-nav');
const optimizerRow = document.getElementById('optimizer-row');
const optimizerNav = document.getElementById('optimizer-nav');
const betaRow = document.getElementById('beta-row');
const betaSlider = document.getElementById('beta-slider');
const betaLabel = document.getElementById('beta-label');
const btnPin = document.getElementById('btn-pin');
const btnPause = document.getElementById('btn-pause');
const btnStep = document.getElementById('btn-step');
const btnView = document.getElementById('btn-view');
const btnFull = document.getElementById('btn-fullscreen');
const segBtns = Array.from(document.querySelectorAll('.seg__btn'));
const hintEl = document.querySelector('.hint');

let speed = 1; // sim-time multiplier
let paused = false;
let pinMode = 0; // cloth pin mode: 0 = corners, 1 = top edge
let landscapeIdx = 0; // gradient-descent: selected loss landscape
let optimizerIdx = 3; // gradient-descent: selected optimizer (default Adam)
let betaVal = 0.9; // gradient-descent: momentum / Adam beta1

// Paint a slider's accent "fill" up to its current value (CSS reads --fill for the
// WebKit track gradient; Firefox uses native ::-moz-range-progress and ignores it).
function setFill(el) {
  const min = parseFloat(el.min);
  const max = parseFloat(el.max);
  const v = parseFloat(el.value);
  const pct = max > min ? ((v - min) / (max - min)) * 100 : 0;
  el.style.setProperty('--fill', `${pct}%`);
}
function refreshFills() {
  for (const el of document.querySelectorAll('input[type="range"]')) setFill(el);
}
// One delegated listener keeps every slider's fill in sync as it's dragged, and
// schedules a (debounced) persist so any slider change lands in URL + storage.
document.querySelector('.panel').addEventListener('input', (e) => {
  if (e.target.matches('input[type="range"]')) {
    setFill(e.target);
    schedulePersist();
  }
});

for (const b of segBtns) {
  b.addEventListener('click', () => loadSim(b.dataset.sim));
}

// Live label while dragging; rebuild only on release so we don't thrash.
countSlider.addEventListener('input', () => {
  countLabel.textContent = countSlider.value;
});
countSlider.addEventListener('change', () => {
  buildWorld(parseInt(countSlider.value, 10));
});

speedSlider.addEventListener('input', () => {
  speed = parseFloat(speedSlider.value);
  speedLabel.textContent = `${speed.toFixed(1)}×`;
});

gravSlider.addEventListener('input', () => {
  const g = parseFloat(gravSlider.value);
  gravLabel.textContent = g.toFixed(cfg.gravDecimals ?? 1);
  // id 0 is each sim's primary tunable: gravity multiplier (collisions),
  // gravitational constant G (n-body), gravity g (pendulum / cloth), or the
  // learning rate (gradient descent).
  world.set_param(0, g);
});

// Wind strength (cloth): id 1.
windSlider.addEventListener('input', () => {
  const w = parseFloat(windSlider.value);
  windLabel.textContent = w.toFixed(1);
  world.set_param(1, w);
});

// Momentum / Adam beta1 (gradient descent): id 3.
betaSlider.addEventListener('input', () => {
  betaVal = parseFloat(betaSlider.value);
  betaLabel.textContent = betaVal.toFixed(2);
  world.set_param(3, betaVal);
});

// Pin mode toggle (cloth): id 2, 0 = corners, 1 = whole top edge.
btnPin.addEventListener('click', () => {
  pinMode = pinMode === 0 ? 1 : 0;
  world.set_param(2, pinMode);
  btnPin.querySelector('.lbl').textContent = pinMode === 1 ? 'Pin: Top Edge' : 'Pin: Corners';
  btnPin.classList.toggle('is-on', pinMode === 1);
  schedulePersist();
});

btnPause.addEventListener('click', () => {
  paused = !paused;
  btnPause.querySelector('.lbl').textContent = paused ? 'Play' : 'Pause';
  btnPause.classList.toggle('is-on', paused);
});

// Step: advance exactly one fixed tick and repaint (handy while paused to scrub
// chaos/cloth frame by frame).
btnStep.addEventListener('click', () => {
  world.step(FIXED);
  render();
});

// Reset View: re-frame the camera on the current sim without rebuilding it.
btnView.addEventListener('click', frameCamera);

// Fullscreen toggle (the resize listener handles the canvas/aspect update).
// requestFullscreen rejects in sandboxed iframes; swallow that so it's a no-op.
btnFull.addEventListener('click', () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen?.().catch(() => {});
});
document.addEventListener('fullscreenchange', () => {
  // The button is icon-only now, so reflect state via a class + label rather than text.
  const fs = !!document.fullscreenElement;
  btnFull.classList.toggle('is-on', fs);
  const fsLabel = fs ? 'Exit fullscreen' : 'Fullscreen';
  btnFull.title = fsLabel;
  btnFull.setAttribute('aria-label', fsLabel);
});

// --- About / Settings modals ---------------------------------------------
const modalAbout = document.getElementById('modal-about');
const modalSettings = document.getElementById('modal-settings');

function openModal(m) {
  m.hidden = false;
  // Force a reflow before adding the class so the entrance transition runs from
  // the hidden state (rAF would stall in a backgrounded/headless tab).
  void m.offsetWidth;
  m.classList.add('is-open');
}
function closeModal(m) {
  m.classList.remove('is-open');
  setTimeout(() => {
    m.hidden = true;
  }, 240);
}

document.getElementById('btn-about').addEventListener('click', () => openModal(modalAbout));
document.getElementById('btn-settings').addEventListener('click', () => openModal(modalSettings));

// Close on scrim or close-button click (both carry data-close); inside-card clicks are ignored.
for (const m of [modalAbout, modalSettings]) {
  m.addEventListener('click', (e) => {
    if (e.target.closest('[data-close]')) closeModal(m);
  });
}
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (!modalAbout.hidden) closeModal(modalAbout);
  if (!modalSettings.hidden) closeModal(modalSettings);
});

// --- settings toggles -----------------------------------------------------
// High quality: full device pixel ratio (capped at 2) vs 1 for performance.
const setQuality = document.getElementById('set-quality');
setQuality.addEventListener('change', () => {
  renderer.setPixelRatio(setQuality.checked ? Math.min(window.devicePixelRatio, 2) : 1);
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Vignette: toggle the decorative edge darkening.
const vignetteEl = document.querySelector('.vignette');
const setVignette = document.getElementById('set-vignette');
setVignette.addEventListener('change', () => {
  vignetteEl.style.display = setVignette.checked ? '' : 'none';
});

// Reduce motion: kill UI transitions/animations regardless of the OS preference.
const setMotion = document.getElementById('set-motion');
setMotion.addEventListener('change', () => {
  document.body.classList.toggle('reduce-motion', setMotion.checked);
});

// Re-push the current slider/toggle state onto a freshly-built World (reset spins
// up a new sim at its built-in defaults, which can drift from the live controls).
function reapplyParams() {
  if (cfg.hasGravityParam) world.set_param(0, parseFloat(gravSlider.value));
  if (cfg.hasWind) world.set_param(1, parseFloat(windSlider.value));
  if (cfg.hasPin) world.set_param(2, pinMode);
  // Gradient descent: id 1 = landscape, id 2 = optimizer, id 3 = beta. (These
  // never collide with wind/pin: contour sims set neither hasWind nor hasPin.)
  if (cfg.contour) {
    world.set_param(1, landscapeIdx);
    world.set_param(2, optimizerIdx);
    if (cfg.hasBeta) world.set_param(3, betaVal);
    refreshContourTexture();
    primeTrails();
  }
}

document.getElementById('btn-reset').addEventListener('click', () => {
  world.reset();
  reapplyParams();
});

// --- grab & throw ---------------------------------------------------------
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
const dragPlane = new THREE.Plane();
const hitPoint = new THREE.Vector3();
const camDir = new THREE.Vector3();
const instVel = new THREE.Vector3();
const lastDragPos = new THREE.Vector3();
const throwVel = new THREE.Vector3();
let grabbed = -1; // instance id being dragged, -1 = none
let lastDragTime = 0;
const MAX_THROW = 40;

const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);

function pointerNDC(e) {
  const rect = canvas.getBoundingClientRect();
  ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
}

function clampToBox(v) {
  if (cfg.boxed) {
    const lim = simBounds - simRadius;
    v.x = clamp(v.x, -lim, lim);
    v.z = clamp(v.z, -lim, lim);
    v.y = clamp(v.y, simRadius, 2 * simBounds - simRadius);
  } else {
    const lim = simBounds;
    v.x = clamp(v.x, -lim, lim);
    v.y = clamp(v.y, -lim, lim);
    v.z = clamp(v.z, -lim, lim);
  }
}

// Nearest body to the click ray within a generous radius. More forgiving than
// exact mesh raycasting and works for the tiny N-body stars too.
function pickBody() {
  raycaster.setFromCamera(ndc, camera);
  const o = raycaster.ray.origin;
  const d = raycaster.ray.direction; // normalized
  const p = positions();
  const pickR = Math.max(simRadius * 1.6, 0.35);
  const pickR2 = pickR * pickR;
  let best = -1;
  let bestT = Infinity;
  for (let i = 0; i < count; i++) {
    const k = i * 3;
    const wx = p[k] - o.x;
    const wy = p[k + 1] - o.y;
    const wz = p[k + 2] - o.z;
    const t = wx * d.x + wy * d.y + wz * d.z; // distance along ray to closest point
    if (t < 0 || t >= bestT) continue; // behind camera, or farther than current best
    const perp2 = wx * wx + wy * wy + wz * wz - t * t;
    if (perp2 <= pickR2) {
      bestT = t;
      best = i;
    }
  }
  return best;
}

function releaseGrab() {
  if (grabbed < 0) return;
  mesh.setColorAt(grabbed, BASE_COLOR);
  mesh.instanceColor.needsUpdate = true;
  world.set_held(-1);
  grabbed = -1;
  controls.enabled = true;
}

// Capture phase on window so this runs BEFORE OrbitControls' canvas listener;
// when we grab a body we stopPropagation so the orbit gesture never starts.
function onPointerDown(e) {
  if (e.button !== 0 || e.target !== canvas) return;
  if (!cfg.grabbable) return; // e.g. pendulum: let OrbitControls orbit instead
  pointerNDC(e);
  const id = pickBody();
  if (id < 0) return; // missed -> let OrbitControls orbit

  grabbed = id;
  controls.enabled = false;
  e.stopPropagation();

  world.set_held(grabbed);
  mesh.setColorAt(grabbed, HELD_COLOR);
  mesh.instanceColor.needsUpdate = true;

  const p = positions();
  const k = grabbed * 3;
  lastDragPos.set(p[k], p[k + 1], p[k + 2]);
  camera.getWorldDirection(camDir);
  dragPlane.setFromNormalAndCoplanarPoint(camDir, lastDragPos);
  throwVel.set(0, 0, 0);
  lastDragTime = performance.now();
}

function onPointerMove(e) {
  if (grabbed < 0) return;
  pointerNDC(e);
  raycaster.setFromCamera(ndc, camera);
  if (!raycaster.ray.intersectPlane(dragPlane, hitPoint)) return;
  clampToBox(hitPoint);

  const now = performance.now();
  const dt = Math.max((now - lastDragTime) / 1000, 1e-3);
  instVel.copy(hitPoint).sub(lastDragPos).divideScalar(dt);
  throwVel.lerp(instVel, 0.6); // smooth cursor jitter
  lastDragTime = now;
  lastDragPos.copy(hitPoint);

  world.set_pos(grabbed, hitPoint.x, hitPoint.y, hitPoint.z);
}

function onPointerUp() {
  if (grabbed < 0) return;
  if (performance.now() - lastDragTime > 120) throwVel.set(0, 0, 0); // paused -> drop
  if (throwVel.length() > MAX_THROW) throwVel.setLength(MAX_THROW);
  world.set_vel(grabbed, throwVel.x, throwVel.y, throwVel.z);
  releaseGrab();
}

window.addEventListener('pointerdown', onPointerDown, true);
window.addEventListener('pointermove', onPointerMove);
window.addEventListener('pointerup', onPointerUp);

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Map a normalized speed [0,1] to a blue(slow) -> orange(fast) color.
function speedToColor(t, out) {
  t = clamp(t, 0, 1);
  return out.setHSL(0.62 - 0.54 * t, 0.72, 0.38 + 0.34 * t);
}

// Map a walker's loss to a color: low loss -> bright mint (converged), high loss
// -> dim grey. sqrt-compressed so a wide loss range still spreads visibly.
function lossToColor(loss, lo, hi, out) {
  let t = hi > lo ? (loss - lo) / (hi - lo) : 0;
  t = clamp(t, 0, 1);
  t = Math.sqrt(t);
  return out.setHSL(0.41, 0.55 * (1 - t) + 0.05, 0.66 - 0.46 * t);
}

// --- render + fixed-timestep loop -----------------------------------------
const FIXED = 1 / 120;
let last = performance.now();
let acc = 0;
let fpsTime = 0;
let fpsFrames = 0;

function render() {
  const p = positions(); // re-fetched AFTER stepping (step may reallocate)
  const byLoss = cfg.colorByLoss;
  const ex = cfg.colorBySpeed || byLoss ? extra() : null;
  const byIndex = cfg.colorByIndex;
  const recolor = ex || byIndex;
  const cs = cfg.centralScale;
  const pend = count / 2; // only used for pendulum index coloring
  // Loss range for color normalization (WASM calls — hoisted out of the loop).
  const lossLo = byLoss ? world.loss_min() : 0;
  const lossHi = byLoss ? world.loss_max() : 1;
  // Cloth renders its own mesh; the instanced spheres stay hidden, so skip them.
  if (cfg.bobs !== false) {
    for (let i = 0; i < count; i++) {
      const k = i * 3;
      dummy.position.set(p[k], p[k + 1], p[k + 2]);
      dummy.scale.setScalar(simRadius * (cs > 1 && i === 0 ? cs : 1));
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      if (recolor) {
        if (i === grabbed) tmpColor.copy(HELD_COLOR);
        else if (byLoss) lossToColor(ex[i], lossLo, lossHi, tmpColor);
        else if (cfg.colorBySpeed) speedToColor(ex[i] / cfg.speedScale, tmpColor);
        else indexToColor((i / 2) | 0, pend, tmpColor); // two bobs share a pendulum hue
        mesh.setColorAt(i, tmpColor);
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (recolor) mesh.instanceColor.needsUpdate = true;
  }
  if (cfg.rods && rodLines) updateRods(p);
  if (cfg.cloth && clothMesh) updateCloth(p);
  if (cfg.contour && trailLines) updateTrails(p);
  controls.update();
  renderer.render(scene, camera);
}

function frame(now) {
  let dt = (now - last) / 1000;
  last = now;
  if (dt > 0.1) dt = 0.1; // clamp after tab-switch / stalls
  if (paused) {
    acc = 0;
  } else {
    acc += dt * speed;
    while (acc >= FIXED) {
      world.step(FIXED);
      acc -= FIXED;
    }
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

// --- state persistence: shareable URL hash + localStorage -------------------
// One compact snapshot of the user-facing controls, written to the location hash
// (shareable) and localStorage (sticky across visits). Only keys relevant to the
// active sim are included. loadSim() reads this same shape back via its `restore`
// argument, so a shared link or a return visit reopens with the exact settings.
const LS_KEY = 'physics-sandbox-state';
let restoring = false; // true while applying persisted state — suppresses re-persist
let persistTimer = 0;

function currentState() {
  const st = { sim: simKey, n: parseInt(countSlider.value, 10), spd: speed };
  if (cfg.hasGravityParam) st.g = parseFloat(gravSlider.value);
  if (cfg.hasWind) st.w = parseFloat(windSlider.value);
  if (cfg.hasPin) st.pin = pinMode;
  if (cfg.contour) {
    st.land = landscapeIdx;
    st.opt = optimizerIdx;
    if (cfg.hasBeta) st.beta = betaVal;
  }
  return st;
}

function writeHash() {
  const hash = '#' + Object.entries(currentState()).map(([k, v]) => `${k}=${v}`).join('&');
  // replaceState avoids spamming browser history (and doesn't fire hashchange).
  history.replaceState(null, '', hash);
}

function saveLocal() {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(currentState()));
  } catch {
    /* storage unavailable (private mode / quota) — non-fatal */
  }
}

// Debounced: a slider drag fires many input events; persist once it settles.
function schedulePersist() {
  if (restoring) return;
  clearTimeout(persistTimer);
  persistTimer = setTimeout(() => {
    writeHash();
    saveLocal();
  }, 250);
}

// Parse a snapshot from the hash (k=v&...) or a localStorage JSON string. Returns
// null unless it names a known sim, so a malformed/foreign source is ignored.
function parseState(raw, fromHash) {
  if (!raw) return null;
  let st = null;
  if (fromHash) {
    st = {};
    for (const part of raw.replace(/^#/, '').split('&')) {
      if (!part) continue;
      const [k, v] = part.split('=');
      if (k === 'sim') st.sim = v;
      else if (v !== undefined && v !== '') st[k] = Number(v);
    }
  } else {
    try {
      st = JSON.parse(raw);
    } catch {
      return null;
    }
  }
  return st && st.sim && SIMS[st.sim] ? st : null;
}

function loadLocal() {
  try {
    return parseState(localStorage.getItem(LS_KEY), false);
  } catch {
    return null;
  }
}

// Re-apply state when the hash is edited/navigated in an already-open tab.
// (writeHash uses replaceState, which doesn't fire this — so no feedback loop.)
window.addEventListener('hashchange', () => {
  const st = parseState(location.hash, true);
  if (!st) return;
  restoring = true;
  loadSim(st.sim, st);
  restoring = false;
});

// --- keyboard shortcuts -----------------------------------------------------
// space=pause, R=reset, F=fullscreen, S=screenshot, 1-N=sims. Ignored while
// typing in a control, with a modal open, or with a modifier held.
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const t = e.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
  if (!modalAbout.hidden || !modalSettings.hidden) return;

  if (e.key >= '1' && e.key <= '9') {
    const seg = segBtns[parseInt(e.key, 10) - 1];
    if (seg) loadSim(seg.dataset.sim);
    return;
  }
  switch (e.key) {
    case ' ': // space toggles pause globally (preventDefault stops page scroll)
      e.preventDefault();
      btnPause.click();
      break;
    case 'r':
    case 'R':
      document.getElementById('btn-reset').click();
      break;
    case 'f':
    case 'F':
      btnFull.click();
      break;
    case 's':
    case 'S':
      e.preventDefault();
      captureScreenshot();
      break;
  }
});

// --- screenshot (PNG download) ----------------------------------------------
// Render the current frame, then snapshot the canvas. preserveDrawingBuffer (set
// on the renderer) keeps the buffer readable here.
function captureScreenshot() {
  render();
  renderer.domElement.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `physics-${simKey}-${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }, 'image/png');
}

// Boot at the persisted state if any (URL hash wins over localStorage), else the
// default sim. restoring=true keeps boot from rewriting a clean URL.
const bootState = parseState(location.hash, true) || loadLocal();
restoring = true;
loadSim(bootState ? bootState.sim : 'collisions', bootState); // build the initial sim + scene
restoring = false;
render(); // paint the first frame before the loop ticks
schedule();

// Engine is up and the first frame is on screen — fade out the boot loader.
const loaderEl = document.getElementById('loader');
if (loaderEl) {
  loaderEl.classList.add('is-hidden');
  setTimeout(() => loaderEl.remove(), 700);
}

document.addEventListener('visibilitychange', () => {
  last = performance.now(); // avoid a huge dt spike when visibility flips
});

// Dev-only: headless previews keep the tab hidden, which pauses rAF. This hook
// lets the harness manually advance the sim and inspect state for verification.
if (import.meta.env.DEV) {
  window.__sandbox = {
    tick(steps = 60) {
      for (let i = 0; i < steps; i++) world.step(FIXED);
      render();
    },
    render,
    loadSim,
    buildWorld,
    setSpeed(v) {
      speed = v;
    },
    setG(v) {
      world.set_param(0, v);
    },
    setPaused(v) {
      paused = v;
    },
    rods() {
      return rodLines
        ? { visible: rodLines.visible, verts: rodGeo.attributes.position.count }
        : null;
    },
    cam() {
      return {
        pos: [camera.position.x, camera.position.y, camera.position.z],
        target: [controls.target.x, controls.target.y, controls.target.z],
      };
    },
    moveCam(x, y, z) {
      camera.position.set(x, y, z);
      controls.update();
    },
    cloth() {
      if (!clothMesh) return null;
      const p = positions();
      let topY = -Infinity;
      let botY = Infinity;
      let maxZ = -Infinity;
      let minZ = Infinity;
      for (let i = 0; i < count; i++) {
        const y = p[i * 3 + 1];
        const z = p[i * 3 + 2];
        if (y > topY) topY = y;
        if (y < botY) botY = y;
        if (z > maxZ) maxZ = z;
        if (z < minZ) minZ = z;
      }
      return {
        visible: clothMesh.visible,
        meshVisible: mesh.visible,
        verts: clothGeo.attributes.position.count,
        tris: clothGeo.index.count / 3,
        topY,
        botY,
        maxZ,
        minZ,
      };
    },
    grab(i, x, y, z) {
      world.set_held(i);
      world.set_pos(i, x, y, z);
    },
    throw(i, vx, vy, vz) {
      world.set_vel(i, vx, vy, vz);
      world.set_held(-1);
    },
    sample() {
      const p = positions();
      let minY = Infinity;
      let maxY = -Infinity;
      let maxAbsXZ = 0;
      let maxAbs = 0;
      let minPairDist = Infinity;
      let nan = false;
      for (let i = 0; i < count; i++) {
        const x = p[i * 3];
        const y = p[i * 3 + 1];
        const z = p[i * 3 + 2];
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) nan = true;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        maxAbsXZ = Math.max(maxAbsXZ, Math.abs(x), Math.abs(z));
        maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y), Math.abs(z));
      }
      if (count <= 400) {
        for (let i = 0; i < count; i++) {
          for (let j = i + 1; j < count; j++) {
            const dx = p[i * 3] - p[j * 3];
            const dy = p[i * 3 + 1] - p[j * 3 + 1];
            const dz = p[i * 3 + 2] - p[j * 3 + 2];
            const d = Math.hypot(dx, dy, dz);
            if (d < minPairDist) minPairDist = d;
          }
        }
      }
      const ex = extra();
      let maxSpeed = 0;
      for (let i = 0; i < ex.length; i++) if (ex[i] > maxSpeed) maxSpeed = ex[i];
      // Spatial spread (stddev of x,y) — for the pendulum this grows from ~0 to
      // large as nearly-identical initial states diverge (a chaos signature).
      let mx = 0;
      let my = 0;
      for (let i = 0; i < count; i++) {
        mx += p[i * 3];
        my += p[i * 3 + 1];
      }
      mx /= count || 1;
      my /= count || 1;
      let vx = 0;
      let vy = 0;
      for (let i = 0; i < count; i++) {
        const ddx = p[i * 3] - mx;
        const ddy = p[i * 3 + 1] - my;
        vx += ddx * ddx;
        vy += ddy * ddy;
      }
      const spreadXY = Math.sqrt((vx + vy) / (count || 1));
      return {
        sim: simKey,
        count,
        radius: simRadius,
        bounds: simBounds,
        minY,
        maxY,
        maxAbsXZ,
        maxAbs,
        minPairDist,
        nan,
        maxSpeed,
        spreadXY,
        extraLen: ex.length,
        gridLen: world.grid_len(),
        gridDim: world.grid_dim(),
        first: [p[0], p[1], p[2]],
      };
    },
    // Advance the gradient-descent sim and report convergence: best-walker loss
    // at start vs end, the largest single-step rise in best-loss (should be ~0),
    // the best walker's world position, and the heatmap metadata.
    descent(steps = 200) {
      const bestLoss = () => {
        const e = extra();
        let m = Infinity;
        for (let i = 0; i < e.length; i++) if (e[i] < m) m = e[i];
        return m;
      };
      const startBest = bestLoss();
      let prevBest = startBest;
      let maxIncrease = 0;
      for (let s = 0; s < steps; s++) {
        world.step(FIXED);
        const b = bestLoss();
        if (b - prevBest > maxIncrease) maxIncrease = b - prevBest;
        prevBest = b;
      }
      render();
      const exF = extra();
      let finalBest = Infinity;
      let bestIdx = 0;
      for (let i = 0; i < exF.length; i++) {
        if (exF[i] < finalBest) {
          finalBest = exF[i];
          bestIdx = i;
        }
      }
      const p = positions();
      let nan = false;
      for (let i = 0; i < count * 3; i++) if (!Number.isFinite(p[i])) nan = true;
      return {
        sim: simKey,
        landscape: landscapeIdx,
        optimizer: optimizerIdx,
        startBest,
        finalBest,
        maxIncrease,
        bestWorld: [p[bestIdx * 3], p[bestIdx * 3 + 1], p[bestIdx * 3 + 2]],
        gridLen: world.grid_len(),
        gridDim: world.grid_dim(),
        lossMin: world.loss_min(),
        lossMax: world.loss_max(),
        nan,
      };
    },
    // Heatmap + trail diagnostics: texel spread (non-uniform?), and the largest
    // trail segment right after a prime (should be ~0 → no teleport streak).
    contour() {
      if (!contourData) return { hasContour: false };
      let lumMin = 255;
      let lumMax = 0;
      const seen = new Set();
      for (let i = 0; i < contourData.length; i += 4) {
        const l = contourData[i];
        if (l < lumMin) lumMin = l;
        if (l > lumMax) lumMax = l;
        seen.add(l);
      }
      let maxSeg = 0;
      if (trailPos) {
        for (let o = 0; o < trailPos.length; o += 6) {
          const dx = trailPos[o + 3] - trailPos[o];
          const dz = trailPos[o + 5] - trailPos[o + 2];
          const d = Math.hypot(dx, dz);
          if (d > maxSeg) maxSeg = d;
        }
      }
      return { hasContour: true, texelMin: lumMin, texelMax: lumMax, distinctLum: seen.size, maxTrailSeg: maxSeg };
    },
    setLandscape(i) {
      landscapeIdx = i;
      world.set_param(1, i);
      refreshContourTexture();
      primeTrails();
    },
    setOptimizer(i) {
      optimizerIdx = i;
      world.set_param(2, i);
      primeTrails();
    },
    // Advance the boids flock and report emergent alignment via the velocity
    // polarization order parameter Φ = |Σ v̂_i| / N (≈0 incoherent, →1 a single
    // aligned flock), measured at start vs end. Headings come from position
    // deltas (velocities aren't exposed to JS). Plus in-bounds + NaN checks.
    boids(steps = 240) {
      const polarization = () => {
        const a = positions().slice(); // copy: the live view mutates on step
        world.step(FIXED);
        const b = positions();
        let sx = 0;
        let sy = 0;
        let sz = 0;
        let used = 0;
        for (let i = 0; i < count; i++) {
          const dx = b[i * 3] - a[i * 3];
          const dy = b[i * 3 + 1] - a[i * 3 + 1];
          const dz = b[i * 3 + 2] - a[i * 3 + 2];
          const d = Math.hypot(dx, dy, dz);
          if (d > 1e-6) {
            sx += dx / d;
            sy += dy / d;
            sz += dz / d;
            used++;
          }
        }
        return used ? Math.hypot(sx, sy, sz) / used : 0;
      };
      const startPhi = polarization(); // consumes 1 step
      for (let s = 0; s < steps; s++) world.step(FIXED);
      const endPhi = polarization(); // consumes 1 step
      render();
      const p = positions();
      let nan = false;
      let maxAbs = 0;
      for (let i = 0; i < count * 3; i++) {
        if (!Number.isFinite(p[i])) nan = true;
        const av = Math.abs(p[i]);
        if (av > maxAbs) maxAbs = av;
      }
      const ex = extra();
      let minSpeed = Infinity;
      let maxSpeed = 0;
      for (let i = 0; i < ex.length; i++) {
        if (ex[i] < minSpeed) minSpeed = ex[i];
        if (ex[i] > maxSpeed) maxSpeed = ex[i];
      }
      return {
        sim: simKey,
        count,
        startPhi,
        endPhi,
        bounds: simBounds,
        maxAbs,
        inBounds: maxAbs <= simBounds + 1e-3,
        minSpeed,
        maxSpeed,
        nan,
      };
    },
  };
}
