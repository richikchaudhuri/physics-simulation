use wasm_bindgen::prelude::*;

/// Every simulation exposes a flat `[x, y, z, ...]` position buffer that JS reads
/// directly out of WASM linear memory each frame (zero-copy).
pub trait Simulation {
    fn step(&mut self, dt: f32);
    fn positions(&self) -> &[f32];
    fn reset(&mut self);
    fn count(&self) -> usize;
    fn radius(&self) -> f32 {
        0.4
    }
    fn bounds(&self) -> f32 {
        4.0
    }
    /// Mark body `i` as held by the mouse (-1 = none). A held body is driven by
    /// JS (`set_pos`) and acts as an infinite-mass obstacle for the rest.
    fn set_held(&mut self, _i: i32) {}
    /// Teleport body `i` (used while dragging a grabbed body).
    fn set_pos(&mut self, _i: usize, _x: f32, _y: f32, _z: f32) {}
    /// Set body `i` velocity (used to fling a body on release).
    fn set_vel(&mut self, _i: usize, _x: f32, _y: f32, _z: f32) {}
}

/// Tiny deterministic LCG so the scene looks the same every load without an rng crate.
struct Rng(u32);
impl Rng {
    fn new(seed: u32) -> Self {
        Rng(seed)
    }
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        ((self.0 >> 8) & 0x00ff_ffff) as f32 / 16_777_216.0
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
}

/// Phase 0 placeholder sim: spheres bouncing under gravity inside an open-top box.
/// Proves the full pipeline (Rust physics -> WASM memory -> instanced render).
struct Bouncer {
    pos: Vec<f32>,
    vel: Vec<f32>,
    n: usize,
    bounds: f32,
    radius: f32,
    /// Index of the body currently grabbed by the mouse, or -1 when none.
    held: i32,
}

impl Bouncer {
    fn new(n: usize) -> Self {
        let bounds = 4.0;
        let radius = 0.4;
        let mut pos = vec![0.0f32; n * 3];
        let mut vel = vec![0.0f32; n * 3];
        let mut rng = Rng::new(0x9E37_79B9);
        for i in 0..n {
            let k = i * 3;
            pos[k] = rng.range(-bounds + radius, bounds - radius);
            pos[k + 1] = rng.range(radius, 2.0 * bounds - radius);
            pos[k + 2] = rng.range(-bounds + radius, bounds - radius);
            vel[k] = rng.range(-3.0, 3.0);
            vel[k + 1] = rng.range(-1.0, 4.0);
            vel[k + 2] = rng.range(-3.0, 3.0);
        }
        Self { pos, vel, n, bounds, radius, held: -1 }
    }
}

impl Simulation for Bouncer {
    fn step(&mut self, dt: f32) {
        let g = -9.81f32;
        let e_wall = 0.85f32; // wall restitution
        let e_ball = 0.95f32; // ball-ball restitution
        let b = self.bounds;
        let r = self.radius;
        let n = self.n;

        // 1) Integrate under gravity (semi-implicit Euler). The grabbed body is
        //    positioned by the mouse, so it skips integration.
        for i in 0..n {
            if i as i32 == self.held {
                continue;
            }
            let k = i * 3;
            self.vel[k + 1] += g * dt;
            self.pos[k] += self.vel[k] * dt;
            self.pos[k + 1] += self.vel[k + 1] * dt;
            self.pos[k + 2] += self.vel[k + 2] * dt;
        }

        // 2) Ball-ball elastic collisions, brute-force O(n^2). Phase 2 will swap
        //    this for a uniform-grid broad-phase to scale up. Masses are equal
        //    (inverse mass 1), except a grabbed body has inverse mass 0 so it
        //    behaves as an immovable obstacle and shoves the others aside.
        let two_r = 2.0 * r;
        let two_r2 = two_r * two_r;
        for i in 0..n {
            let ki = i * 3;
            let im_i = if i as i32 == self.held { 0.0 } else { 1.0 };
            for j in (i + 1)..n {
                let kj = j * 3;
                let dx = self.pos[ki] - self.pos[kj];
                let dy = self.pos[ki + 1] - self.pos[kj + 1];
                let dz = self.pos[ki + 2] - self.pos[kj + 2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < two_r2 && d2 > 1e-9 {
                    let im_j = if j as i32 == self.held { 0.0 } else { 1.0 };
                    let im_sum = im_i + im_j;
                    if im_sum == 0.0 {
                        continue; // both held: nothing movable
                    }
                    let dist = d2.sqrt();
                    let inv = 1.0 / dist;
                    let nx = dx * inv;
                    let ny = dy * inv;
                    let nz = dz * inv;

                    // Positional correction: split the overlap by inverse mass.
                    let corr = (two_r - dist) / im_sum;
                    self.pos[ki] += nx * corr * im_i;
                    self.pos[ki + 1] += ny * corr * im_i;
                    self.pos[ki + 2] += nz * corr * im_i;
                    self.pos[kj] -= nx * corr * im_j;
                    self.pos[kj + 1] -= ny * corr * im_j;
                    self.pos[kj + 2] -= nz * corr * im_j;

                    // Impulse along the contact normal (only if approaching).
                    let rvx = self.vel[ki] - self.vel[kj];
                    let rvy = self.vel[ki + 1] - self.vel[kj + 1];
                    let rvz = self.vel[ki + 2] - self.vel[kj + 2];
                    let vrel = rvx * nx + rvy * ny + rvz * nz;
                    if vrel < 0.0 {
                        let jn = -(1.0 + e_ball) * vrel / im_sum;
                        self.vel[ki] += jn * nx * im_i;
                        self.vel[ki + 1] += jn * ny * im_i;
                        self.vel[ki + 2] += jn * nz * im_i;
                        self.vel[kj] -= jn * nx * im_j;
                        self.vel[kj + 1] -= jn * ny * im_j;
                        self.vel[kj + 2] -= jn * nz * im_j;
                    }
                }
            }
        }

        // 3) Box walls (closed box: floor, ceiling, 4 sides). The grabbed body is
        //    clamped on the JS side, so skip it here.
        for i in 0..n {
            if i as i32 == self.held {
                continue;
            }
            let k = i * 3;
            if self.pos[k + 1] < r {
                self.pos[k + 1] = r;
                self.vel[k + 1] = -self.vel[k + 1] * e_wall;
            }
            if self.pos[k + 1] > 2.0 * b - r {
                self.pos[k + 1] = 2.0 * b - r;
                self.vel[k + 1] = -self.vel[k + 1] * e_wall;
            }
            if self.pos[k] < -b + r {
                self.pos[k] = -b + r;
                self.vel[k] = -self.vel[k] * e_wall;
            }
            if self.pos[k] > b - r {
                self.pos[k] = b - r;
                self.vel[k] = -self.vel[k] * e_wall;
            }
            if self.pos[k + 2] < -b + r {
                self.pos[k + 2] = -b + r;
                self.vel[k + 2] = -self.vel[k + 2] * e_wall;
            }
            if self.pos[k + 2] > b - r {
                self.pos[k + 2] = b - r;
                self.vel[k + 2] = -self.vel[k + 2] * e_wall;
            }
        }
    }
    fn positions(&self) -> &[f32] {
        &self.pos
    }
    fn reset(&mut self) {
        *self = Bouncer::new(self.n);
    }
    fn count(&self) -> usize {
        self.n
    }
    fn radius(&self) -> f32 {
        self.radius
    }
    fn bounds(&self) -> f32 {
        self.bounds
    }
    fn set_held(&mut self, i: i32) {
        self.held = i;
    }
    fn set_pos(&mut self, i: usize, x: f32, y: f32, z: f32) {
        let k = i * 3;
        if k + 2 < self.pos.len() {
            self.pos[k] = x;
            self.pos[k + 1] = y;
            self.pos[k + 2] = z;
        }
    }
    fn set_vel(&mut self, i: usize, x: f32, y: f32, z: f32) {
        let k = i * 3;
        if k + 2 < self.vel.len() {
            self.vel[k] = x;
            self.vel[k + 1] = y;
            self.vel[k + 2] = z;
        }
    }
}

/// The single concrete type that crosses the wasm-bindgen boundary. It owns a
/// trait object so JS only ever talks to `World`, while Rust swaps the sim behind it.
#[wasm_bindgen]
pub struct World {
    sim: Box<dyn Simulation>,
}

#[wasm_bindgen]
impl World {
    #[wasm_bindgen(constructor)]
    pub fn new(kind: u32, n: usize) -> World {
        console_error_panic_hook::set_once();
        let sim: Box<dyn Simulation> = match kind {
            // Phase 1+ will add: 1 => NBody, 2 => Particles, 3 => Pendulum, 4 => Cloth
            _ => Box::new(Bouncer::new(n)),
        };
        World { sim }
    }

    pub fn step(&mut self, dt: f32) {
        self.sim.step(dt);
    }

    /// Pointer to the first f32 of the position buffer in WASM linear memory.
    pub fn positions_ptr(&self) -> *const f32 {
        self.sim.positions().as_ptr()
    }

    /// Number of f32 values (3 per body).
    pub fn len(&self) -> usize {
        self.sim.positions().len()
    }

    pub fn count(&self) -> usize {
        self.sim.count()
    }

    pub fn radius(&self) -> f32 {
        self.sim.radius()
    }

    pub fn bounds(&self) -> f32 {
        self.sim.bounds()
    }

    pub fn reset(&mut self) {
        self.sim.reset();
    }

    /// Grab body `i` with the mouse (-1 releases). While held it is moved by JS
    /// via `set_pos` and acts as an immovable obstacle for the other bodies.
    pub fn set_held(&mut self, i: i32) {
        self.sim.set_held(i);
    }

    /// Teleport body `i` to (x, y, z) — used while dragging a grabbed body.
    pub fn set_pos(&mut self, i: usize, x: f32, y: f32, z: f32) {
        self.sim.set_pos(i, x, y, z);
    }

    /// Set the velocity of body `i` — used to fling a body on release.
    pub fn set_vel(&mut self, i: usize, x: f32, y: f32, z: f32) {
        self.sim.set_vel(i, x, y, z);
    }
}
