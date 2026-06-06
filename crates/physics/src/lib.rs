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
    /// Optional per-body scalar (e.g. speed) the renderer maps to color.
    /// Empty when the sim doesn't expose one.
    fn extra(&self) -> &[f32] {
        &[]
    }
    /// Tunable parameter hook; the meaning of `id` is per-sim (e.g. 0 = G).
    fn set_param(&mut self, _id: u32, _v: f32) {}
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

/// Sim #1: Newtonian N-body gravity. Bodies attract each other (O(n^2)) with
/// Plummer softening to tame close encounters, advanced with velocity-Verlet
/// (symplectic, energy-stable). The preset is a rotating disk around a heavy
/// central mass, which develops spiral/clumpy structure as it evolves.
struct NBody {
    pos: Vec<f32>,
    vel: Vec<f32>,
    acc: Vec<f32>,
    acc_old: Vec<f32>,
    mass: Vec<f32>,
    speed: Vec<f32>, // per-body |v|, exposed via `extra()` for color
    n: usize,
    g: f32,
    soft2: f32,
    radius: f32,
    bounds: f32,
    held: i32,
}

impl NBody {
    fn new(n: usize) -> Self {
        let n = n.max(2);
        let g = 1.0f32;
        let soft = 0.18f32;
        let radius = 0.06f32;
        let bounds = 10.0f32;

        let mut pos = vec![0.0f32; n * 3];
        let mut vel = vec![0.0f32; n * 3];
        let acc = vec![0.0f32; n * 3];
        let acc_old = vec![0.0f32; n * 3];
        let mut mass = vec![1.0f32; n];
        let speed = vec![0.0f32; n];
        let mut rng = Rng::new(0x00C0_FFEE);

        // Heavy central body anchors the disk; its mass scales with body count
        // so orbital speeds stay reasonable as the disk gets denser.
        let m_central = 60.0 + n as f32 * 0.5;
        mass[0] = m_central; // pos/vel[0] stay at the origin, at rest

        let tau = std::f32::consts::TAU;
        for i in 1..n {
            let k = i * 3;
            let rho = rng.range(1.5, 8.0);
            let ang = rng.range(0.0, tau);
            pos[k] = rho * ang.cos();
            pos[k + 1] = rng.range(-0.25, 0.25); // thin disk in the xz-plane
            pos[k + 2] = rho * ang.sin();
            // Circular-orbit speed about the central mass; the disk's own
            // self-gravity then perturbs these orbits into structure.
            let v = (g * m_central / rho).sqrt();
            vel[k] = -v * ang.sin(); // tangent for rotation about +y
            vel[k + 1] = 0.0;
            vel[k + 2] = v * ang.cos();
        }

        let mut s = Self {
            pos,
            vel,
            acc,
            acc_old,
            mass,
            speed,
            n,
            g,
            soft2: soft * soft,
            radius,
            bounds,
            held: -1,
        };
        s.compute_acc();
        s
    }

    /// Pairwise gravitational acceleration into `self.acc` (Newton's third law
    /// halves the work). No per-step heap allocation.
    fn compute_acc(&mut self) {
        for a in self.acc.iter_mut() {
            *a = 0.0;
        }
        let n = self.n;
        let g = self.g;
        let soft2 = self.soft2;
        for i in 0..n {
            let ki = i * 3;
            let xi = self.pos[ki];
            let yi = self.pos[ki + 1];
            let zi = self.pos[ki + 2];
            let mi = self.mass[i];
            for j in (i + 1)..n {
                let kj = j * 3;
                let dx = self.pos[kj] - xi;
                let dy = self.pos[kj + 1] - yi;
                let dz = self.pos[kj + 2] - zi;
                let d2 = dx * dx + dy * dy + dz * dz + soft2;
                let inv = 1.0 / d2.sqrt();
                let inv3 = inv * inv * inv;
                let gi = g * mi * inv3; // accel coeff applied to j (from i)
                let gj = g * self.mass[j] * inv3; // accel coeff applied to i (from j)
                self.acc[ki] += gj * dx;
                self.acc[ki + 1] += gj * dy;
                self.acc[ki + 2] += gj * dz;
                self.acc[kj] -= gi * dx;
                self.acc[kj + 1] -= gi * dy;
                self.acc[kj + 2] -= gi * dz;
            }
        }
    }
}

impl Simulation for NBody {
    fn step(&mut self, dt: f32) {
        let n = self.n;
        let half_dt2 = 0.5 * dt * dt;
        // Velocity-Verlet, half 1: x += v*dt + 0.5*a*dt^2 (held body is driven by JS).
        for i in 0..n {
            if i as i32 == self.held {
                continue;
            }
            let k = i * 3;
            self.pos[k] += self.vel[k] * dt + self.acc[k] * half_dt2;
            self.pos[k + 1] += self.vel[k + 1] * dt + self.acc[k + 1] * half_dt2;
            self.pos[k + 2] += self.vel[k + 2] * dt + self.acc[k + 2] * half_dt2;
        }
        // Recompute accel at the new positions (keep the old set for the average).
        std::mem::swap(&mut self.acc, &mut self.acc_old);
        self.compute_acc();
        // Half 2: v += 0.5*(a_old + a_new)*dt, and cache speed for coloring.
        for i in 0..n {
            if i as i32 == self.held {
                continue;
            }
            let k = i * 3;
            self.vel[k] += 0.5 * (self.acc_old[k] + self.acc[k]) * dt;
            self.vel[k + 1] += 0.5 * (self.acc_old[k + 1] + self.acc[k + 1]) * dt;
            self.vel[k + 2] += 0.5 * (self.acc_old[k + 2] + self.acc[k + 2]) * dt;
            let vx = self.vel[k];
            let vy = self.vel[k + 1];
            let vz = self.vel[k + 2];
            self.speed[i] = (vx * vx + vy * vy + vz * vz).sqrt();
        }
    }
    fn positions(&self) -> &[f32] {
        &self.pos
    }
    fn reset(&mut self) {
        *self = NBody::new(self.n);
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
    fn extra(&self) -> &[f32] {
        &self.speed
    }
    fn set_param(&mut self, id: u32, v: f32) {
        if id == 0 {
            self.g = v.max(0.0);
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
            // Phase 2+ will add: 2 => Particles, 3 => Pendulum, 4 => Cloth
            1 => Box::new(NBody::new(n)),
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

    /// Pointer to the per-body scalar buffer (e.g. speed) for color mapping.
    pub fn extra_ptr(&self) -> *const f32 {
        self.sim.extra().as_ptr()
    }

    /// Length of the `extra` buffer (0 when the active sim exposes none).
    pub fn extra_len(&self) -> usize {
        self.sim.extra().len()
    }

    /// Set a tunable parameter on the active sim (id is sim-specific; 0 = G).
    pub fn set_param(&mut self, id: u32, v: f32) {
        self.sim.set_param(id, v);
    }
}
