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
    /// Optional scalar field sampled on a square grid (row-major), used by the
    /// gradient-descent sim to paint a loss-landscape heatmap. Empty otherwise.
    fn grid(&self) -> &[f32] {
        &[]
    }
    /// Raw (min, max) of the `grid` field, so JS can normalize the heatmap.
    fn loss_range(&self) -> (f32, f32) {
        (0.0, 1.0)
    }
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

/// Sim #0: spheres colliding inside a closed box. Sphere-sphere elastic
/// collisions and wall bounces, accelerated by a uniform-grid broad-phase so the
/// body count scales to thousands. Gravity is tunable (0 = a weightless gas).
struct Bouncer {
    pos: Vec<f32>,
    vel: Vec<f32>,
    speed: Vec<f32>, // per-body |v|, exposed via `extra()` for color
    n: usize,
    bounds: f32,
    radius: f32,
    gravity: f32, // downward accel (m/s^2); set_param scales Earth gravity
    /// Index of the body currently grabbed by the mouse, or -1 when none.
    held: i32,
    // --- uniform-grid broad-phase (rebuilt each step; no per-step alloc) ---
    cell: f32,            // cell edge = collision diameter (2 * radius)
    gdim: usize,          // cells per axis (box is gdim^3 cells)
    cell_start: Vec<u32>, // CSR offsets into `bucket`, len gdim^3 + 1
    cursor: Vec<u32>,     // per-cell scatter cursor, len gdim^3
    bucket: Vec<u32>,     // body indices grouped by cell, len n
}

impl Bouncer {
    fn new(n: usize) -> Self {
        let n = n.max(1);
        let bounds = 4.0f32;
        // Shrink the balls as the box fills so packing stays sane (~0.4 at the
        // 200-body default, smaller for crowds, never microscopic).
        let radius = (0.4 * (200.0 / n as f32).cbrt()).clamp(0.1, 0.7);
        let mut pos = vec![0.0f32; n * 3];
        let mut vel = vec![0.0f32; n * 3];
        let speed = vec![0.0f32; n];
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
        // Cell edge = collision diameter, so any overlapping pair lands in the
        // same or an adjacent cell -> a 3x3x3 neighborhood test suffices.
        let cell = 2.0 * radius;
        let gdim = ((2.0 * bounds / cell).ceil() as usize).max(1);
        let ncells = gdim * gdim * gdim;
        Self {
            pos,
            vel,
            speed,
            n,
            bounds,
            radius,
            gravity: 9.81,
            held: -1,
            cell,
            gdim,
            cell_start: vec![0; ncells + 1],
            cursor: vec![0; ncells],
            bucket: vec![0; n],
        }
    }

    /// Grid cell index containing body `i` (clamped into the box).
    #[inline]
    fn cell_of(&self, i: usize) -> usize {
        let k = i * 3;
        let b = self.bounds;
        let inv = 1.0 / self.cell;
        let g = self.gdim as i32;
        // x,z span [-b, b]; y spans [0, 2b] (floor at 0).
        let cx = (((self.pos[k] + b) * inv) as i32).clamp(0, g - 1);
        let cy = ((self.pos[k + 1] * inv) as i32).clamp(0, g - 1);
        let cz = (((self.pos[k + 2] + b) * inv) as i32).clamp(0, g - 1);
        ((cz * g + cy) * g + cx) as usize
    }

    /// Counting-sort the bodies into grid cells: fills `cell_start` (CSR offsets)
    /// and `bucket` (body indices grouped by cell). Allocation-free.
    fn rebuild_grid(&mut self) {
        let ncells = self.cell_start.len() - 1;
        for c in self.cell_start.iter_mut() {
            *c = 0;
        }
        // Tally counts into cell_start[cell + 1].
        for i in 0..self.n {
            let c = self.cell_of(i);
            self.cell_start[c + 1] += 1;
        }
        // Prefix-sum: counts -> start offsets.
        for c in 1..=ncells {
            self.cell_start[c] += self.cell_start[c - 1];
        }
        // Scatter body indices into each cell's slice.
        self.cursor.copy_from_slice(&self.cell_start[..ncells]);
        for i in 0..self.n {
            let c = self.cell_of(i);
            let slot = self.cursor[c] as usize;
            self.bucket[slot] = i as u32;
            self.cursor[c] += 1;
        }
    }
}

impl Simulation for Bouncer {
    fn step(&mut self, dt: f32) {
        let g = -self.gravity;
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

        // 2) Ball-ball elastic collisions via a uniform-grid broad-phase: bucket
        //    the bodies into a grid (cell edge = 2r), then test each body only
        //    against the 27 cells around it. This turns the old O(n^2) all-pairs
        //    sweep into ~O(n) for evenly spread bodies. Two relaxation passes
        //    settle dense stacks; the grid is rebuilt once (corrections are tiny).
        //    Equal masses (inverse mass 1), except a grabbed body has inverse
        //    mass 0 so it acts as an immovable obstacle that shoves others aside.
        self.rebuild_grid();
        let two_r = 2.0 * r;
        let two_r2 = two_r * two_r;
        let gd = self.gdim as i32;
        for _pass in 0..2 {
            for cz in 0..gd {
                for cy in 0..gd {
                    for cx in 0..gd {
                        let home = ((cz * gd + cy) * gd + cx) as usize;
                        let hs = self.cell_start[home] as usize;
                        let he = self.cell_start[home + 1] as usize;
                        for hi in hs..he {
                            let i = self.bucket[hi] as usize;
                            let ki = i * 3;
                            let im_i = if i as i32 == self.held { 0.0 } else { 1.0 };
                            // Scan the 3x3x3 neighborhood (clamped to the box).
                            for dz in -1..=1 {
                                let nz = cz + dz;
                                if nz < 0 || nz >= gd {
                                    continue;
                                }
                                for dy in -1..=1 {
                                    let ny = cy + dy;
                                    if ny < 0 || ny >= gd {
                                        continue;
                                    }
                                    for dx in -1..=1 {
                                        let nx = cx + dx;
                                        if nx < 0 || nx >= gd {
                                            continue;
                                        }
                                        let nb = ((nz * gd + ny) * gd + nx) as usize;
                                        let ns = self.cell_start[nb] as usize;
                                        let ne = self.cell_start[nb + 1] as usize;
                                        for nj in ns..ne {
                                            let j = self.bucket[nj] as usize;
                                            if j <= i {
                                                continue; // dedupe each pair + skip self
                                            }
                                            let kj = j * 3;
                                            let ex = self.pos[ki] - self.pos[kj];
                                            let ey = self.pos[ki + 1] - self.pos[kj + 1];
                                            let ez = self.pos[ki + 2] - self.pos[kj + 2];
                                            let d2 = ex * ex + ey * ey + ez * ez;
                                            if d2 >= two_r2 || d2 <= 1e-9 {
                                                continue;
                                            }
                                            let im_j =
                                                if j as i32 == self.held { 0.0 } else { 1.0 };
                                            let im_sum = im_i + im_j;
                                            if im_sum == 0.0 {
                                                continue; // both held: nothing movable
                                            }
                                            let dist = d2.sqrt();
                                            let inv = 1.0 / dist;
                                            let ux = ex * inv;
                                            let uy = ey * inv;
                                            let uz = ez * inv;

                                            // Positional correction split by inverse mass.
                                            let corr = (two_r - dist) / im_sum;
                                            self.pos[ki] += ux * corr * im_i;
                                            self.pos[ki + 1] += uy * corr * im_i;
                                            self.pos[ki + 2] += uz * corr * im_i;
                                            self.pos[kj] -= ux * corr * im_j;
                                            self.pos[kj + 1] -= uy * corr * im_j;
                                            self.pos[kj + 2] -= uz * corr * im_j;

                                            // Impulse along the normal (only if approaching).
                                            let rvx = self.vel[ki] - self.vel[kj];
                                            let rvy = self.vel[ki + 1] - self.vel[kj + 1];
                                            let rvz = self.vel[ki + 2] - self.vel[kj + 2];
                                            let vrel = rvx * ux + rvy * uy + rvz * uz;
                                            if vrel < 0.0 {
                                                let jn = -(1.0 + e_ball) * vrel / im_sum;
                                                self.vel[ki] += jn * ux * im_i;
                                                self.vel[ki + 1] += jn * uy * im_i;
                                                self.vel[ki + 2] += jn * uz * im_i;
                                                self.vel[kj] -= jn * ux * im_j;
                                                self.vel[kj + 1] -= jn * uy * im_j;
                                                self.vel[kj + 2] -= jn * uz * im_j;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3) Box walls (closed box: floor, ceiling, 4 sides). The grabbed body is
        //    clamped on the JS side, so it only gets its speed cached. Cache
        //    per-body |v| here for the renderer's speed coloring.
        for i in 0..n {
            let k = i * 3;
            if i as i32 != self.held {
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
    fn extra(&self) -> &[f32] {
        &self.speed
    }
    fn set_param(&mut self, id: u32, v: f32) {
        if id == 0 {
            // v is a gravity multiplier (1.0 = Earth gravity, 0 = weightless).
            self.gravity = 9.81 * v.max(0.0);
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

/// Sim #2: double pendulums (chaos). Each pendulum is two point masses on rigid
/// massless rods swinging in a plane. The motion follows the standard Lagrangian
/// equations of motion, integrated with RK4 (4th order, low energy drift). Many
/// pendulums launch from nearly-identical angles; their exponential divergence
/// (sensitive dependence on initial conditions) shows as the colored arms fan out.
struct DoublePendulum {
    th1: Vec<f32>, // upper-arm angle from the downward vertical
    th2: Vec<f32>, // lower-arm angle from the downward vertical
    w1: Vec<f32>,  // angular velocities
    w2: Vec<f32>,
    pos: Vec<f32>, // [bob1, bob2] world coords per pendulum, len 2*p*3
    p: usize,      // number of pendulums
    m1: f32,
    m2: f32,
    l1: f32,
    l2: f32,
    g: f32,
}

impl DoublePendulum {
    fn new(p: usize) -> Self {
        let p = p.max(1);
        let mut th1 = vec![0.0f32; p];
        let mut th2 = vec![0.0f32; p];
        let w1 = vec![0.0f32; p];
        let w2 = vec![0.0f32; p];
        // Launch from a near-identical high angle; a tiny per-pendulum offset on
        // theta2 (~0.11 deg steps) is all chaos needs to fan them out over time.
        let base = 2.3f32; // ~132 deg from straight down -> energetic, flips over
        for i in 0..p {
            th1[i] = base;
            th2[i] = base + i as f32 * 0.002;
        }
        let mut s = Self {
            th1,
            th2,
            w1,
            w2,
            pos: vec![0.0f32; p * 6],
            p,
            m1: 1.0,
            m2: 1.0,
            l1: 1.0,
            l2: 1.0,
            g: 9.81,
        };
        s.sync_pos();
        s
    }

    /// Angular accelerations (alpha1, alpha2) for one pendulum's state. Standard
    /// double-pendulum EoM (angles from the downward vertical).
    #[inline]
    fn accel(&self, th1: f32, th2: f32, w1: f32, w2: f32) -> (f32, f32) {
        let (m1, m2, l1, l2, g) = (self.m1, self.m2, self.l1, self.l2, self.g);
        let d = th1 - th2;
        let sd = d.sin();
        let cd = d.cos();
        let denom = 2.0 * m1 + m2 - m2 * (2.0 * th1 - 2.0 * th2).cos();

        let num1 = -g * (2.0 * m1 + m2) * th1.sin()
            - m2 * g * (th1 - 2.0 * th2).sin()
            - 2.0 * sd * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * cd);
        let a1 = num1 / (l1 * denom);

        let num2 = 2.0
            * sd
            * (w1 * w1 * l1 * (m1 + m2) + g * (m1 + m2) * th1.cos() + w2 * w2 * l2 * m2 * cd);
        let a2 = num2 / (l2 * denom);
        (a1, a2)
    }

    /// One classical RK4 step on a single pendulum's 4D state (th1, th2, w1, w2).
    #[inline]
    fn rk4(&self, th1: f32, th2: f32, w1: f32, w2: f32, dt: f32) -> (f32, f32, f32, f32) {
        // y' = (w1, w2, a1, a2)
        let (a1, a2) = self.accel(th1, th2, w1, w2);
        let (k1t1, k1t2, k1w1, k1w2) = (w1, w2, a1, a2);

        let (a1, a2) = self.accel(
            th1 + 0.5 * dt * k1t1,
            th2 + 0.5 * dt * k1t2,
            w1 + 0.5 * dt * k1w1,
            w2 + 0.5 * dt * k1w2,
        );
        let (k2t1, k2t2, k2w1, k2w2) = (w1 + 0.5 * dt * k1w1, w2 + 0.5 * dt * k1w2, a1, a2);

        let (a1, a2) = self.accel(
            th1 + 0.5 * dt * k2t1,
            th2 + 0.5 * dt * k2t2,
            w1 + 0.5 * dt * k2w1,
            w2 + 0.5 * dt * k2w2,
        );
        let (k3t1, k3t2, k3w1, k3w2) = (w1 + 0.5 * dt * k2w1, w2 + 0.5 * dt * k2w2, a1, a2);

        let (a1, a2) = self.accel(
            th1 + dt * k3t1,
            th2 + dt * k3t2,
            w1 + dt * k3w1,
            w2 + dt * k3w2,
        );
        let (k4t1, k4t2, k4w1, k4w2) = (w1 + dt * k3w1, w2 + dt * k3w2, a1, a2);

        let s = dt / 6.0;
        (
            th1 + s * (k1t1 + 2.0 * k2t1 + 2.0 * k3t1 + k4t1),
            th2 + s * (k1t2 + 2.0 * k2t2 + 2.0 * k3t2 + k4t2),
            w1 + s * (k1w1 + 2.0 * k2w1 + 2.0 * k3w1 + k4w1),
            w2 + s * (k1w2 + 2.0 * k2w2 + 2.0 * k3w2 + k4w2),
        )
    }

    /// Recompute Cartesian bob positions (in the xy-plane, z = 0) from angles.
    fn sync_pos(&mut self) {
        for i in 0..self.p {
            let (s1, c1) = self.th1[i].sin_cos();
            let (s2, c2) = self.th2[i].sin_cos();
            let x1 = self.l1 * s1;
            let y1 = -self.l1 * c1;
            let x2 = x1 + self.l2 * s2;
            let y2 = y1 - self.l2 * c2;
            let k = i * 6;
            self.pos[k] = x1;
            self.pos[k + 1] = y1;
            self.pos[k + 2] = 0.0;
            self.pos[k + 3] = x2;
            self.pos[k + 4] = y2;
            self.pos[k + 5] = 0.0;
        }
    }
}

impl Simulation for DoublePendulum {
    fn step(&mut self, dt: f32) {
        for i in 0..self.p {
            let (t1, t2, v1, v2) = self.rk4(self.th1[i], self.th2[i], self.w1[i], self.w2[i], dt);
            self.th1[i] = t1;
            self.th2[i] = t2;
            self.w1[i] = v1;
            self.w2[i] = v2;
        }
        self.sync_pos();
    }
    fn positions(&self) -> &[f32] {
        &self.pos
    }
    fn reset(&mut self) {
        *self = DoublePendulum::new(self.p);
    }
    fn count(&self) -> usize {
        self.p * 2 // two bobs per pendulum
    }
    fn radius(&self) -> f32 {
        0.09
    }
    fn bounds(&self) -> f32 {
        self.l1 + self.l2
    }
    fn set_param(&mut self, id: u32, v: f32) {
        if id == 0 {
            self.g = v.max(0.0); // gravitational acceleration
        }
    }
}

/// Sim #3: a mass-spring cloth (soft body). A grid of point masses is linked by
/// three families of distance constraints — structural (axis neighbours), shear
/// (cell diagonals) and bend (2-apart) — that together resist stretch, shear and
/// folding. Motion uses Verlet integration (velocity is implicit in the position
/// history) with Jakobsen position-based constraint relaxation: a few iterations
/// per step nudge each constrained pair back toward its rest length. The top edge
/// (or just the two top corners) is pinned, and an oscillating wind pushes the
/// sheet along +z so it billows.
struct Cloth {
    pos: Vec<f32>,        // current positions, len r*r*3
    prev: Vec<f32>,       // previous positions (Verlet history), len r*r*3
    pinned: Vec<bool>,    // per-particle pin flag, len r*r
    edges_a: Vec<u32>,    // constraint endpoint A
    edges_b: Vec<u32>,    // constraint endpoint B
    edges_rest: Vec<f32>, // constraint rest length
    r: usize,             // grid resolution (r x r particles)
    g: f32,               // gravity
    wind: f32,            // wind strength along +z
    pin_mode: u32,        // 0 = two top corners, 1 = whole top edge
    t: f32,               // elapsed time (drives the wind oscillation)
}

impl Cloth {
    fn new(r: usize) -> Self {
        let r = r.clamp(2, 200);
        let size = 6.0f32;
        let spacing = size / (r - 1) as f32;
        let top_y = 3.0f32;
        let mut pos = vec![0.0f32; r * r * 3];
        // Lay the grid in the xy-plane: row 0 is the top, columns run left->right,
        // centred on x = 0 and hanging down from top_y.
        for row in 0..r {
            for col in 0..r {
                let k = (row * r + col) * 3;
                pos[k] = -size * 0.5 + col as f32 * spacing;
                pos[k + 1] = top_y - row as f32 * spacing;
                pos[k + 2] = 0.0;
            }
        }
        let prev = pos.clone();
        let pinned = vec![false; r * r];
        let mut s = Self {
            pos,
            prev,
            pinned,
            edges_a: Vec::new(),
            edges_b: Vec::new(),
            edges_rest: Vec::new(),
            r,
            g: 9.81,
            wind: 6.0,
            pin_mode: 0,
            t: 0.0,
        };
        s.build_edges();
        s.apply_pins();
        s
    }

    #[inline]
    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.r + col
    }

    /// Record one distance constraint, capturing the current separation as its
    /// rest length (the grid starts unstretched).
    fn add_edge(&mut self, a: usize, b: usize) {
        let ka = a * 3;
        let kb = b * 3;
        let dx = self.pos[ka] - self.pos[kb];
        let dy = self.pos[ka + 1] - self.pos[kb + 1];
        let dz = self.pos[ka + 2] - self.pos[kb + 2];
        let rest = (dx * dx + dy * dy + dz * dz).sqrt();
        self.edges_a.push(a as u32);
        self.edges_b.push(b as u32);
        self.edges_rest.push(rest);
    }

    /// Build structural (axis neighbours), shear (cell diagonals) and bend
    /// (skip-one neighbours) constraints over the grid.
    fn build_edges(&mut self) {
        let r = self.r;
        for row in 0..r {
            for col in 0..r {
                let i = self.idx(row, col);
                // Structural: right and down neighbours.
                if col + 1 < r {
                    let j = self.idx(row, col + 1);
                    self.add_edge(i, j);
                }
                if row + 1 < r {
                    let j = self.idx(row + 1, col);
                    self.add_edge(i, j);
                }
                // Shear: both diagonals of the cell to the lower-right.
                if col + 1 < r && row + 1 < r {
                    let br = self.idx(row + 1, col + 1);
                    self.add_edge(i, br);
                    let tr = self.idx(row, col + 1);
                    let bl = self.idx(row + 1, col);
                    self.add_edge(tr, bl);
                }
                // Bend: skip-one neighbours stiffen the sheet against folding.
                if col + 2 < r {
                    let j = self.idx(row, col + 2);
                    self.add_edge(i, j);
                }
                if row + 2 < r {
                    let j = self.idx(row + 2, col);
                    self.add_edge(i, j);
                }
            }
        }
    }

    /// Recompute which particles are pinned from the current `pin_mode`. Row 0 is
    /// the top, so its flat indices are simply `0..r`.
    fn apply_pins(&mut self) {
        for p in self.pinned.iter_mut() {
            *p = false;
        }
        let r = self.r;
        if self.pin_mode == 1 {
            for col in 0..r {
                self.pinned[col] = true; // whole top edge
            }
        } else {
            self.pinned[0] = true; // top-left corner
            self.pinned[r - 1] = true; // top-right corner
        }
    }
}

impl Simulation for Cloth {
    fn step(&mut self, dt: f32) {
        self.t += dt;
        let damping = 0.99f32;
        let g = self.g;
        let n = self.r * self.r;
        let dt2 = dt * dt;
        // Wind gusts along +z: a global oscillation, modulated per-particle by x
        // and time so the sheet ripples instead of translating rigidly.
        let wind_base = self.wind * (0.6 + 0.4 * (self.t * 1.7).sin());

        // 1) Verlet integrate every free particle:
        //    x' = x + (x - x_prev)*damping + a*dt^2.  prev <- x.
        for i in 0..n {
            if self.pinned[i] {
                continue;
            }
            let k = i * 3;
            let px = self.pos[k];
            let py = self.pos[k + 1];
            let pz = self.pos[k + 2];
            let az = wind_base * (0.7 + 0.3 * (px * 1.3 + self.t * 2.0).sin());
            let vx = (px - self.prev[k]) * damping;
            let vy = (py - self.prev[k + 1]) * damping;
            let vz = (pz - self.prev[k + 2]) * damping;
            self.pos[k] = px + vx;
            self.pos[k + 1] = py + vy - g * dt2;
            self.pos[k + 2] = pz + vz + az * dt2;
            self.prev[k] = px;
            self.prev[k + 1] = py;
            self.prev[k + 2] = pz;
        }

        // 2) Jakobsen constraint relaxation: pull each pair back toward its rest
        //    length over a few iterations (approximates a stiff solve). A pinned
        //    end has infinite mass; with one pinned end the free end takes the
        //    whole correction, otherwise the pair splits it evenly.
        let iters = 5;
        let ne = self.edges_a.len();
        for _ in 0..iters {
            for e in 0..ne {
                let a = self.edges_a[e] as usize;
                let b = self.edges_b[e] as usize;
                let pa = self.pinned[a];
                let pb = self.pinned[b];
                if pa && pb {
                    continue;
                }
                let rest = self.edges_rest[e];
                let ka = a * 3;
                let kb = b * 3;
                let dx = self.pos[kb] - self.pos[ka];
                let dy = self.pos[kb + 1] - self.pos[ka + 1];
                let dz = self.pos[kb + 2] - self.pos[ka + 2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 <= 1e-12 {
                    continue;
                }
                let d = d2.sqrt();
                let diff = (d - rest) / d; // signed fractional stretch
                if pa {
                    self.pos[kb] -= dx * diff;
                    self.pos[kb + 1] -= dy * diff;
                    self.pos[kb + 2] -= dz * diff;
                } else if pb {
                    self.pos[ka] += dx * diff;
                    self.pos[ka + 1] += dy * diff;
                    self.pos[ka + 2] += dz * diff;
                } else {
                    let h = 0.5 * diff;
                    self.pos[ka] += dx * h;
                    self.pos[ka + 1] += dy * h;
                    self.pos[ka + 2] += dz * h;
                    self.pos[kb] -= dx * h;
                    self.pos[kb + 1] -= dy * h;
                    self.pos[kb + 2] -= dz * h;
                }
            }
        }
    }
    fn positions(&self) -> &[f32] {
        &self.pos
    }
    fn reset(&mut self) {
        *self = Cloth::new(self.r);
    }
    fn count(&self) -> usize {
        self.r * self.r
    }
    fn radius(&self) -> f32 {
        0.05
    }
    fn bounds(&self) -> f32 {
        6.0
    }
    fn set_param(&mut self, id: u32, v: f32) {
        match id {
            0 => self.g = v.max(0.0),  // gravity
            1 => self.wind = v.max(0.0), // wind strength
            2 => {
                self.pin_mode = if v >= 0.5 { 1 } else { 0 };
                self.apply_pins();
            }
            _ => {}
        }
    }
}

/// Resolution of the heatmap grid (GRID_N x GRID_N samples of the loss field).
const GRID_N: usize = 128;
/// World half-extent the loss-landscape domain is mapped onto (the contour plane
/// spans [-WORLD_HALF, WORLD_HALF] on both axes regardless of the math domain).
const WORLD_HALF: f32 = 5.0;

/// A 2D loss landscape with an analytic value and gradient. These are the
/// classic optimizer test functions; each carries a square viewing window
/// (`domain`) chosen to frame its interesting structure and minima.
#[derive(Clone, Copy)]
enum Landscape {
    Bowl,       // x^2 + y^2                     — convex, min at origin
    Saddle,     // x^2 - y^2                      — a saddle, no minimum
    Rosenbrock, // (1-x)^2 + 100(y-x^2)^2         — banana valley, min at (1,1)
    Himmelblau, // (x^2+y-11)^2 + (x+y^2-7)^2     — four equal minima
    Rastrigin,  // 20 + Σ(xi^2 - 10cos(2π xi))    — many local minima, global at origin
}

impl Landscape {
    fn from_u32(v: u32) -> Self {
        match v {
            1 => Landscape::Saddle,
            2 => Landscape::Rosenbrock,
            3 => Landscape::Himmelblau,
            4 => Landscape::Rastrigin,
            _ => Landscape::Bowl,
        }
    }

    fn f(&self, x: f32, y: f32) -> f32 {
        match self {
            Landscape::Bowl => x * x + y * y,
            Landscape::Saddle => x * x - y * y,
            Landscape::Rosenbrock => {
                let a = 1.0 - x;
                let b = y - x * x;
                a * a + 100.0 * b * b
            }
            Landscape::Himmelblau => {
                let a = x * x + y - 11.0;
                let b = x + y * y - 7.0;
                a * a + b * b
            }
            Landscape::Rastrigin => {
                let pi2 = 2.0 * std::f32::consts::PI;
                20.0 + (x * x - 10.0 * (pi2 * x).cos()) + (y * y - 10.0 * (pi2 * y).cos())
            }
        }
    }

    /// Exact gradient (∂f/∂x, ∂f/∂y) — no finite differences, so even stiff
    /// landscapes step cleanly.
    fn grad(&self, x: f32, y: f32) -> (f32, f32) {
        match self {
            Landscape::Bowl => (2.0 * x, 2.0 * y),
            Landscape::Saddle => (2.0 * x, -2.0 * y),
            Landscape::Rosenbrock => {
                let dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
                let dy = 200.0 * (y - x * x);
                (dx, dy)
            }
            Landscape::Himmelblau => {
                let a = x * x + y - 11.0;
                let b = x + y * y - 7.0;
                let dx = 4.0 * x * a + 2.0 * b;
                let dy = 2.0 * a + 4.0 * y * b;
                (dx, dy)
            }
            Landscape::Rastrigin => {
                let pi2 = 2.0 * std::f32::consts::PI;
                let dx = 2.0 * x + 10.0 * pi2 * (pi2 * x).sin();
                let dy = 2.0 * y + 10.0 * pi2 * (pi2 * y).sin();
                (dx, dy)
            }
        }
    }

    /// (center_x, center_y, half) — the square window in math-space that the
    /// contour is sampled over and that walkers are clamped into.
    fn domain(&self) -> (f32, f32, f32) {
        match self {
            Landscape::Bowl => (0.0, 0.0, 3.0),
            Landscape::Saddle => (0.0, 0.0, 3.0),
            Landscape::Rosenbrock => (0.0, 1.0, 2.5),
            Landscape::Himmelblau => (0.0, 0.0, 5.0),
            Landscape::Rastrigin => (0.0, 0.0, 5.12),
        }
    }
}

/// A first-order optimizer. SGD/Momentum/RMSProp/Adam share the per-walker
/// moment accumulators on `GradientDescent`; the unused ones just stay zero.
#[derive(Clone, Copy)]
enum Opt {
    Sgd,
    Momentum,
    RmsProp,
    Adam,
}

impl Opt {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => Opt::Sgd,
            1 => Opt::Momentum,
            2 => Opt::RmsProp,
            _ => Opt::Adam,
        }
    }
}

/// Sim #4: gradient-descent optimizer race over a 2D loss landscape. A swarm of
/// "walkers" each run the same optimizer from different starts and descend the
/// surface; the surface itself is sampled into a `GRID_N x GRID_N` heatmap that
/// JS paints as a top-down contour. Positions are emitted in world space (the
/// math domain is mapped onto the XZ-plane at y=0) so the renderer stays generic.
struct GradientDescent {
    wx: Vec<f32>,  // walker x in math-space, len k
    wy: Vec<f32>,  // walker y in math-space, len k
    m_x: Vec<f32>, // 1st moment / momentum accumulator, len k
    m_y: Vec<f32>,
    v_x: Vec<f32>, // 2nd moment accumulator (RMSProp/Adam), len k
    v_y: Vec<f32>,
    t_step: u32,        // global step counter, drives Adam bias correction
    pos: Vec<f32>,      // world positions [x,0,z, ...], len k*3 (zero-copy to JS)
    loss: Vec<f32>,     // per-walker f value, len k (exposed via extra())
    grid_buf: Vec<f32>, // GRID_N*GRID_N samples of f, row-major (y outer, x inner)
    loss_min: f32,      // raw min over the grid (heatmap normalization)
    loss_max: f32,      // raw max over the grid
    k: usize,           // walker count
    landscape: Landscape,
    opt: Opt,
    lr: f32,   // learning rate
    beta: f32, // momentum coefficient / Adam beta1
    seed: u32, // RNG seed — identical across optimizers for a fair race
}

impl GradientDescent {
    fn new(n: usize) -> Self {
        let k = n.clamp(1, 4096);
        let mut s = Self {
            wx: vec![0.0; k],
            wy: vec![0.0; k],
            m_x: vec![0.0; k],
            m_y: vec![0.0; k],
            v_x: vec![0.0; k],
            v_y: vec![0.0; k],
            t_step: 0,
            pos: vec![0.0; k * 3],
            loss: vec![0.0; k],
            grid_buf: vec![0.0; GRID_N * GRID_N],
            loss_min: 0.0,
            loss_max: 1.0,
            k,
            landscape: Landscape::Bowl,
            opt: Opt::Adam,
            lr: 0.05,
            beta: 0.9,
            seed: 0x9e37_79b9,
        };
        s.rebuild_grid();
        s.seed();
        s
    }

    /// Map a math-space point into world space. The domain is centred on the
    /// world origin and scaled to ±WORLD_HALF, so a single `half` keeps the
    /// mapping square (no aspect distortion) and walkers land on the contour.
    #[inline]
    fn to_world(&self, x: f32, y: f32) -> (f32, f32, f32) {
        let (cx, cz, half) = self.landscape.domain();
        let wx = (x - cx) / half * WORLD_HALF;
        let wz = (y - cz) / half * WORLD_HALF;
        (wx, 0.0, wz)
    }

    /// Refill `pos` (world) and `loss` from the current walker coordinates.
    fn sync_outputs(&mut self) {
        for i in 0..self.k {
            let (wx, wy, wz) = self.to_world(self.wx[i], self.wy[i]);
            let k = i * 3;
            self.pos[k] = wx;
            self.pos[k + 1] = wy;
            self.pos[k + 2] = wz;
            self.loss[i] = self.landscape.f(self.wx[i], self.wy[i]);
        }
    }

    /// Re-sample the loss field over the current landscape's domain and cache its
    /// raw range. Only called on construction and landscape change (not per step).
    fn rebuild_grid(&mut self) {
        let (cx, cz, half) = self.landscape.domain();
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        let span = 2.0 * half;
        for j in 0..GRID_N {
            let ty = j as f32 / (GRID_N - 1) as f32;
            let y = cz - half + ty * span;
            for i in 0..GRID_N {
                let tx = i as f32 / (GRID_N - 1) as f32;
                let x = cx - half + tx * span;
                let f = self.landscape.f(x, y);
                self.grid_buf[j * GRID_N + i] = f;
                if f < lo {
                    lo = f;
                }
                if f > hi {
                    hi = f;
                }
            }
        }
        self.loss_min = lo;
        self.loss_max = hi;
    }

    /// Spread walkers deterministically across ~80% of the domain, zero the
    /// optimizer accumulators and reset the step counter. The same seed is used
    /// every time so switching optimizers compares them from identical starts.
    fn seed(&mut self) {
        let (cx, cz, half) = self.landscape.domain();
        let spread = half * 0.8;
        let mut rng = Rng::new(self.seed);
        for i in 0..self.k {
            self.wx[i] = cx + rng.range(-spread, spread);
            self.wy[i] = cz + rng.range(-spread, spread);
            self.m_x[i] = 0.0;
            self.m_y[i] = 0.0;
            self.v_x[i] = 0.0;
            self.v_y[i] = 0.0;
        }
        self.t_step = 0;
        self.sync_outputs();
    }
}

impl Simulation for GradientDescent {
    fn step(&mut self, _dt: f32) {
        // One optimizer iteration per frame (gradient descent is step-indexed,
        // not time-indexed, so dt is ignored).
        self.t_step += 1;
        let t = self.t_step as f32;
        let beta1 = self.beta;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let lr = self.lr;
        // Bias-correction denominators (Adam): hoisted out of the walker loop.
        let bc1 = 1.0 - beta1.powf(t);
        let bc2 = 1.0 - beta2.powf(t);
        let (cx, cz, half) = self.landscape.domain();
        let (lo_x, hi_x) = (cx - half, cx + half);
        let (lo_y, hi_y) = (cz - half, cz + half);
        // Gradient clipping: without it the stiff landscapes (Rosenbrock,
        // Rastrigin) explode to NaN within a few steps at lr 0.05.
        let clip = 1.0e3f32;
        for i in 0..self.k {
            let (mut gx, mut gy) = self.landscape.grad(self.wx[i], self.wy[i]);
            let gn = (gx * gx + gy * gy).sqrt();
            if gn > clip {
                let s = clip / gn;
                gx *= s;
                gy *= s;
            }
            match self.opt {
                Opt::Sgd => {
                    self.wx[i] -= lr * gx;
                    self.wy[i] -= lr * gy;
                }
                Opt::Momentum => {
                    self.m_x[i] = beta1 * self.m_x[i] + gx;
                    self.m_y[i] = beta1 * self.m_y[i] + gy;
                    self.wx[i] -= lr * self.m_x[i];
                    self.wy[i] -= lr * self.m_y[i];
                }
                Opt::RmsProp => {
                    self.v_x[i] = beta2 * self.v_x[i] + (1.0 - beta2) * gx * gx;
                    self.v_y[i] = beta2 * self.v_y[i] + (1.0 - beta2) * gy * gy;
                    self.wx[i] -= lr * gx / (self.v_x[i].sqrt() + eps);
                    self.wy[i] -= lr * gy / (self.v_y[i].sqrt() + eps);
                }
                Opt::Adam => {
                    self.m_x[i] = beta1 * self.m_x[i] + (1.0 - beta1) * gx;
                    self.m_y[i] = beta1 * self.m_y[i] + (1.0 - beta1) * gy;
                    self.v_x[i] = beta2 * self.v_x[i] + (1.0 - beta2) * gx * gx;
                    self.v_y[i] = beta2 * self.v_y[i] + (1.0 - beta2) * gy * gy;
                    let mhx = self.m_x[i] / bc1;
                    let mhy = self.m_y[i] / bc1;
                    let vhx = self.v_x[i] / bc2;
                    let vhy = self.v_y[i] / bc2;
                    self.wx[i] -= lr * mhx / (vhx.sqrt() + eps);
                    self.wy[i] -= lr * mhy / (vhy.sqrt() + eps);
                }
            }
            // Keep walkers on the sampled plane (also a final NaN backstop).
            self.wx[i] = self.wx[i].clamp(lo_x, hi_x);
            self.wy[i] = self.wy[i].clamp(lo_y, hi_y);
        }
        self.sync_outputs();
    }
    fn positions(&self) -> &[f32] {
        &self.pos
    }
    fn reset(&mut self) {
        // Deliberately unlike the other sims: re-seed walkers and zero the
        // accumulators but PRESERVE the user's landscape/optimizer/lr/beta. A
        // fresh new() would snap the UI selections back to Bowl/Adam/0.05.
        self.seed();
    }
    fn count(&self) -> usize {
        self.k
    }
    fn radius(&self) -> f32 {
        0.08
    }
    fn bounds(&self) -> f32 {
        WORLD_HALF
    }
    fn extra(&self) -> &[f32] {
        &self.loss
    }
    fn grid(&self) -> &[f32] {
        &self.grid_buf
    }
    fn loss_range(&self) -> (f32, f32) {
        (self.loss_min, self.loss_max)
    }
    fn set_param(&mut self, id: u32, v: f32) {
        match id {
            0 => self.lr = v.max(0.0), // learning rate
            1 => {
                self.landscape = Landscape::from_u32(v as u32);
                self.rebuild_grid();
                self.seed();
            }
            2 => {
                self.opt = Opt::from_u32(v as u32);
                self.seed(); // fair restart from identical positions
            }
            3 => self.beta = v.clamp(0.0, 0.999), // momentum / Adam beta1
            _ => {}
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
            1 => Box::new(NBody::new(n)),
            2 => Box::new(DoublePendulum::new(n)),
            3 => Box::new(Cloth::new(n)),
            4 => Box::new(GradientDescent::new(n)),
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

    /// Pointer to the loss-landscape heatmap field (gradient-descent sim only).
    pub fn grid_ptr(&self) -> *const f32 {
        self.sim.grid().as_ptr()
    }

    /// Number of f32 samples in the heatmap (0 when the sim exposes none).
    pub fn grid_len(&self) -> usize {
        self.sim.grid().len()
    }

    /// Side length of the (square) heatmap grid, i.e. sqrt(grid_len); 0 if none.
    pub fn grid_dim(&self) -> usize {
        let n = self.sim.grid().len();
        if n == 0 {
            0
        } else {
            (n as f64).sqrt() as usize
        }
    }

    /// World half-extent the contour plane spans on each axis (sizes the plane).
    pub fn domain_extent(&self) -> f32 {
        WORLD_HALF
    }

    /// Raw minimum of the loss field (for heatmap / color normalization).
    pub fn loss_min(&self) -> f32 {
        self.sim.loss_range().0
    }

    /// Raw maximum of the loss field.
    pub fn loss_max(&self) -> f32 {
        self.sim.loss_range().1
    }
}
