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
            // Phase 4 will add: 3 => Cloth
            1 => Box::new(NBody::new(n)),
            2 => Box::new(DoublePendulum::new(n)),
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
