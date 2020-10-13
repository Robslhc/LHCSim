import taichi as ti
import numpy as np
from .pressure_solver.MGPCGSolver import MGPCGSolver
from utils import clamp


@ti.data_oriented
class apic_solver:
    # cell type
    FLUID = 0
    AIR = 1
    SOLID = 2

    def __init__(self,
                 world,
                 grid_res,
                 bound,
                 pressure_solver='MGPCG',
                 max_num_particles=2**27):
        # world and boundary
        self.world = world
        self.bound = bound

        # simulation resolution
        self.n_grid = grid_res
        self.dx = self.world / self.n_grid
        self.default_dt = 1e-3

        # fluid density
        self.rho = 1000

        # particle variables
        self.n_particles = ti.field(dtype=ti.i32, shape=())
        self.x = ti.Vector.field(3, dtype=ti.f32)  # particle position
        self.pv = ti.Vector.field(3, dtype=ti.f32)  # particle velocity
        self.c_x = ti.Vector.field(3, dtype=ti.f32)
        self.c_y = ti.Vector.field(3, dtype=ti.f32)
        self.c_z = ti.Vector.field(3, dtype=ti.f32)

        # grid paramaters
        self.u = ti.field(dtype=ti.f32,
                          shape=(self.n_grid + 1, self.n_grid, self.n_grid))
        self.v = ti.field(dtype=ti.f32,
                          shape=(self.n_grid, self.n_grid + 1, self.n_grid))
        self.w = ti.field(dtype=ti.f32,
                          shape=(self.n_grid, self.n_grid, self.n_grid + 1))
        self.u_weight = ti.field(dtype=ti.f32,
                                 shape=(self.n_grid + 1, self.n_grid,
                                        self.n_grid))
        self.v_weight = ti.field(dtype=ti.f32,
                                 shape=(self.n_grid, self.n_grid + 1,
                                        self.n_grid))
        self.w_weight = ti.field(dtype=ti.f32,
                                 shape=(self.n_grid, self.n_grid,
                                        self.n_grid + 1))
        self.u_temp = ti.field(dtype=ti.f32,
                               shape=(self.n_grid + 1, self.n_grid,
                                      self.n_grid))
        self.v_temp = ti.field(dtype=ti.f32,
                               shape=(self.n_grid, self.n_grid + 1,
                                      self.n_grid))
        self.w_temp = ti.field(dtype=ti.f32,
                               shape=(self.n_grid, self.n_grid,
                                      self.n_grid + 1))
        self.cell_type = ti.field(dtype=ti.i32,
                                  shape=(self.n_grid, self.n_grid,
                                         self.n_grid))  # cell type
        self.valid = ti.field(dtype=ti.i32,
                              shape=(self.n_grid + 1, self.n_grid + 1,
                                     self.n_grid + 1))  # extrapolate utils
        self.valid_temp = ti.field(dtype=ti.i32,
                                   shape=(self.n_grid + 1, self.n_grid + 1,
                                          self.n_grid + 1))  # extapolate utils

        # pressure and solver
        self.p = ti.field(dtype=ti.f32,
                          shape=(self.n_grid, self.n_grid, self.n_grid))
        self.psolve_algo = pressure_solver
        if self.psolve_algo == 'MGPCG':
            self.p_solver = MGPCGSolver(self.n_grid, self.u, self.v, self.w,
                                        self.cell_type)

        # place particle
        self.block_particle = ti.root.pointer(ti.i, max_num_particles)
        self.block_particle.place(self.x, self.pv, self.c_x, self.c_y, self.c_z)

    def add_object(self, geometry, material, par_res=None):
        if par_res is None:
            par_res = 2

        if geometry.type == 'cube':
            param = np.array([
                geometry.x_start, geometry.x_end, geometry.y_start,
                geometry.y_end, geometry.z_start, geometry.z_end
            ])
            self.voxelize_cube(param, par_res)
        else:
            print("Geometry {} not supported!")

    @ti.kernel
    def voxelize_cube(self, geometry_param: ti.ext_arr(), par_res: ti.f32):
        x_start = geometry_param[0]
        lx = geometry_param[1] - x_start
        y_start = geometry_param[2]
        ly = geometry_param[3] - y_start
        z_start = geometry_param[4]
        lz = geometry_param[5] - z_start

        vol = lx * ly * lz
        num_new_particles = ti.cast((vol / self.dx**3) * par_res**3, ti.i32)

        bound = (self.bound + 1.0) * self.dx

        # generate particle
        for p in range(self.n_particles[None],
                       self.n_particles[None] + num_new_particles):
            px = ti.Vector.zero(ti.f32, 3)
            px.x = clamp(x_start + ti.random() * lx, bound + 1e-4,
                         self.world - bound - 1e-4)
            px.y = clamp(y_start + ti.random() * ly, bound + 1e-4,
                         self.world - bound - 1e-4)
            px.z = clamp(z_start + ti.random() * lz, bound + 1e-4,
                         self.world - bound - 1e-4)
            self.x[p] = px
            self.pv[p] = ti.Vector.zero(ti.f32, 3)
            self.c_x[p] = ti.Vector.zero(ti.f32, 3)
            self.c_y[p] = ti.Vector.zero(ti.f32, 3)
            self.c_z[p] = ti.Vector.zero(ti.f32, 3)

        self.n_particles[None] += num_new_particles

        # mark boundary
        for i, j, k in self.cell_type:
            if i < self.bound or i >= self.n_grid - self.bound or j < self.bound or j >= self.n_grid - self.bound or k < self.bound or k >= self.n_grid - self.bound:
                self.cell_type[i, j, k] = self.SOLID

    def solver_init(self):
        self.mark_cell()
        self.init_field()

    #define helper functions dealing with cell type
    @ti.func
    def is_valid(self, i, j, k):
        return i >= 0 and i < self.n_grid and j >= 0 and j < self.n_grid and k >= 0 and k < self.n_grid

    @ti.func
    def is_fluid(self, i, j, k):
        return self.is_valid(i, j, k) and self.cell_type[i, j, k] == self.FLUID

    @ti.func
    def is_air(self, i, j, k):
        return self.is_valid(i, j, k) and self.cell_type[i, j, k] == self.AIR

    @ti.func
    def is_solid(self, i, j, k):
        return self.is_valid(i, j, k) and self.cell_type[i, j, k] == self.SOLID

    @ti.kernel
    def mark_cell(self):
        for i, j, k in self.cell_type:
            if not self.is_solid(i, j, k):
                self.cell_type[i, j, k] = self.AIR

        for p in self.x:
            xp = self.x[p]
            idx = ti.cast(ti.floor(xp / self.dx), ti.i32)

            if not self.is_solid(idx[0], idx[1], idx[2]):
                self.cell_type[idx] = self.FLUID

    @ti.kernel
    def init_field(self):
        for i, j, k in self.u:
            self.u[i, j, k] = 0.0
            self.u_weight[i, j, k] = 0.0

        for i, j, k in self.v:
            self.v[i, j, k] = 0.0
            self.v_weight[i, j, k] = 0.0

        for i, j, k in self.w:
            self.w[i, j, k] = 0.0
            self.w_weight[i, j, k] = 0.0

        for i, j, k in self.p:
            self.p[i, j, k] = 0.0

    def step(self, frame_dt):
        substeps = int(frame_dt / self.default_dt) + 1
        dt = frame_dt / substeps

        for i in range(substeps):
            # P2G
            self.p2g(dt)

            # grid op
            self.apply_gravity(dt)
            self.grid_bounding_box()

            self.extrapolate_velocity()
            self.grid_bounding_box()

            self.solve_pressure(dt)
            self.apply_pressure(dt)
            self.grid_bounding_box()

            self.extrapolate_velocity()
            self.grid_bounding_box()

            # G2P
            self.g2p(dt)
            self.advect_particles(dt)
            self.mark_cell()

            self.u.fill(0.0)
            self.v.fill(0.0)
            self.w.fill(0.0)
            self.u_weight.fill(0.0)
            self.v_weight.fill(0.0)
            self.w_weight.fill(0.0)

    @ti.kernel
    def p2g(self, dt: ti.f32):
        stagger_u = ti.Vector([0, 0.5, 0.5])
        stagger_v = ti.Vector([0.5, 0, 0.5])
        stagger_w = ti.Vector([0.5, 0.5, 0])

        for p in self.x:
            xp = self.x[p]
            self.p2g1d(stagger_u, xp, self.pv[p].x, self.c_x[p], self.u, self.u_weight)
            self.p2g1d(stagger_v, xp, self.pv[p].y, self.c_y[p], self.v, self.v_weight)
            self.p2g1d(stagger_w, xp, self.pv[p].z, self.c_z[p], self.w, self.w_weight)

        # grid normalization
        for i, j, k in self.u_weight:
            if self.u_weight[i, j, k] > 0:
                self.u[i, j, k] = self.u[i, j, k] / self.u_weight[i, j, k]

        for i, j, k in self.v_weight:
            if self.v_weight[i, j, k] > 0:
                self.v[i, j, k] = self.v[i, j, k] / self.v_weight[i, j, k]

        for i, j, k in self.w_weight:
            if self.w_weight[i, j, k] > 0:
                self.w[i, j, k] = self.w[i, j, k] / self.w_weight[i, j, k]

    @ti.func
    def p2g1d(self, stagger, xp, vp, cp, grid_v, grid_m):
        inv_dx = 1 / self.dx

        # use trilinear interpolation kernel
        base = (xp * inv_dx - stagger).cast(ti.i32)

        for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
            offset = ti.Vector([i, j, k])
            fx = xp * inv_dx - (base + offset + stagger)
            dpos = fx * self.dx
            weight = (1 - abs(fx.x)) * (1 - abs(fx.y)) * (1 - abs(fx.z))
            grid_v[base + offset] += weight * (vp + cp.dot(dpos))
            grid_m[base + offset] += weight

    @ti.kernel
    def g2p(self, dt: ti.f32):
        stagger_u = ti.Vector([0, 0.5, 0.5])
        stagger_v = ti.Vector([0.5, 0, 0.5])
        stagger_w = ti.Vector([0.5, 0.5, 0])

        for p in self.x:
            xp = self.x[p]
            vp = self.pv[p]

            self.pv[p].x, self.c_x[p] = self.g2p1d(stagger_u, xp, self.u)
            self.pv[p].y, self.c_y[p] = self.g2p1d(stagger_v, xp, self.v)
            self.pv[p].z, self.c_z[p] = self.g2p1d(stagger_w, xp, self.w)

    @ti.func
    def g2p1d(self, stagger, xp, grid_v):
        inv_dx = 1 / self.dx

        # use trilinear interpolation kernel
        base = (xp * inv_dx - stagger).cast(ti.i32)
        new_v = 0.0
        new_c = ti.Vector.zero(ti.f32, 3)

        for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
            offset = ti.Vector([i, j, k])
            fx = xp * inv_dx - (base + offset + stagger)
            dpos = fx
            weight = (1 - abs(fx.x)) * (1 - abs(fx.y)) * (1 - abs(fx.z))
            new_v += weight * grid_v[base + offset]
            new_c += 4 * weight * dpos * grid_v[base + offset] * inv_dx

        return new_v, new_c

    @ti.kernel
    def apply_gravity(self, dt: ti.f32):
        for i, j, k in self.v:
            self.v[i, j, k] -= 9.8 * dt

    def solve_pressure(self, dt):
        scale_A = dt / (self.rho * self.dx**2)
        scale_b = 1 / self.dx

        self.p_solver.system_init(scale_A, scale_b)
        self.p_solver.solve(500)

        self.p.copy_from(self.p_solver.p)

    @ti.kernel
    def apply_pressure(self, dt: ti.f32):
        scale = dt / (self.rho * self.dx)

        for i, j, k in ti.ndrange(self.n_grid, self.n_grid, self.n_grid):
            if self.is_fluid(i - 1, j, k) or self.is_fluid(i, j, k):
                if self.is_solid(i - 1, j, k) or self.is_solid(i, j, k):
                    self.u[i, j, k] = 0
                else:
                    self.u[i, j, k] -= scale * (self.p[i, j, k] -
                                                self.p[i - 1, j, k])

            if self.is_fluid(i, j - 1, k) or self.is_fluid(i, j, k):
                if self.is_solid(i, j - 1, k) or self.is_solid(i, j, k):
                    self.v[i, j, k] = 0
                else:
                    self.v[i, j, k] -= scale * (self.p[i, j, k] -
                                                self.p[i, j - 1, k])

            if self.is_fluid(i, j, k - 1) or self.is_fluid(i, j, k):
                if self.is_solid(i, j, k - 1) or self.is_solid(i, j, k):
                    self.w[i, j, k] = 0
                else:
                    self.w[i, j, k] -= scale * (self.p[i, j, k] -
                                                self.p[i, j, k - 1])

    @ti.kernel
    def grid_bounding_box(self):
        for i, j, k in self.u:
            if self.is_solid(i - 1, j, k) or self.is_solid(i, j, k):
                self.u[i, j, k] = 0.0

        for i, j, k in self.v:
            if self.is_solid(i, j - 1, k) or self.is_solid(i, j, k):
                self.v[i, j, k] = 0.0

        for i, j, k in self.w:
            if self.is_solid(i, j, k - 1) or self.is_solid(i, j, k):
                self.w[i, j, k] = 0.0

    def extrapolate_velocity(self):
        extrapolate_times = 10

        # reference: https://gitee.com/citadel2020/taichi_demos/blob/master/mgpcgflip/mgpcgflip.py
        @ti.kernel
        def mark_valid_u():
            for i, j, k in self.u:
                # NOTE that the the air-liquid interface is valid
                if self.is_fluid(i - 1, j, k) or self.is_fluid(i, j, k):
                    self.valid[i, j, k] = 1
                else:
                    self.valid[i, j, k] = 0

        @ti.kernel
        def mark_valid_v():
            for i, j, k in self.v:
                # NOTE that the the air-liquid interface is valid
                if self.is_fluid(i, j - 1, k) or self.is_fluid(i, j, k):
                    self.valid[i, j, k] = 1
                else:
                    self.valid[i, j, k] = 0

        @ti.kernel
        def mark_valid_w():
            for i, j, k in self.w:
                # NOTE that the the air-liquid interface is valid
                if self.is_fluid(i, j, k - 1) or self.is_fluid(i, j, k):
                    self.valid[i, j, k] = 1
                else:
                    self.valid[i, j, k] = 0

        @ti.kernel
        def diffuse_quantity(dst: ti.template(), src: ti.template(),
                             valid_dst: ti.template(), valid: ti.template()):
            for i, j, k in dst:
                if valid[i, j, k] == 0:
                    s = 0.0
                    count = 0
                    for m, n, q in ti.static(
                            ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                        if 1 == valid[i + m, j + n, k + q]:
                            s += src[i + m, j + n, k + q]
                            count += 1
                    if count > 0:
                        dst[i, j, k] = s / float(count)
                        valid_dst[i, j, k] = 1

        mark_valid_u()
        for i in range(extrapolate_times):
            self.u_temp.copy_from(self.u)
            self.valid_temp.copy_from(self.valid)
            diffuse_quantity(self.u, self.u_temp, self.valid, self.valid_temp)

        mark_valid_v()
        for i in range(extrapolate_times):
            self.v_temp.copy_from(self.v)
            self.valid_temp.copy_from(self.valid)
            diffuse_quantity(self.v, self.v_temp, self.valid, self.valid_temp)

        mark_valid_w()
        for i in range(extrapolate_times):
            self.w_temp.copy_from(self.w)
            self.valid_temp.copy_from(self.valid)
            diffuse_quantity(self.w, self.w_temp, self.valid, self.valid_temp)

    @ti.kernel
    def advect_particles(self, dt: ti.f32):
        for p in self.x:
            pos = self.x[p]
            pv = self.pv[p]

            pos += pv * dt

            if pos.x <= self.bound * self.dx:  # left boundary
                pos.x = (self.bound + 1.0) * self.dx + 1e-4
                pv.x = 0
            if pos.x >= (self.n_grid - self.bound) * self.dx:  # right boundary
                pos.x = (self.n_grid - self.bound - 1.0) * self.dx - 1e-4
                pv.x = 0
            if pos.y <= self.bound * self.dx:  # bottom boundary
                pos.y = (self.bound + 1.0) * self.dx + 1e-4
                pv.y = 0
            if pos.y >= (self.n_grid - self.bound) * self.dx:  # top boundary
                pos.y = (self.n_grid - self.bound - 1.0) * self.dx - 1e-4
                pv.y = 0
            if pos.z <= self.bound * self.dx:  # front boundary
                pos.z = (self.bound + 1.0) * self.dx + 1e-4
                pv.z = 0
            if pos.z >= (self.n_grid - self.bound) * self.dx:  # back boundary
                pos.z = (self.n_grid - self.bound - 1.0) * self.dx - 1e-4
                pv.z = 0

            self.x[p] = pos
            self.pv[p] = pv
