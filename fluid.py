import taichi as ti
import numpy as np
import os
from utils import vec3
from pressure_solver.MGPCGSolver import MGPCGSolver

#define cell type
FLUID = 0
AIR = 1
SOLID = 2
'''
class fluid: the class defining fluid simulation
params:
    geometry: the initial geometry of the fluid
    simulation_algorithm: simulation algorithm of the fluid
    particle_res: the number of particles per axis in one grid
    render_type: how to render the fluid object (currently support 'PLY')
    pressure_solver: implicit pressure solver (currently support 'MGPCG')
    flip_blending: flip blending parameter (only valid when simulation_algorithm is 'PIC-FLIP')
'''


@ti.data_oriented
class fluid:
    def __init__(self,
                 geometry,
                 simulation_algorithm,
                 grid_res,
                 particle_res,
                 render_type,
                 pressure_solver=None,
                 flip_blending=None):
        self.m_id = 1

        # simulation parameter
        self.sim_algo = simulation_algorithm
        self.n_grid = grid_res
        self.n_par = particle_res
        self.render_type = render_type

        self.p_rho = 1
        self.p_vol = 0  # set in voxelize
        self.p_mass = 0  # set in voxelize
        self.E = 400

        # object geometry
        self.geometry = geometry

        # particle parameters
        self.px = ti.Vector.field(3, dtype=ti.f32)
        self.pv = ti.Vector.field(3, dtype=ti.f32)
        if self.sim_algo == 'MPM':
            self.C = ti.Matrix.field(3, 3, dtype=ti.f32)
            self.J = ti.field(dtype=ti.f32)
        elif self.sim_algo == 'APIC':
            self.c_x = ti.Vector.field(3, dtype=ti.f32)
            self.c_y = ti.Vector.field(3, dtype=ti.f32)
            self.c_z = ti.Vector.field(3, dtype=ti.f32)

        # grid paramaters
        if self.sim_algo in ['PIC', 'APIC', 'FLIP', 'PIC-FLIP']:
            # MAC grid velocities
            self.u = ti.field(dtype=ti.f32,
                              shape=(self.n_grid + 1, self.n_grid,
                                     self.n_grid))
            self.v = ti.field(dtype=ti.f32,
                              shape=(self.n_grid, self.n_grid + 1,
                                     self.n_grid))
            self.w = ti.field(dtype=ti.f32,
                              shape=(self.n_grid, self.n_grid,
                                     self.n_grid + 1))
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

            if self.sim_algo == 'FLIP' or self.sim_algo == 'PIC-FLIP':
                self.u_last = ti.field(dtype=ti.f32,
                                       shape=(self.n_grid + 1, self.n_grid,
                                              self.n_grid))
                self.v_last = ti.field(dtype=ti.f32,
                                       shape=(self.n_grid, self.n_grid + 1,
                                              self.n_grid))
                self.w_last = ti.field(dtype=ti.f32,
                                       shape=(self.n_grid, self.n_grid,
                                              self.n_grid + 1))

            # Cell type
            self.cell_type = ti.field(dtype=ti.i32,
                                      shape=(self.n_grid, self.n_grid,
                                             self.n_grid))

            # Extrapolate utils
            self.valid = ti.field(dtype=ti.i32,
                                  shape=(self.n_grid + 1, self.n_grid + 1,
                                         self.n_grid + 1))
            self.valid_temp = ti.field(dtype=ti.i32,
                                       shape=(self.n_grid + 1, self.n_grid + 1,
                                              self.n_grid + 1))

            # pressure and solver
            self.p = ti.field(dtype=ti.f32,
                              shape=(self.n_grid, self.n_grid, self.n_grid))
            self.psolve_algo = pressure_solver
            assert self.psolve_algo == 'MGPCG'
            if self.psolve_algo == 'MGPCG':
                self.p_solver = MGPCGSolver(self.n_grid, self.u, self.v,
                                            self.w, self.cell_type)

        if self.sim_algo == 'PIC-FLIP':
            self.FLIP_BLENDING = flip_blending

        # memory allocation particle
        max_npar = (self.n_grid * self.n_par)**3
        self.block_particle = ti.root.pointer(ti.i, max_npar)
        if self.sim_algo == 'MPM':
            self.block_particle.place(self.px, self.pv, self.C, self.J)
        elif self.sim_algo == 'APIC':
            self.block_particle.place(self.px, self.pv, self.c_x, self.c_y,
                                      self.c_z)
        else:
            self.block_particle.place(self.px, self.pv)

        # rendering parameters
        if self.render_type == 'PLY':
            if not os.path.exists('result/PLY/{}'.format(self.sim_algo)):
                os.mkdir('result/PLY/{}'.format(self.sim_algo))

            self.series_prefix = 'result/PLY/{}/fluid.ply'.format(
                self.sim_algo)

    def voxelize(self, world, bound_grid):
        self.geometry.voxelize(self.px, world, self.n_grid, bound_grid,
                               self.n_par)
        if self.sim_algo != 'MPM':
            # mark boundary
            self.mark_boundary(bound_grid)
            self.mark_cell(world / self.n_grid)
        else:
            self.particle_radius = world / self.n_grid / self.n_par
            self.p_vol = self.particle_radius**3
            self.p_mass = self.p_rho * self.p_vol

    @ti.kernel
    def mark_boundary(self, bound: ti.i32):
        for i, j, k in self.cell_type:
            if i < bound or i > self.n_grid - bound or j < bound or j > self.n_grid - bound or k < bound or k > self.n_grid - bound:
                self.cell_type[i, j, k] = SOLID

    @ti.kernel
    def init_properties(self):
        # init particles
        for p in self.px:
            self.pv[p] = vec3(0.0, 0.0, 0.0)
            if ti.static(self.sim_algo == 'MPM'):
                self.C[p] = ti.Matrix.zero(ti.f32, 3, 3)
                self.J[p] = 1.0
            elif ti.static(self.sim_algo == 'APIC'):
                self.c_x[p] = vec3(0.0, 0.0, 0.0)
                self.c_y[p] = vec3(0.0, 0.0, 0.0)
                self.c_z[p] = vec3(0.0, 0.0, 0.0)

        # init field
        if ti.static(self.sim_algo != 'MPM'):
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

    def output_PLY(self, frame_id):
        np_px = self.px.to_numpy().copy()
        np_px = np.reshape(np_px, (-1, 3))
        np_px = np_px[np.where((np_px[:, 0] != 0.0) | (np_px[:, 1] != 0.0)
                               | (np_px[:, 2] != 0.0))]

        nparticles = np_px.shape[0]

        writer = ti.PLYWriter(num_vertices=nparticles)
        writer.add_vertex_pos(np_px[:, 0], np_px[:, 1], np_px[:, 2])

        writer.export_frame_ascii(frame_id, self.series_prefix)

    def one_step(self, grid_v, grid_m, dx, dt, bound):
        if self.sim_algo == 'MPM':
            self.mpm_one_step(grid_v, grid_m, dx, dt, bound)
        elif self.sim_algo in ['PIC', 'APIC', 'FLIP', 'PIC-FLIP']:
            self.hybrid_one_step(dx, dt, bound)
        else:
            print("Algorithm {} for fluid is not supported.")
            exit(0)

    def mpm_one_step(self, grid_v, grid_m, dx, dt, bound):
        # P2G
        self.center_P2G(grid_v, grid_m, dx, dt)

        # grid operation
        self.apply_gravity(grid_v, dt)
        self.grid_boundary_center(grid_v, bound)

        # G2P
        self.center_G2P(grid_v, dx, dt)

    @ti.kernel
    def apply_gravity(self, grid_v: ti.template(), dt: ti.f32):
        for i, j, k in grid_v:
            grid_v[i, j, k].y -= dt * 9.8

    @ti.kernel
    def grid_boundary_center(self, grid_v: ti.template(), bound: ti.i32):
        for i, j, k in grid_v:
            if i < bound and grid_v[i, j, k].x < 0:
                grid_v[i, j, k].x = 0
            if i > self.n_grid - bound and grid_v[i, j, k].x > 0:
                grid_v[i, j, k].x = 0
            if j < bound and grid_v[i, j, k].y < 0:
                grid_v[i, j, k].y = 0
            if j > self.n_grid - bound and grid_v[i, j, k].y > 0:
                grid_v[i, j, k].y = 0
            if k < bound and grid_v[i, j, k].z < 0:
                grid_v[i, j, k].z = 0
            if k > self.n_grid - bound and grid_v[i, j, k].z > 0:
                grid_v[i, j, k].z = 0

    @ti.kernel
    def center_G2P(self, grid_v: ti.template(), dx: ti.f32, dt: ti.f32):
        inv_dx = 1 / dx

        for p in self.px:
            xp = self.px[p]
            base = (xp * inv_dx - 0.5).cast(ti.i32)
            fx = xp * inv_dx - base.cast(ti.f32)

            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2,
                 0.5 * (fx - 0.5)**2]  # Bspline

            new_v = ti.Vector.zero(ti.f32, 3)
            new_C = ti.Matrix.zero(ti.f32, 3, 3)

            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        dpos = vec3(i, j, k) - fx
                        g_v = grid_v[base + vec3(i, j, k)]
                        weight = w[i][0] * w[j][1] * w[k][2]
                        new_v += weight * g_v
                        new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

            self.pv[p] = new_v
            self.px[p] += dt * self.pv[p]
            self.J[p] *= 1 + dt * new_C.trace()
            self.C[p] = new_C

    @ti.kernel
    def center_P2G(self, grid_v: ti.template(), grid_m: ti.template(),
                   dx: ti.f32, dt: ti.f32):
        inv_dx = 1 / dx

        for p in self.px:
            xp = self.px[p]
            base = (xp * inv_dx - 0.5).cast(ti.i32)
            fx = xp * inv_dx - base.cast(ti.f32)

            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2,
                 0.5 * (fx - 0.5)**2]  # Bspline

            stress = -4 * dt * self.E * self.p_vol * (self.J[p] -
                                                      1) * inv_dx**2
            affine = ti.Matrix([[stress, 0, 0], [0, stress, 0], [0, 0, stress]
                                ]) + self.p_mass * self.C[p]

            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        offset = vec3(i, j, k)
                        dpos = (offset.cast(ti.f32) - fx) * dx
                        weight = w[i][0] * w[j][1] * w[k][2]
                        grid_v[base +
                               offset] += weight * (self.p_mass * self.pv[p] +
                                                    affine @ dpos)
                        grid_m[base + offset] += weight * self.p_mass

        # velocity normalize
        for i, j, k in grid_m:
            if grid_m[i, j, k] > 0:
                grid_v[i, j, k] = grid_v[i, j, k] / grid_m[i, j, k]

    def hybrid_one_step(self, dx, dt, bound):
        # P2G
        self.MAC_P2G(dx, dt)

        # save temp grid velocity
        if self.sim_algo == 'FLIP' or self.sim_algo == 'PIC-FLIP':
            self.u_last.copy_from(self.u)
            self.v_last.copy_from(self.v)
            self.w_last.copy_from(self.w)

        # Apply gravity
        self.MAC_apply_gravity(dt)
        self.grid_boundary_MAC()

        # Extrapolate velocity
        self.extrapolate_velocity()
        self.grid_boundary_MAC()

        # Apply pressure
        self.solve_pressure(dx, dt)
        self.apply_pressure(dx, dt)
        self.grid_boundary_MAC()

        # Extrapolate velocity
        self.extrapolate_velocity()
        self.grid_boundary_MAC()

        # G2P
        self.MAC_G2P(dx, dt)
        self.advect_particles(dx, dt, bound)
        self.mark_cell(dx)

        self.u.fill(0.0)
        self.v.fill(0.0)
        self.w.fill(0.0)
        self.u_weight.fill(0.0)
        self.v_weight.fill(0.0)
        self.w_weight.fill(0.0)

    #define helper functions dealing with cell type
    @ti.func
    def is_valid(self, i, j, k):
        return i >= 0 and i < self.n_grid and j >= 0 and j < self.n_grid and k >= 0 and k < self.n_grid

    @ti.func
    def is_fluid(self, i, j, k):
        return self.is_valid(i, j, k) and self.cell_type[i, j, k] == FLUID

    @ti.func
    def is_air(self, i, j, k):
        return self.is_valid(i, j, k) and self.cell_type[i, j, k] == AIR

    @ti.func
    def is_solid(self, i, j, k):
        return self.is_valid(i, j, k) and self.cell_type[i, j, k] == SOLID

    @ti.func
    def MAC_P2G_onedir(self, dx, stagger, xp, vp, grid_v, grid_m):
        inv_dx = 1 / dx

        base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
        fx = xp * inv_dx - (base.cast(ti.f32) + stagger)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2,
             0.5 * (fx - 0.5)**2]  # Bspline

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = vec3(i, j, k)
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_v[base + offset] += weight * vp
            grid_m[base + offset] += weight

    @ti.func
    def MAC_P2G_onedir_apic(self, dx, stagger, xp, vp, cp, grid_v, grid_m):
        inv_dx = 1 / dx

        base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
        fx = xp * inv_dx - (base.cast(ti.f32) + stagger)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2,
             0.5 * (fx - 0.5)**2]  # Bspline

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = vec3(i, j, k)
            weight = w[i][0] * w[j][1] * w[k][2]
            dpos = (offset.cast(ti.f32) - fx) * dx
            grid_v[base + offset] += weight * (vp + cp.dot(dpos))
            grid_m[base + offset] += weight

    @ti.kernel
    def MAC_P2G(self, dx: ti.f32, dt: ti.f32):
        stagger_u = vec3(0, 0.5, 0.5)
        stagger_v = vec3(0.5, 0, 0.5)
        stagger_w = vec3(0.5, 0.5, 0)

        for p in self.px:
            xp = self.px[p]

            if ti.static(self.sim_algo == 'APIC'):
                self.MAC_P2G_onedir_apic(dx, stagger_u, xp, self.pv[p].x,
                                         self.c_x[p], self.u, self.u_weight)
                self.MAC_P2G_onedir_apic(dx, stagger_v, xp, self.pv[p].y,
                                         self.c_y[p], self.v, self.v_weight)
                self.MAC_P2G_onedir_apic(dx, stagger_w, xp, self.pv[p].z,
                                         self.c_z[p], self.w, self.w_weight)
            else:
                self.MAC_P2G_onedir(dx, stagger_u, xp, self.pv[p].x, self.u,
                                    self.u_weight)
                self.MAC_P2G_onedir(dx, stagger_v, xp, self.pv[p].y, self.v,
                                    self.v_weight)
                self.MAC_P2G_onedir(dx, stagger_w, xp, self.pv[p].z, self.w,
                                    self.w_weight)

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
    def MAC_G2P_onedir(self, dx, stagger, xp, grid_v):
        inv_dx = 1 / dx

        # x direction
        base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
        fx = xp * inv_dx - (base.cast(ti.f32) + stagger)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2,
             0.5 * (fx - 0.5)**2]  # Bspline
        new_v = 0.0

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = vec3(i, j, k)
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * grid_v[base + offset]

        return new_v

    @ti.func
    def MAC_G2P_onedir_flip(self, dx, stagger, xp, grid_v, grid_v_last, vp,
                            flip_blending):
        inv_dx = 1 / dx

        # x direction
        base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
        fx = xp * inv_dx - (base.cast(ti.f32) + stagger)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2,
             0.5 * (fx - 0.5)**2]  # Bspline
        new_v = 0.0
        new_v_delta = 0.0

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = vec3(i, j, k)
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * grid_v[base + offset]
            new_v_delta += weight * (grid_v[base + offset] -
                                     grid_v_last[base + offset])

        return flip_blending * (vp + new_v_delta) + (1 - flip_blending) * new_v

    @ti.func
    def MAC_G2P_onedir_apic(self, dx, stagger, xp, grid_v):
        inv_dx = 1 / dx

        # x direction
        base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
        fx = xp * inv_dx - (base.cast(ti.f32) + stagger)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2,
             0.5 * (fx - 0.5)**2]  # Bspline
        new_v = 0.0
        new_c = vec3(0.0, 0.0, 0.0)

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = vec3(i, j, k)
            dpos = offset.cast(ti.f32) - fx
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * grid_v[base + offset]
            new_c += 4 * weight * dpos * grid_v[base + offset] * inv_dx

        return new_v, new_c

    @ti.kernel
    def MAC_G2P(self, dx: ti.f32, dt: ti.f32):
        stagger_u = vec3(0, 0.5, 0.5)
        stagger_v = vec3(0.5, 0, 0.5)
        stagger_w = vec3(0.5, 0.5, 0)

        for p in self.px:
            xp = self.px[p]

            if ti.static(self.sim_algo == 'APIC'):
                self.pv[p].x, self.c_x[p] = self.MAC_G2P_onedir_apic(
                    dx, stagger_u, xp, self.u)
                self.pv[p].y, self.c_y[p] = self.MAC_G2P_onedir_apic(
                    dx, stagger_v, xp, self.v)
                self.pv[p].z, self.c_z[p] = self.MAC_G2P_onedir_apic(
                    dx, stagger_w, xp, self.w)
            elif ti.static(self.sim_algo == 'FLIP'):
                self.pv[p].x = self.MAC_G2P_onedir_flip(
                    dx, stagger_u, xp, self.u, self.u_last, self.pv[p].x, 1.0)
                self.pv[p].y = self.MAC_G2P_onedir_flip(
                    dx, stagger_v, xp, self.v, self.v_last, self.pv[p].y, 1.0)
                self.pv[p].z = self.MAC_G2P_onedir_flip(
                    dx, stagger_w, xp, self.w, self.w_last, self.pv[p].z, 1.0)
            elif ti.static(self.sim_algo == 'PIC-FLIP'):
                self.pv[p].x = self.MAC_G2P_onedir_flip(
                    dx, stagger_u, xp, self.u, self.u_last, self.pv[p].x,
                    self.FLIP_BLENDING)
                self.pv[p].y = self.MAC_G2P_onedir_flip(
                    dx, stagger_v, xp, self.v, self.v_last, self.pv[p].y,
                    self.FLIP_BLENDING)
                self.pv[p].z = self.MAC_G2P_onedir_flip(
                    dx, stagger_w, xp, self.w, self.w_last, self.pv[p].z,
                    self.FLIP_BLENDING)
            else:
                self.pv[p].x = self.MAC_G2P_onedir(dx, stagger_u, xp, self.u)
                self.pv[p].y = self.MAC_G2P_onedir(dx, stagger_v, xp, self.v)
                self.pv[p].z = self.MAC_G2P_onedir(dx, stagger_w, xp, self.w)

    @ti.kernel
    def MAC_apply_gravity(self, dt: ti.f32):
        for i, j, k in self.v:
            self.v[i, j, k] -= 9.8 * dt

    @ti.kernel
    def grid_boundary_MAC(self):
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
        for i in range(10):
            self.u_temp.copy_from(self.u)
            self.valid_temp.copy_from(self.valid)
            diffuse_quantity(self.u, self.u_temp, self.valid, self.valid_temp)

        mark_valid_v()
        for i in range(10):
            self.v_temp.copy_from(self.v)
            self.valid_temp.copy_from(self.valid)
            diffuse_quantity(self.v, self.v_temp, self.valid, self.valid_temp)

        mark_valid_w()
        for i in range(10):
            self.w_temp.copy_from(self.w)
            self.valid_temp.copy_from(self.valid)
            diffuse_quantity(self.w, self.w_temp, self.valid, self.valid_temp)

    def solve_pressure(self, dx, dt):
        scale_A = dt / (1000 * dx**2)
        scale_b = 1 / dx

        self.p_solver.system_init(scale_A, scale_b)
        self.p_solver.solve(500)

        self.p.copy_from(self.p_solver.p)

    @ti.kernel
    def apply_pressure(self, dx: ti.f32, dt: ti.f32):
        scale = dt / (1000 * dx)

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
    def advect_particles(self, dx: ti.f32, dt: ti.f32, bound: ti.i32):
        for p in self.px:
            pos = self.px[p]
            pv = self.pv[p]

            pos += pv * dt

            if pos.x <= bound * dx:  # left boundary
                pos.x = bound * dx
                pv.x = 0
            if pos.x >= (self.n_grid - bound) * dx:  # right boundary
                pos.x = (self.n_grid - bound) * dx
                pv.x = 0
            if pos.y <= bound * dx:  # bottom boundary
                pos.y = bound * dx
                pv.y = 0
            if pos.y >= (self.n_grid - bound) * dx:  # top boundary
                pos.y = (self.n_grid - bound) * dx
                pv.y = 0
            if pos.z <= bound * dx:  # front boundary
                pos.z = bound * dx
                pv.z = 0
            if pos.z >= (self.n_grid - bound) * dx:  # back boundary
                pos.z = (self.n_grid - bound) * dx
                pv.z = 0

            self.px[p] = pos
            self.pv[p] = pv

    @ti.kernel
    def mark_cell(self, dx: ti.f32):
        for i, j, k in self.cell_type:
            if not self.is_solid(i, j, k):
                self.cell_type[i, j, k] = AIR

        for p in self.px:
            xp = self.px[p]
            idx = ti.cast(ti.floor(xp / dx), ti.i32)

            if not self.is_solid(idx[0], idx[1], idx[2]):
                self.cell_type[idx] = FLUID
