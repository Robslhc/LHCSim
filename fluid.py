import taichi as ti
import numpy as np
from utils import vec3
import os
'''
class fluid: the class defining fluid simulation
params:
    geometry: the initial geometry of the fluid
    simulation_algorithm: simulation algorithm of the fluid
    particle_res: the number of particles per axis in one grid
    render_type: how to render the fluid object (currently support 'PLY')
'''


@ti.data_oriented
class fluid:
    def __init__(self, geometry, simulation_algorithm, grid_res, particle_res,
                 render_type):
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
        self.C = ti.Matrix.field(3, 3, dtype=ti.f32)
        self.J = ti.field(dtype=ti.f32)

        # memory allocation particle
        max_npar = (self.n_grid * self.n_par)**3
        self.block_particle = ti.root.pointer(ti.i, max_npar)
        self.block_particle.place(self.px, self.pv, self.C, self.J)

        # rendering parameters
        if self.render_type == 'PLY':
            if not os.path.exists('result/PLY/{}'.format(self.sim_algo)):
                os.mkdir('result/PLY/{}'.format(self.sim_algo))

            self.series_prefix = 'result/PLY/{}/fluid.ply'.format(
                self.sim_algo)

    def voxelize(self, world, bound_grid):
        self.geometry.voxelize(self.px, world, self.n_grid, bound_grid,
                               self.n_par)
        self.particle_radius = world / self.n_grid / self.n_par
        self.p_vol = self.particle_radius**3
        self.p_mass = self.p_rho * self.p_vol

    @ti.kernel
    def init_properties(self):
        for p in self.px:
            self.pv[p] = vec3(0.0, 0.0, 0.0)
            self.C[p] = ti.Matrix.zero(ti.f32, 3, 3)
            self.J[p] = 1.0

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
        else:
            print("Algorithm {} for fluid is not supported.")
            exit(0)

    def mpm_one_step(self, grid_v, grid_m, dx, dt, bound):
        # P2G
        self.P2G(grid_v, grid_m, dx, dt)

        # grid operation
        self.apply_gravity(grid_v, dt)
        self.grid_boundary_center(grid_v, bound)

        # G2P
        self.G2P(grid_v, dx, dt)

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
    def G2P(self, grid_v: ti.template(), dx: ti.f32, dt: ti.f32):
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
    def P2G(self, grid_v: ti.template(), grid_m: ti.template(), dx: ti.f32,
            dt: ti.f32):
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
