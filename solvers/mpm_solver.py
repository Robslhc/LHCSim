import taichi as ti
import numpy as np
from utils import clamp


@ti.data_oriented
class mpm_solver:
    # material
    material_water = 0

    def __init__(self, world, grid_res, bound, max_num_particles=2**27):
        # world and boundary
        self.world = world
        self.bound = bound

        # simulation resolution
        self.n_grid = grid_res
        self.dx = self.world / self.n_grid
        self.default_dt = 1e-4

        # particle properties
        self.p_vol = self.dx**3
        self.p_rho = 1
        self.p_mass = self.p_rho * self.p_vol

        # Water bulk modulus
        self.K = 400

        # particle variables
        self.n_particles = ti.field(dtype=ti.i32, shape=())
        self.x = ti.Vector.field(3, dtype=ti.f32)  # particle position
        self.v = ti.Vector.field(3, dtype=ti.f32)  # particle velocity
        self.C = ti.Matrix.field(
            3, 3, dtype=ti.f32)  # particle affine velocity field
        self.J = ti.field(dtype=ti.f32)  # deformation volume ratio
        self.material_id = ti.field(dtype=ti.i32)

        # grid variables
        self.grid_v = ti.Vector.field(3,
                                      dtype=ti.f32)  # grid momentum / velocity
        self.grid_m = ti.field(dtype=ti.f32)  # grid mass

        # place particle
        self.block_particle = ti.root.pointer(ti.i, max_num_particles)
        self.block_particle.place(self.x, self.v, self.C, self.J,
                                  self.material_id)

        # place grid
        grid_blocksize = 16
        assert self.n_grid % grid_blocksize == 0
        self.block_grid0 = ti.root.pointer(ti.ijk,
                                           self.n_grid // grid_blocksize)
        self.block_grid1 = self.block_grid0.dense(ti.ijk, grid_blocksize)
        self.block_grid1.place(self.grid_v, self.grid_m)

    def add_object(self, geometry, material, par_res=None):
        if par_res is None:
            par_res = 2

        if geometry.type == 'cube':
            assert material == self.material_water
            param = np.array([
                geometry.x_start, geometry.x_end, geometry.y_start,
                geometry.y_end, geometry.z_start, geometry.z_end
            ])
            self.voxelize_cube(param, material, par_res)
        else:
            print("Geometry {} not supported!")

    @ti.kernel
    def voxelize_cube(self, geometry_param: ti.ext_arr(), material: ti.i32,
                      par_res: ti.f32):
        x_start = geometry_param[0]
        lx = geometry_param[1] - x_start
        y_start = geometry_param[2]
        ly = geometry_param[3] - y_start
        z_start = geometry_param[4]
        lz = geometry_param[5] - z_start

        vol = lx * ly * lz
        num_new_particles = ti.cast((vol / self.dx**3) * par_res**3, ti.i32)

        bound = self.bound * self.dx

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
            self.v[p] = ti.Vector.zero(ti.f32, 3)
            self.C[p] = ti.Matrix.zero(ti.f32, 3, 3)
            self.J[p] = 1.0
            self.material_id[p] = material

        self.n_particles[None] += num_new_particles

    def step(self, frame_dt):
        substeps = int(frame_dt / self.default_dt) + 1
        dt = frame_dt / substeps

        for i in range(substeps):
            self.block_grid0.deactivate_all()
            self.p2g(dt)

            # grid operation
            self.grid_gravity(dt)
            self.grid_bounding_box()

            self.g2p(dt)

    @ti.kernel
    def p2g(self, dt: ti.f32):
        inv_dx = 1 / self.dx

        for p in self.x:
            xp = self.x[p]
            base = (xp * inv_dx - 0.5).cast(ti.i32)
            fx = xp * inv_dx - base.cast(ti.f32)

            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2,
                 0.5 * (fx - 0.5)**2]  # Bspline

            stress = ti.Matrix.identity(ti.f32, 3)
            if self.material_id[p] == self.material_water:
                stress = stress * self.K * (self.J[p] - 1)

            stress = (-4 * dt * self.p_vol * inv_dx**2) * stress

            affine = stress + self.p_mass * self.C[p]

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(ti.f32) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]
                self.grid_v[base +
                            offset] += weight * (self.p_mass * self.v[p] +
                                                 affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass

        # velocity normalize
        for i, j, k in self.grid_m:
            if self.grid_m[i, j, k] > 0:
                self.grid_v[i, j,
                            k] = self.grid_v[i, j, k] / self.grid_m[i, j, k]

    @ti.kernel
    def grid_gravity(self, dt: ti.f32):
        for i, j, k in self.grid_v:
            self.grid_v[i, j, k].y -= 9.8 * dt

    @ti.kernel
    def grid_bounding_box(self):
        for i, j, k in self.grid_v:
            if i < self.bound and self.grid_v[i, j, k].x < 0:
                self.grid_v[i, j, k].x = 0
            if i > self.n_grid - self.bound and self.grid_v[i, j, k].x > 0:
                self.grid_v[i, j, k].x = 0
            if j < self.bound and self.grid_v[i, j, k].y < 0:
                self.grid_v[i, j, k].y = 0
            if j > self.n_grid - self.bound and self.grid_v[i, j, k].y > 0:
                self.grid_v[i, j, k].y = 0
            if k < self.bound and self.grid_v[i, j, k].z < 0:
                self.grid_v[i, j, k].z = 0
            if k > self.n_grid - self.bound and self.grid_v[i, j, k].z > 0:
                self.grid_v[i, j, k].z = 0

    @ti.kernel
    def g2p(self, dt: ti.f32):
        inv_dx = 1 / self.dx

        for p in self.x:
            xp = self.x[p]
            base = (xp * inv_dx - 0.5).cast(ti.i32)
            fx = xp * inv_dx - base.cast(ti.f32)

            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2,
                 0.5 * (fx - 0.5)**2]  # Bspline

            new_v = ti.Vector.zero(ti.f32, 3)
            new_C = ti.Matrix.zero(ti.f32, 3, 3)

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = offset.cast(ti.f32) - fx
                g_v = self.grid_v[base + offset]
                weight = w[i][0] * w[j][1] * w[k][2]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

            self.v[p] = new_v
            self.x[p] += dt * self.v[p]
            self.J[p] *= 1 + dt * new_C.trace()
            self.C[p] = new_C
