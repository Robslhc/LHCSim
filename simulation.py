import taichi as ti
import numpy as np
from utils import *
'''
class simulation: the class for whole simulation and rendering
params:
    world: the length of the simulation world per axis (assuming the world as a cube)
    bound_grid: the number of grid layers per axis (currently only supports positive integers)
    grid_res: grid resolution for simulation
    obj_list: the objects to simulate and render
    render_algo: rendering algorithm (currently supports 'PLY')
'''


@ti.data_oriented
class simulation:
    def __init__(self, world, bound_grid, grid_res, obj_list, render_algo):
        # world and boundary
        self.world = world
        self.bound = bound_grid

        # object list
        self.obj_list = obj_list

        # simulation resolution
        self.n_grid = grid_res
        self.dx = self.world / self.n_grid

        # grid parameters
        self.grid_v = ti.Vector.field(3, dtype=ti.f32)
        self.grid_m = ti.field(dtype=ti.f32)

        # memory allocation grid
        grid_blocksize = 16
        assert self.n_grid % grid_blocksize == 0
        self.block0 = ti.root.pointer(ti.ijk, self.n_grid // grid_blocksize)
        self.block1 = self.block0.dense(ti.ijk, grid_blocksize)
        self.block1.place(self.grid_v, self.grid_m)

        # render parameters
        self.render_algo = render_algo

    def init_particle(self):
        for obj in self.obj_list:
            # init particle positions
            obj.voxelize(self.world, self.bound)

            # init particle properties
            obj.init_properties()

    def render(self, frame_id):
        if self.render_algo == 'PLY':
            for obj in self.obj_list:
                assert obj.render_type == 'PLY'
                obj.output_PLY(frame_id)

    def one_step(self, dt):
        # dt setting
        # dt = 0.001

        # deactivate sparse structure
        self.block0.deactivate_all()

        # fluid one step
        for obj in self.obj_list:
            obj.one_step(self.grid_v, self.grid_m, self.dx, dt, self.bound)
