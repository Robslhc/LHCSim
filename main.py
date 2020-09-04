import taichi as ti
import numpy as np
from geometry import cube
from fluid import fluid
from simulation import simulation
from utils import *
import argparse

ti.init(arch=ti.cpu, default_fp=ti.f32)

parser = argparse.ArgumentParser(description='Physics based animation')
parser.add_argument('scene', help='Simulation scene')
parser.add_argument('--worldSize', help='Scale of the simulation world')
parser.add_argument('--gridRes', help='Grid resolution')
parser.add_argument('--particleRes',
                    help='Number of particles in each grid per axis')
parser.add_argument('--videoSec',
                    help='Sepecify the total length of the output video')
parser.add_argument('--subSteps', help='Set the simulation substeps per frame')
parser.add_argument('--fluidAlgo', help='Set the fluid simulation algorithm')


def create_fluidDamBreak(gp, algo):
    # define fluid object
    fluid_geom = cube(0, 6, 0, 6, 0, 6)
    fluid_obj = fluid(geometry=fluid_geom,
                      simulation_algorithm=algo,
                      grid_res=gp['grid_res'],
                      particle_res=gp['particle_res'],
                      render_type='PLY')

    # define scene
    sim = simulation(gp['world'],
                     gp['bound_grid'],
                     gp['grid_res'], [fluid_obj],
                     render_algo='PLY')

    return sim


def main(args):
    # global parameters with default value
    scene = args.scene
    world = 10
    grid_res = 64
    particle_res = 2
    bound_grid = 3
    video_t = 10
    substeps = 20
    dt = 0.001

    if args.worldSize:
        world = int(args.worldSize)

    if args.gridRes:
        grid_res = int(args.gridRes)

    if args.particleRes:
        particle_res = int(args.particleRes)

    if args.videoSec:
        video_t = int(args.video_t)

    if args.subSteps:
        substeps = int(args.subSteps)

    gp = {
        'world': world,
        'grid_res': grid_res,
        'particle_res': particle_res,
        'bound_grid': bound_grid
    }

    if scene == 'FluidDamBreak':
        assert args.fluidAlgo is not None
        sim = create_fluidDamBreak(gp, args.fluidAlgo)
    else:
        print("Scene {} not supported.".format(scene))
        exit()

    # init simulation
    sim.init_particle()

    # render the initial particles
    sim.render(0)

    # start simulate
    t = 0.0
    for frame in range(1, 24 * video_t):
        for substep in range(substeps):
            sim.one_step(dt)
            t += dt

        # display simulation result
        print("frame {}: ".format(frame))
        for i, obj in enumerate(sim.obj_list):
            pv = obj.pv.to_numpy()
            max_vel = np.max(np.linalg.norm(pv, 2, axis=1))
            print("obj {}, time = {}, dt = {}, maxv = {}".format(
                i, t, dt, max_vel))

        sim.render(frame)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
