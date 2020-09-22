import taichi as ti
import numpy as np
from geometry import cube, sphere
from solvers.mpm_solver import mpm_solver
from renderer.PLY_renderer import PLY_renderer
import argparse
import math
import os

ti.init(arch=ti.gpu, default_fp=ti.f32)

parser = argparse.ArgumentParser(description='Physics based animation')
parser.add_argument('scene', help='Simulation scene')
parser.add_argument('--worldSize', help='Scale of the simulation world')
parser.add_argument('--gridRes', help='Grid resolution')
parser.add_argument('--particleRes',
                    help='Number of particles in each grid per axis')
parser.add_argument('--videoSec',
                    help='Sepecify the total length of the output video')
parser.add_argument('--fluidAlgo', help='Set the fluid simulation algorithm')
parser.add_argument('--fluidPSolver',
                    help='Pressure solver of fluid simulation')
parser.add_argument('--fluidFLIPBlending',
                    help='Specifying the flip blending of PIC-FLIP Algorithm')


def main(args):
    # global parameters with default value
    scene = args.scene
    world = 10
    grid_res = 64
    particle_res = 2
    bound_grid = 3
    video_t = 10
    substeps = 20
    frame_dt = 0.01

    if args.worldSize:
        world = int(args.worldSize)

    if args.gridRes:
        grid_res = int(args.gridRes)

    if args.particleRes:
        particle_res = int(args.particleRes)

    if args.videoSec:
        video_t = int(args.videoSec)

    if scene in ['FluidDamBreak']:
        if args.fluidAlgo == 'MPM':
            solver = mpm_solver(world, grid_res, bound_grid)
        else:
            print("Algorithm {} not supported".format(args.fluidAlgo))
            exit()

        # render define
        renderer = PLY_renderer(px=solver.x,
                                output_dir=os.path.join(os.getcwd(), 'result'))

        if scene == 'FluidDamBreak':
            nstart = solver.n_particles[None]
            solver.add_object(cube(0, 6, 0, 6, 0, 6),
                              mpm_solver.material_water,
                              par_res=particle_res)
            nend = solver.n_particles[None]
            renderer.add_object(nstart, nend, mpm_solver.material_water)
    else:
        print("Scene {} not supported".format(scene))
        exit()

    # start simulate
    t = 0.0
    for frame in range(1, 24 * video_t):
        solver.step(frame_dt)
        t += frame_dt

        # display simulation result
        print("frame {}: ".format(frame))
        pv = solver.v.to_numpy()
        max_vel = np.max(np.linalg.norm(pv, 2, axis=1))
        if not math.isnan(max_vel):
            print("time = {}, dt = {}, maxv = {}".format(t, frame_dt, max_vel))
        else:
            print("Numerical Error occured!")
            exit()

        renderer.output_PLY(frame)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
