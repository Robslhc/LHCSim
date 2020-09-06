# LHCSim
LHCSim is a 3D physics simulation engine developed based on taichi. LHCSim is developed on the purpose of learning simulation algorithms.

## Usage
```
python main.py scene [options]
```

Global options are:
+ --worldSize: Scale of the simulation world per axis.(default 10)
+ --gridRes: Grid resolution per axis.(default 64)
+ --particleRes: Number of particles in each grid per axis.(default 2)
+ --videoSec: Sepecify the total length of the output video. (default 10)
+ --subSteps: Set the simulation substeps per frame. (default 20)

Other options are introduced in each scene.

## Supported Simulation Algorithm
### Fluid
1. MPM: Moving Least Square Material Point Method(MLS-MPM)[1] with explicit time integeration.
2. PIC / FLIP / PIC-FLIP: Particle in Cell Method(PIC)[2], Fluid Implicit Particle Method(FLIP)[3], and also blending them together.
3. APIC: Affine Particle in Cell Method[4]

### Fluid Pressure Solver
1. MGPCG: Multigrid Preconditioned Conjugate Gradient Method[5]

## Rendering
Currently TaichiSim only supports exporting the result to PLY file. The demos are rendered in Houdini software.

## Supported Scenes
### Fluid Simulation
Note:
+ Use --fluidAlgo to specify the simulation algorithm.
+ When the simulation algorithm is PIC / FLIP / PIC-FLIP / APIC, specify the pressure solver by setting --fluidPSolver.
+ When the simulation algorithm is PIC-FLIP, specify the FLIP blending paramater by setting --fluidFLIPBlending.
#### FluidDamBreak
![](demos/FluidDamBreak/DambreakAPIC.gif)

See other demos [here](./demos/FluidDambreak/).
#### FluidSphereFall(WIP)

## Reference
[1]Hu, Y., Fang, Y., Ge, Z., et al. 2018. A Moving Least Squares Material Point Method with Displacement Discontinuity and Two-Way Rigid Body Coupling. ACM Trans. Graph. 37, 4.

[2] Harlow, F.H. 1962. The particle-in-cell method for numerical solution of problems in fluid dynamics.

[3] Brackbill, J.U. and Ruppel, H.M. 1986. FLIP: A method for adaptively zoned, particle-in-cell calculations of fluid flows in two dimensions. Journal of Computational Physics 65, 2, 314–343.

[4] Jiang, C., Schroeder, C., Selle, A., Teran, J., and Stomakhin, A. 2015. The Affine Particle-in-Cell Method. ACM Trans. Graph. 34, 4.

[5] McAdams, A., Sifakis, E., and Teran, J. 2010. A parallel multigrid Poisson solver for fluids simulation on large grids. Computer Animation 2010 - ACM SIGGRAPH / Eurographics Symposium Proceedings, SCA 2010, Eurographics Association, 65–73.