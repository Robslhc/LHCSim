# TaichiSim
TaichiSim is a 3D physics simulation engine developed based on taichi

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

## Rendering
Currently TaichiSim only supports exporting the result to PLY file. The demo are rendered in Houdini software.

## Supported Scenes
### FluidDamBreak
Demo gif needed!
Please Specify the option --fluidAlgo.

## Reference
[1] Hu, Y., Fang, Y., Ge, Z., Qu, Z., Zhu, Y., Pradhana, A., & Jiang, C. (2018). A Moving Least Squares Material Point Method with Displacement Discontinuity and Two-Way Rigid Body Coupling. ACM Trans. Graph., 37(4). https://doi.org/10.1145/3197517.3201293