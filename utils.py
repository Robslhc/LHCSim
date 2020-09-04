import taichi as ti


@ti.pyfunc
def clamp(x, a, b):
    return max(a, min(b, x))


@ti.pyfunc
def particle_idx(i, j, k, ix, jx, kx, ngrid, npar):
    gid = i * ngrid**2 + j * ngrid + k
    pid = gid * npar**3 + ix * npar**2 + jx * npar + kx

    return pid


@ti.pyfunc
def vec3(x, y, z):
    return ti.Vector([x, y, z])
