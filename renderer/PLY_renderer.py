import taichi as ti
import numpy as np
import os


@ti.data_oriented
class PLY_renderer:
    def __init__(self, px, output_dir):
        self.px = px
        self.output_dir = output_dir

        self.particle_positions = []
        self.particle_material = []

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def add_object(self, nstart, nend, material):
        self.particle_positions.append((nstart, nend))
        self.particle_material.append(material)

    def output_PLY(self, frame_id):
        np_px = self.px.to_numpy().copy()
        np_px = np.reshape(np_px, (-1, 3))

        if not os.path.exists(os.path.join(self.output_dir, 'PLY')):
            os.mkdir(os.path.join(self.output_dir, 'PLY'))

        for i, pos in enumerate(self.particle_positions):
            start = pos[0]
            end = pos[1]
            out_px = np_px[start:end]
            nparticles = end - start

            series_prefix = '{}/PLY/mpm_mat{}.ply'.format(
                self.output_dir, self.particle_material[i])

            writer = ti.PLYWriter(num_vertices=nparticles)
            writer.add_vertex_pos(out_px[:, 0], out_px[:, 1], out_px[:, 2])
            writer.export_frame_ascii(frame_id, series_prefix)
