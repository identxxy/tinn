import taichi as ti
from math import pi

@ti.data_oriented
class Encoding:
    def __init__(self, n_input_dims, json) -> None:
        self.otype = json['otype']
        self.n_input_dims = n_input_dims
        if self.otype.lower() == 'identity':
            self.n_output_dims = n_input_dims
            self.kernel = self.identity
        elif self.otype.lower() == 'frequency':
            self.n_frequencies = json['n_frequencies']
            self.n_output_dims = n_input_dims * self.n_frequencies
            self.kernel = self.frequency
        else:
            raise NotImplementedError(f'Encoding {self.otype} not support.')
        print(f'Using encoding {self.kernel.__name__}')

    @ti.kernel
    def frequency(self, input_vf: ti.template(), output_vf: ti.template()):
        for I in ti.grouped(input_vf):
            for n in ti.static(range(input_vf.n)):
                for f in ti.static(range(self.n_frequencies)):
                    output_vf[I][2 * n * f + 0] = ti.cos(input_vf[I][n] * (2**f))
                    output_vf[I][2 * n * f + 1] = ti.sin(input_vf[I][n] * (2**f))

    @ti.kernel
    def identity(self, input_vf: ti.template(), output_vf: ti.template()):
        for I in ti.grouped(input_vf):
            output_vf[I] = input_vf[I]

