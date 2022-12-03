import taichi as ti
from math import pi

@ti.data_oriented
class Encoding:
    def __init__(self, n_input_dims, json) -> None:
        self.otype = json['otype']
        self.n_input_dims = n_input_dims
        if self.otype.lower() == 'identity':
            self.n_output_dims = n_input_dims
            self.func = self.identity
        elif self.otype.lower() == 'frequency':
            self.n_frequencies = json['n_frequencies']
            self.n_output_dims = n_input_dims * self.n_frequencies
            self.func = self.frequency
        else:
            raise NotImplementedError(f'Encoding {self.otype} not support.')
        print(f'Using encoding {self.func.__name__}')

    @ti.kernel
    def encode_all(self,
        input_vf: ti.template(),
        output_vf: ti.template()
    ):
        for I in ti.grouped(input_vf):
            self.func(input_vf[I], output_vf[I])

    @ti.kernel
    def encode_one(self,
        input_vf: ti.template(),
        output_vf: ti.template(),
        at: ti.template()
    ):
        for I in ti.grouped(input_vf):
            # ugly hack
            yes = 1
            for d in ti.static(range(at.shape[1])):
                if I[1+d] < at[0, d] or I[1+d] >= at[1, d]: # 0 is the batch_size
                    yes = 0
            if yes == 1:
                self.func(input_vf[I], output_vf[I])

    @ti.func
    def frequency(self, input_vf: ti.template(), output_vf: ti.template()):
        for f in ti.static(range(self.n_frequencies)):
            for i in ti.static(range(input_vf.n)):
                output_vf[2 * i * f + 0] = ti.cos(input_vf[i] * (2**f) * 2 * pi)
                output_vf[2 * i * f + 1] = ti.sin(input_vf[i] * (2**f) * 2 * pi)

    @ti.func
    def identity(self, input_vf: ti.template(), output_vf: ti.template()):
        output_vf = input_vf
