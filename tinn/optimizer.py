import taichi as ti

@ti.data_oriented
class Optimizer:
    def __init__(self, json) -> None:
        self.otype = json['otype']
        if self.otype.lower() == 'sgd':
            self.learning_rate = json['learning_rate']
            self.step_layer_all = self.sgd_layer_all
            self.step_layer_one = self.sgd_layer_one
        elif self.otype.lower() == 'adam':
            self.learning_rate = json['learning_rate']
            self.beta1 = json['beta1']
            self.beta2 = json['beta2']
            self.epsilon = json['epsilon']
            self.l2_reg = json['l2_reg']
            self.step_layer_all = self.adam_layer_all
            self.step_layer_one = self.adam_layer_one
            raise NotImplementedError('Only naive SGD supported.')
        else:
            raise NotImplementedError('Only naive SGD supported.')
    
    def set_network(self, network):
        self.layers_l = network.layers_l
        if self.otype.lower() == 'adam':
            self.m_l = []
            self.v_l = []
            for l in self.layers_l:
                self.m_l.append(ti.Matrix.field(l.n, l.m, l.dtype, l.shape))
                self.v_l.append(ti.Matrix.field(l.n, l.m, l.dtype, l.shape))
            # TODO
    
    def step_all(self):
        for l in self.layers_l:
            self.step_layer_all(l)

    def step_one(self, at):
        for l in self.layers_l:
            self.step_layer_one(l, at)
    
    @ti.kernel
    def sgd_layer_all(self, l: ti.template()):
        for I in ti.grouped(l):
            l[I] -= self.learning_rate * l.grad[I]

    @ti.kernel
    def sgd_layer_one(self, l: ti.template(), mask: ti.template()):
        for I in ti.grouped(mask):
            if mask[I] == 1:
                l[I] -= self.learning_rate * l.grad[I]
    
    # TODO
    @ti.kernel
    def adam_layer_all(self, l: ti.template()):
        for I in ti.grouped(l):
            l[I] -= self.learning_rate * l.grad[I]

    @ti.kernel
    def adam_layer_one(self, l: ti.template(), mask: ti.template()):
        for I in ti.grouped(mask):
            if mask[I] == 1:
                l[I] -= self.learning_rate * l.grad[I]
    