import taichi as ti

@ti.data_oriented
class Optimizer:
    def __init__(self, json) -> None:
        self.otype = json['otype']
        if self.otype.lower() != 'sgd':
            raise NotImplementedError('Only naive SGD supported.')
        self.learning_rate = json['learning_rate']
    
    def set_network(self, network):
        self.layers_l = network.layers_l
    
    def step(self):
        for l in self.layers_l:
            self.step_layer(l)
    
    @ti.kernel
    def step_layer(self, l: ti.template()):
        for I in ti.grouped(l):
            l[I] -= self.learning_rate * l.grad[I]
