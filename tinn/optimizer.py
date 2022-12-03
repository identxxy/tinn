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
    
    def step_all(self):
        for l in self.layers_l:
            self.step_layer_all(l)

    def step_one(self, at):
        for l in self.layers_l:
            self.step_layer_one(l, at)
    
    @ti.kernel
    def step_layer_all(self, l: ti.template()):
        for I in ti.grouped(l):
            l[I] -= self.learning_rate * l.grad[I]
            # if l.grad[I].sum() == 0. :
            #     print('0grad, ', I)

    @ti.kernel
    def step_layer_one(self, l: ti.template(), at: ti.template()):
        for I in ti.grouped(l):
            # ugly hack
            yes = 1
            for d in ti.static(range(at.shape[1])):
                if I[d] < at[0, d] or I[d] >= at[1, d]:
                    yes = 0
            if yes == 1:
                l[I] -= self.learning_rate * l.grad[I]
                if l.grad[I].sum() == 0. :
                    print('0grad, ', I)