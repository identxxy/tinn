import taichi as ti
from .network import Network
from .loss import Loss
from .optimizer import Optimizer

# loss_val is `loss[None]`
class ForwardContext:
    def __init__(self, output, losses, loss) -> None:
        self.output = output
        self.losses = losses
        self.loss = loss

@ti.data_oriented
class Trainer:
    def __init__(self,
            network: Network,
            optimizer: Optimizer,
            loss: Loss
        ) -> None:
        self.network = network
        self.optimizer = optimizer
        self.loss_kernel = loss.kernel

        # connect optimizer and network
        self.optimizer.set_network(network)
        self.loss_sf = ti.field(ti.f32, shape=(), needs_grad=True)
        # FowardContext
        self.forward_context = ForwardContext(None, None, self.loss_sf)

        self.io_shape = None
        self.n_elements = None
    
    def training_step(self, input_vf: ti.template(), target_vf: ti.template(), run_optimizer=True):
        ''' Train all NN network one step.

        The first dimension of `input_vf` and `output_vf` is batch_size,
        while the remaining dimension is the NN array shape.
        This function allocate output field for the network to forward.
        Args:
            input_vf:   VectorField of shape (batch_size, network.grid_shape)
            output_vf:  VectorField of shape (batch_size, network.grid_shape)
        '''
        assert input_vf.shape == target_vf.shape
        io_shape = input_vf.shape
        if self.io_shape != io_shape:
            self.io_shape = io_shape
            self.forward_context.output = ti.Vector.field(self.network.n_output_dims, self.network.dtype, shape=io_shape, needs_grad=True)
            self.forward_context.losses = ti.field(self.network.dtype, shape=io_shape, needs_grad=True)
            self.n_elements = 1
            for i in io_shape:
                self.n_elements *= i
            print(f'Trainer got new io_shape {io_shape}, totally elements number {self.n_elements}.')
            print(f'Allocated new field to store network output and loss.')
        with ti.ad.Tape(self.loss_sf):
            self.network.all_forward(input_vf, self.forward_context.output)
            self.loss_kernel(self.forward_context.output, target_vf, self.forward_context.losses)
            self.sum_loss()

        if run_optimizer:
            self.optimizer.step()

        return self.forward_context

    @ti.kernel
    def sum_loss(self):
        for I in ti.grouped(self.forward_context.losses):
            self.loss_sf[None] += self.forward_context.losses[I]
