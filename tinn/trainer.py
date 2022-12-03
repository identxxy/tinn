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

        self.loss_func = loss.loss_func
        self.loss_all = loss.loss_all
        self.loss_one = loss.loss_one

        # connect optimizer and network
        self.optimizer.set_network(network)
        self.loss_sf_all = ti.field(ti.f32, shape=(), needs_grad=True)
        self.loss_sf_one = ti.field(ti.f32, shape=(), needs_grad=True)
        # FowardContext
        self.output = None
        self.losses = None

        self.io_shape = None
        self.n_elements = None
    
    def training_step_all(self, input_vf: ti.template(), target_vf: ti.template(), run_optimizer=True):
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
            self.output = ti.Vector.field(self.network.n_output_dims, self.network.dtype, shape=io_shape, needs_grad=True)
            self.losses = ti.field(self.network.dtype, shape=io_shape, needs_grad=True)
            self.n_elements = 1
            for i in io_shape:
                self.n_elements *= i
            print(f'Trainer created new internal field {io_shape} for all NN, totally elements number {self.n_elements}.')
        with ti.ad.Tape(self.loss_sf_all):
            self.network._train_all(input_vf, self.output)
            self.loss_all(1. / self.io_shape[0],  self.output, target_vf, self.losses)
            self.sum_loss_all()

        if run_optimizer:
            self.optimizer.step_all()

        return ForwardContext(self.output, self.losses, self.loss_sf_all)

    def training_step_one(self,
            input_vf: ti.template(),
            target_vf: ti.template(),
            at: ti.template(),
            run_optimizer=True
        ):
        ''' Train a ranged NN network one step.

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
            self.output = ti.Vector.field(self.network.n_output_dims, self.network.dtype, shape=io_shape, needs_grad=True)
            self.losses = ti.field(self.network.dtype, shape=io_shape, needs_grad=True)
            print(f'Trainer created new internal field {io_shape} for one NN.')
        with ti.ad.Tape(self.loss_sf_one):
            self.network._train_one(input_vf, self.output, at)
            self.loss_one(1. / self.io_shape[0],  self.output, target_vf, self.losses, at)
            self.sum_loss_one(at)

        if run_optimizer:
            self.optimizer.step_one(at)

        return ForwardContext(self.output, self.losses, self.loss_sf_one)

    @ti.kernel
    def sum_loss_all(self):
        for I in ti.grouped(self.losses):
            self.loss_sf_all[None] += self.losses[I]

    @ti.kernel
    def sum_loss_one(self, at: ti.template()):
        for I in ti.grouped(self.losses):
            # ugly hack
            yes = 1
            for d in ti.static(range(at.shape[1])):
                if I[1+d] < at[0, d] or I[1+d] >= at[1, d]: # 0 is the batch_size
                    yes = 0
            if yes == 1:
                self.loss_sf_one[None] += self.losses[I]
