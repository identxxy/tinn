import taichi as ti
from .encoding import Encoding
from .network import Network

@ti.data_oriented
class NetworkWithInputEncoding(Network):
    def __init__(self, n_input_dims, n_output_dims, encoding_json, network_json, grid_shape=1) -> None:
        self.encoding = Encoding(n_input_dims, encoding_json)
        self.n_encoded_dims = self.encoding.n_output_dims
        super().__init__(self.n_encoded_dims, n_output_dims, network_json, grid_shape)

        # encoding field
        self.encode_vf_train = None
        self.io_shape_train = None
        self.encode_vf_eval = None
        self.io_shape_eval = None
    
    def all_inference(self, input_vf: ti.template(), output_vf: ti.template()):
        ''' Forward all NN network. Using eval field.

        The first dimension of `input_vf` and `output_vf` is batch_size,
        while the remaining dimension is the NN array shape.
        Args:
            input_vf:   VectorField of shape (batch_size, network.grid_shape)
            output_vf:  VectorField of shape (batch_size, network.grid_shape)
        '''
        assert input_vf.shape == output_vf.shape
        io_shape = input_vf.shape
        self.batch_size = io_shape[0]
        if (self.encode_vf_eval == None) or self.io_shape_eval != io_shape:
            self.io_shape_eval = io_shape
            self.encode_vf_eval = ti.Vector.field(self.n_encoded_dims, Network.dtype, shape=io_shape, needs_grad=True)
            print(f'Encoding create new internal field of shape {io_shape} for inference.')
        self.encoding.kernel(input_vf, self.encode_vf_eval)
        Network.all_inference(self, self.encode_vf_eval, output_vf)

    def _all_train(self, input_vf: ti.template(), output_vf: ti.template()):
        ''' Forward all NN network. Using train field. It should be called by Trainer, with `ti.ad.Tape()`.

        The first dimension of `input_vf` and `output_vf` is batch_size,
        while the remaining dimension is the NN array shape.
        Args:
            input_vf:   VectorField of shape (batch_size, network.grid_shape)
            output_vf:  VectorField of shape (batch_size, network.grid_shape)
        '''
        assert input_vf.shape == output_vf.shape
        io_shape = input_vf.shape
        self.batch_size = io_shape[0]
        if (self.encode_vf_train == None) or self.io_shape_train != io_shape:
            self.io_shape_train = io_shape
            self.encode_vf_train = ti.Vector.field(self.n_encoded_dims, Network.dtype, shape=io_shape, needs_grad=True)
            print(f'Encoding create new internal field of shape {io_shape} for training.')
        self.encoding.kernel(input_vf, self.encode_vf_train)
        Network._all_train(self, self.encode_vf_train, output_vf)
