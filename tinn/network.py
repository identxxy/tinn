import taichi as ti

@ti.data_oriented
class Network:
    # precision type
    dtype = ti.f32
    
    def __init__(self, n_input_dims, n_output_dims, json) -> None:
        self.otype = json['otype']  # no use... just naive matrix multiplication.
        self.grid_shape = json['grid_shape']
        self.activation = activation_dict[json['activation']]
        self.output_activation = activation_dict[json['output_activation']]
        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        self.n_neurons = json['n_neurons']
        self.n_hidden_layers = json['n_hidden_layers']

        self.hidden_layers_w_l = []
        self.hidden_layers_b_l = []
        if self.n_hidden_layers == 0:   # just a matmul
            self.input_layer_w = ti.Matrix.field(self.n_output_dims, self.n_input_dims, Network.dtype, shape=self.grid_shape, needs_grad=True)
            self.input_layer_b = ti.Vector.field(self.n_output_dims, Network.dtype, shape=self.grid_shape, needs_grad=True)
            self.output_layer_w = self.input_layer_w
            self.output_layer_b = self.input_layer_b

        self.input_layer_w = ti.Matrix.field(self.n_hidden_layers, self.n_input_dims, Network.dtype, shape=self.grid_shape, needs_grad=True)
        self.input_layer_b = ti.Vector.field(self.n_hidden_layers, Network.dtype, shape=self.grid_shape, needs_grad=True)
        self.output_layer_w = ti.Matrix.field(self.n_output_dims, self.n_hidden_layers, Network.dtype, shape=self.grid_shape, needs_grad=True)
        self.output_layer_b = ti.Vector.field(self.n_output_dims, Network.dtype, shape=self.grid_shape, needs_grad=True)
    
        for i in range(self.n_hidden_layers - 1):    # (depth - 1) N x N matrix
            self.hidden_layers_w_l.append(
                ti.Matrix.field(self.n_neurons, self.n_neurons, Network.dtype, shape=self.grid_shape, needs_grad=True)
            )
            self.hidden_layers_b_l.append(
                ti.Vector.field(self.n_neurons, Network.dtype, shape=self.grid_shape, needs_grad=True)
            )

        # Seems inference is just forward... But not with `ti.ad.Tape()`.
        self.inference = self.all_forward

    def all_forward(self, input_vf: ti.template(), output_vf: ti.template()):
        ''' Forward all NN network.

        The first dimension of `input_vf` and `output_vf` is batchsize,
        while the remaining dimension is the NN array shape.
        Args:
            input_vf:   VectorField of shape (batchsize, network.grid_shape)
            output_vf:  VectorField of shape (batchsize, network.grid_shape)
        '''
        assert input_vf.shape == output_vf.shape
        io_shape = input_vf.shape
        self.batchsize = io_shape[0]
        self.mid_vf = ti.Vector.field(self.n_neurons, Network.dtype, shape=io_shape, needs_grad=True)
        if self.n_hidden_layers == 0:
            self.all_forward_last_layer(input_vf, output_vf, self.output_layer_w, self.output_layer_b)
        else:
            self.all_forward_one_layer(input_vf, self.mid_vf, self.input_layer_w, self.input_layer_b)
            for i in range(self.n_hidden_layers - 1):
                self.all_forward_one_layer(self.mid_vf, self.mid_vf, self.hidden_layers_w_l[i], self.hidden_layers_b_l[i])
            self.all_forward_last_layer(self.mid_vf, output_vf, self.output_layer_w, self.output_layer_b)

    @ti.kernel
    def all_forward_one_layer(self,
            input_vf: ti.template(),
            output_vf: ti.template(),
            weight_layer: ti.template(),
            bias_layer: ti.template()
        ):
        ''' Forward one layer in all NN network.
        
        Args:
            input_vf:    VectorField of shape (batchsize)
            output_vf:   VectorField of shape (batchsize)
        '''
        for I in ti.grouped(weight_layer):
            for i in range(self.batchsize):
                v = input_vf[i, I]
                w = weight_layer[I]
                b = bias_layer[I]
                # results
                r = w @ v + b
                self.activation(r)
        
            output_vf[i, I] = r

    @ti.kernel
    def all_forward_last_layer(self,
            input_vf: ti.template(),
            output_vf: ti.template(),
            weight_layer: ti.template(),
            bias_layer: ti.template()
        ):
        ''' Forward the last layer in all NN network.
        
        Args:
            input_vf:    VectorField of shape (batchsize)
            output_vf:   VectorField of shape (batchsize)
        '''
        for I in ti.grouped(weight_layer):
            for i in range(self.batchsize):
                v = input_vf[i, I]
                w = weight_layer[I]
                b = bias_layer[I]
                # results
                r = w @ v + b
                self.output_activation(r)
        
            output_vf[i, I] = r

    @property
    def layers_l(self):
        ''' All layers including weight and bias from input to output.
        '''
        l = [self.input_layer_w, self.input_layer_b]
        if self.n_hidden_layers == 0:
            return l
        for i in range(self.n_hidden_layers - 1):
            l.append(self.hidden_layers_w_l[i])
            l.append(self.hidden_layers_b_l[i])
        l.append(self.output_layer_w)
        l.append(self.output_layer_b)

# TODO
@ti.data_oriented
class Encoding:
    def __init__(self, json) -> None:
        self.otype = json['otype']

# TODO
@ti.data_oriented
class NetworkWithInputEncoding(Network):
    def __init__(self, n_input_dims, n_output_dims, encoding_json, network_json) -> None:
        self.encoding = Encoding(encoding_json)
        super().__init__(n_input_dims, n_output_dims, network_json)

##### activation functions ####

@ti.func
def relu(vec: ti.template()):
    vec *= (vec > 0)

@ti.func
def sigmoid(vec: ti.template()):
    vec = 1.0 / (1 + ti.exp(-vec))

@ti.func
def dummy(vec: ti.template()):
    pass

activation_dict = {
    'relu': relu,
    'sigmoid': sigmoid,
    'none': dummy
}
