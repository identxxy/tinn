import taichi as ti
@ti.data_oriented
class Network:
    # precision type
    dtype = ti.f32
    
    def __init__(self, n_input_dims, n_output_dims, json, grid_shape=1) -> None:
        self.otype = json['otype']  # no use... just naive matrix multiplication.
        self.grid_shape = grid_shape
        self.activation = activation_dict[json['activation'].lower()]
        self.output_activation = activation_dict[json['output_activation'].lower()]
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
            self.init_weight(self.input_layer_w)
            self.layers_l = [self.output_layer_w, self.output_layer_b]

        else: 
            self.input_layer_w = ti.Matrix.field(self.n_neurons, self.n_input_dims, Network.dtype, shape=self.grid_shape, needs_grad=True)
            self.input_layer_b = ti.Vector.field(self.n_neurons, Network.dtype, shape=self.grid_shape, needs_grad=True)
            self.init_weight(self.input_layer_w)
            self.layers_l = [self.input_layer_w, self.input_layer_b]

            for i in range(self.n_hidden_layers - 1):    # (depth - 1) N x N matrix
                hwl = ti.Matrix.field(self.n_neurons, self.n_neurons, Network.dtype, shape=self.grid_shape, needs_grad=True)
                hbl = ti.Vector.field(self.n_neurons, Network.dtype, shape=self.grid_shape, needs_grad=True)
                self.init_weight(hwl)
                self.hidden_layers_w_l.append(hwl)
                self.hidden_layers_b_l.append(hbl)
                self.layers_l.append(hwl)
                self.layers_l.append(hbl)
            self.output_layer_w = ti.Matrix.field(self.n_output_dims, self.n_neurons, Network.dtype, shape=self.grid_shape, needs_grad=True)
            self.output_layer_b = ti.Vector.field(self.n_output_dims, Network.dtype, shape=self.grid_shape, needs_grad=True)
            self.init_weight(self.output_layer_w)
            self.layers_l.append(self.output_layer_w)
            self.layers_l.append(self.output_layer_b)

        # intermediate field
        self.mid_vf_train = None
        self.io_shape_train = None
        self.mid_vf_eval = None
        self.io_shape_eval = None

    def all_forward(self, input_vf: ti.template(), output_vf: ti.template()):
        ''' Forward all NN network. Using train field.

        The first dimension of `input_vf` and `output_vf` is batch_size,
        while the remaining dimension is the NN array shape.
        Args:
            input_vf:   VectorField of shape (batch_size, network.grid_shape)
            output_vf:  VectorField of shape (batch_size, network.grid_shape)
        '''
        assert input_vf.shape == output_vf.shape
        io_shape = input_vf.shape
        self.batch_size = io_shape[0]
        if (self.mid_vf_train == None) or self.io_shape_train != io_shape:
            self.io_shape_train = io_shape
            self.mid_vf_train = ti.Vector.field(self.n_neurons, Network.dtype, shape=io_shape, needs_grad=True)
            print(f'Network create new internal field of shape {io_shape}')
        if self.n_hidden_layers == 0:
            self.all_forward_last_layer(input_vf, output_vf, self.output_layer_w, self.output_layer_b)
        else:
            self.all_forward_one_layer(input_vf, self.mid_vf_train, self.input_layer_w, self.input_layer_b)
            for i in range(self.n_hidden_layers - 1):
                self.all_forward_one_layer(self.mid_vf_train, self.mid_vf_train, self.hidden_layers_w_l[i], self.hidden_layers_b_l[i])
            self.all_forward_last_layer(self.mid_vf_train, output_vf, self.output_layer_w, self.output_layer_b)

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
        if (self.mid_vf_eval == None) or self.io_shape_eval != io_shape:
            self.io_shape_eval = io_shape
            self.mid_vf_eval = ti.Vector.field(self.n_neurons, Network.dtype, shape=io_shape, needs_grad=True)
            print(f'Network create new internal field of shape {io_shape}')
        if self.n_hidden_layers == 0:
            self.all_forward_last_layer(input_vf, output_vf, self.output_layer_w, self.output_layer_b)
        else:
            self.all_forward_one_layer(input_vf, self.mid_vf_eval, self.input_layer_w, self.input_layer_b)
            for i in range(self.n_hidden_layers - 1):
                self.all_forward_one_layer(self.mid_vf_eval, self.mid_vf_eval, self.hidden_layers_w_l[i], self.hidden_layers_b_l[i])
            self.all_forward_last_layer(self.mid_vf_eval, output_vf, self.output_layer_w, self.output_layer_b)

    @ti.kernel
    def all_forward_one_layer(self,
            input_vf: ti.template(),
            output_vf: ti.template(),
            weight_layer: ti.template(),
            bias_layer: ti.template()
        ):
        ''' Forward one layer in all NN network.
        
        Args:
            input_vf:    VectorField of shape (batch_size)
            output_vf:   VectorField of shape (batch_size)
        '''
        for I in ti.grouped(weight_layer):
            for i in range(self.batch_size):
                v = input_vf[i, I]
                w = weight_layer[I]
                b = bias_layer[I]
                # results
                r = w @ v + b
                ti.static(self.activation)(r)
        
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
            input_vf:    VectorField of shape (batch_size)
            output_vf:   VectorField of shape (batch_size)
        '''
        for I in ti.grouped(weight_layer):
            for i in range(self.batch_size):
                v = input_vf[i, I]
                w = weight_layer[I]
                b = bias_layer[I]
                # results
                r = w @ v + b
                ti.static(self.output_activation)(r)
        
                output_vf[i, I] = r

    @property
    def get_layers_l(self):
        ''' All layers including weight and bias from input to output.
        '''
        return self.layers_l

    @ti.kernel
    def init_weight(self, f: ti.template()):
        # ref: https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/autodiff/diff_sph/diff_sph.py#L104
        q1 = ti.sqrt(6 / self.n_input_dims) * 0.1   ### IMPORTANT scale
        for I in ti.grouped(f):
            for n in ti.static(range(f.n)):
                for m in ti.static(range(f.m)):
                    f[I][n, m] = (ti.random() * 2 - 1) *q1

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
