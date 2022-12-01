import taichi as ti

ti.init(ti.cuda, random_seed=0)

n_in_dims = 2
n_out_dims = 3

depth = 4   # must > 0
width = 4
batch_size = 64
grid_shape = (3, 3)

lr = 1e-5

#### input ouput ####
io_shape = (batch_size, grid_shape[0], grid_shape[1])
vif = ti.Vector.field(n_in_dims, ti.f32, shape=io_shape, needs_grad=True)
vof = ti.Vector.field(n_out_dims, ti.f32, shape=io_shape, needs_grad=True)
tgt = ti.Vector.field(n_out_dims, ti.f32, shape=io_shape, needs_grad=True)
#####################

#### network structure #######
# first layer
i_w = ti.Matrix.field(width, n_in_dims, ti.f32, shape=grid_shape, needs_grad=True)
i_b = ti.Vector.field(width, ti.f32, shape=grid_shape, needs_grad=True)
# last layer
o_w = ti.Matrix.field(n_out_dims, width, ti.f32, shape=grid_shape, needs_grad=True)
o_b = ti.Vector.field(n_out_dims, ti.f32, shape=grid_shape, needs_grad=True)
# hidden layer
h_w_l = []
h_b_l = []
for i in range(depth - 1):    # (depth - 1) N x N matrix
    h_w_l.append(ti.Matrix.field(width, width, ti.f32, shape=grid_shape, needs_grad=True))
    h_b_l.append(ti.Vector.field(width, ti.f32, shape=grid_shape, needs_grad=True))
# a temp buffer for intermediate result
vmf = ti.Vector.field(width, ti.f32, shape=io_shape, needs_grad=True)

loss_sf = ti.field(ti.f32, shape=(), needs_grad=True)
###############################


@ti.func
def relu(vec: ti.template()):
    vec *= (vec > 0)

@ti.func
def sigmoid(vec: ti.template()):
    vec = 1.0 / (1 + ti.exp(-vec))

activation = relu
@ti.kernel
def forward_layer(vif: ti.template(), vof: ti.template(), wf: ti.template(), bf: ti.template()):
    for I in ti.grouped(wf):
        for i in range(vif.shape[0]):
            v = vif[i, I]
            w = wf[I]
            b = bf[I]
            # results
            r = w @ v + b
            activation(r)
        
            vof[i, I] = r

def forward(vif, vof):
    forward_layer(vif, vmf, i_w, i_b)
    for i in range(depth - 1):
        forward_layer(vmf, vmf, h_w_l[i], h_b_l[i])
    forward_layer(vmf, vof, o_w, o_b)

@ti.kernel
def l2_loss(scale: ti.f32, vof: ti.template(), tgt: ti.template(), loss_sf: ti.template()):
    for I in ti.grouped(vof):
        loss_sf[None] += ((tgt[I] - vof[I])**2).sum() * scale


# Potential bug...
@ti.kernel
def init_mat_random(f: ti.template()):
    for I in ti.grouped(f):
        for n in ti.static(range(f.n)):
            for m in ti.static(range(f.m)):
                f[I][n, m] = ti.random()

@ti.kernel
def init_vec_random(f: ti.template()):
    for I in ti.grouped(f):
        for n in ti.static(range(f.n)):
            f[I][n] = ti.random()

@ti.kernel
def step_layer(lr: ti.f32, f: ti.template()):
    for I in ti.grouped(f):
        f[I] -= lr * f.grad[I]

def step(lr):
    step_layer(lr, i_w)
    step_layer(lr, i_b)
    for i in range(depth - 1):
        step_layer(lr, h_w_l[i])
        step_layer(lr, h_b_l[i])
    step_layer(lr, o_w)
    step_layer(lr, o_b)

#### init ####
init_vec_random(vif)

init_mat_random(i_w)
init_vec_random(i_b)

for i in range(depth - 1):
    init_mat_random(h_w_l[i])
    init_vec_random(h_b_l[i])

init_mat_random(o_w)
init_vec_random(o_b)
###############

#### train ####
# forward(vf)
n_element = io_shape[0] * io_shape[1] * io_shape[2]
for e in range(10):
    with ti.ad.Tape(loss_sf):
        forward(vif, vof)
        l2_loss(1. / n_element, vof, tgt, loss_sf)
    step(lr)

    print(f'==== loss ==== epoch {e}')
    print(loss_sf[None])

print('==== weight ====')
print('i_w', i_w)
print('i_b', i_b)
print('o_w', o_w)
print('o_b', o_b)
print('==== grad ====')
print('i_w.grad', i_w.grad)
print('i_b.grad', i_b.grad)
print('o_w.grad', o_w.grad)
print('o_b.grad', o_b.grad)
###################
print('==============')
print(f'depth: {depth}')
print(f'width: {width}')
print(f'batch_size: {batch_size}')
print(f'grid_shape: {grid_shape}')
print('==============')