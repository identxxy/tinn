import taichi as ti

ti.init(ti.cuda, random_seed=0)

n = 4
res = 2

vif = ti.Vector.field(2, ti.f32, shape=res, needs_grad=True)
vof = ti.Vector.field(3, ti.f32, shape=res, needs_grad=True)
tgt = ti.Vector.field(3, ti.f32, shape=res, needs_grad=True)

i_w = ti.Matrix.field(n, 2, ti.f32, shape=(), needs_grad=True)
i_b = ti.Vector.field(n, ti.f32, shape=(), needs_grad=True)
h1_w = ti.Matrix.field(n, n, ti.f32, shape=(), needs_grad=True)
h1_b = ti.Vector.field(n, ti.f32, shape=(), needs_grad=True)
h2_w = ti.Matrix.field(n, n, ti.f32, shape=(), needs_grad=True)
h2_b = ti.Vector.field(n, ti.f32, shape=(), needs_grad=True)
o_w = ti.Matrix.field(3, n, ti.f32, shape=(), needs_grad=True)
o_b = ti.Vector.field(3, ti.f32, shape=(), needs_grad=True)

loss_sf = ti.field(ti.f32, shape=(), needs_grad=True)

@ti.func
def relu(vec: ti.template()):
    vec *= (vec > 0)

@ti.kernel
def forward(vif: ti.template(), vof: ti.template()):
    for I in ti.grouped(vif):
        # 
        v = vif[I]
        wi = i_w[None]
        bi = i_b[None]
        w1 = h1_w[None]
        b1 = h1_b[None]
        w2 = h2_w[None]
        b2 = h2_b[None]
        wo = o_w[None]
        bo = o_b[None]
        # results
        ri = wi @ v + bi
        relu(ri)
        rh = w1 @ ri + b1
        relu(rh)
        rh = w2 @ rh + b2
        relu(rh)
        ro = wo @ rh + bo
        relu(ro)
        
        vof[I] = ro


@ti.kernel
def l2_loss(vof: ti.template(), tgt: ti.template(), loss_sf: ti.template()):
    for I in ti.grouped(vof):
        loss_sf[None] += ((tgt[I] - vof[I])**2).sum()


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
def step(lr: ti.f32):
    for I in ti.grouped(i_w):
        i_w[I] -= lr * i_w.grad[I]
    for I in ti.grouped(i_b):
        i_b[I] -= lr * i_b.grad[I]

    for I in ti.grouped(h1_w):
        h1_w[I] -= lr * h1_w.grad[I]
    for I in ti.grouped(h1_b):
        h1_b[I] -= lr * h1_b.grad[I]

    for I in ti.grouped(h2_w):
        h2_w[I] -= lr * h2_w.grad[I]
    for I in ti.grouped(h2_b):
        h2_b[I] -= lr * h2_b.grad[I]
        
    for I in ti.grouped(o_w):
        o_w[I] -= lr * o_w.grad[I]
    for I in ti.grouped(o_b):
        o_b[I] -= lr * o_b.grad[I]


vif[0][0] = 1.0
vif[0][1] = -2.0
vif[1][0] = 2.0
vif[1][1] = -4.0


init_mat_random(i_w)
init_vec_random(i_b)

init_mat_random(h1_w)
init_vec_random(h1_b)

init_mat_random(h2_w)
init_vec_random(h2_b)

init_mat_random(o_w)
init_vec_random(o_b)

# forward(vf)
for i in range(100):
    with ti.ad.Tape(loss_sf):
        forward(vif, vof)     # call twice
        l2_loss(vof, tgt, loss_sf)
    step(1e-3)
    print('==== loss ====')
    print(loss_sf[None])

# print('==== weight ====')
# print('i_w', i_w)
# print('i_b', i_b)
# print('h1_w', h1_w)
# print('h1_b', h1_b)
# print('h2_w', h2_w)
# print('h2_b', h2_b)
# print('==== grad ====')
# print('i_w.grad', i_w.grad)
# print('i_b.grad', i_b.grad)
# print('h1_w.grad', h1_w.grad)
# print('h1_b.grad', h1_b.grad)
# print('h2_w.grad', h2_w.grad)
# print('h2_b.grad', h2_b.grad)