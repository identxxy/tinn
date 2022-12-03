import taichi as ti
import time

@ti.kernel
def meshgrid_coord_2d(x: ti.u32, y: ti.u32, input_vf: ti.template()):
    ''' Generate a flattened 2D coord in the 1st dim of `input_vf`

    The `input_vf` should be of shape (x * y, grid_shape),
    totally `grid_shape` identical matrices generated.

    Args:
        x:          block_size[0]
        y:          block_size[1]
        input_vf:   VectorField of shape (x * y, grid_shape), and n = 2
    '''
    assert input_vf.shape[0] == x * y
    for I in ti.grouped(input_vf):
        xi = I[0] // x
        yi = I[0] % x
        input_vf[I][0] = (xi + 0.5) / x
        input_vf[I][1] = (yi + 0.5) / y

@ti.kernel
def generate_random_uniform_2d(input_vf: ti.template()):
    ''' Generate a uniform 2D random in the 1st dim of `input_vf`

    The `input_vf` should be of shape (x * y, grid_shape),
    totally `grid_shape` *different* distribution matrices generated.

    Args:
        input_vf:   VectorField of shape (x * y, grid_shape), and n = 2
    '''
    for I in ti.grouped(input_vf):
        input_vf[I][0] = ti.random(ti.f32)
        input_vf[I][1] = ti.random(ti.f32)

@ti.kernel
def sample_texture_from_2d_coord(
        texture: ti.template(),
        coord: ti.template(),
        result:ti.template()
    ):
    ''' Linear interpolation sampling.

    Args:
        texture:    ScalrField of shape (w, h, c)
        coord:      VectorField of shape (...), n = 2
        result:     VectorField of shape (...), n = 3
    '''
    assert coord.shape == result.shape, 'coord.shape != result.shape'
    for I in ti.grouped(coord):
        xf = coord[I][0] * texture.shape[0] - 0.5
        yf = coord[I][1] * texture.shape[1] - 0.5

        xi = ti.floor(xf, ti.i32)
        x1i = xi + 1
        yi = ti.floor(yf, ti.i32)
        y1i = yi + 1
        wx0 = xf - xi
        wx1 = x1i - xf
        wy0 = yf - yi
        wy1 = y1i - yf

        # repeat padding
        xi = ti.max(0, xi)
        x1i = ti.min(x1i, texture.shape[0]-1)
        yi = ti.max(0, yi)
        y1i = ti.min(y1i, texture.shape[1]-1)

        for c in ti.static(range(3)):
            xy00 = texture[xi, yi, c]
            xy01 = texture[xi, y1i, c]
            xy10 = texture[x1i, yi, c]
            xy11 = texture[xi, yi, c]
            result[I][c] = wx0 * wy0 * xy00 + wx0 * wy1 * xy01 + wx1 * wy0 * xy10 + wx1 * wy1 * xy11

@ti.kernel
def sample_texture_from_block_2d_coord_bw(
        texture: ti.template(),
        coord: ti.template(),
        result:ti.template()
    ):
    ''' Linear interpolation sampling.

    `coord` is coord in block, of shape (batch_size, grid_shape[0], grid_shape[1])

    Args:
        texture:    ScalrField of shape (w, h, c)
        coord:      VectorField of shape (batch_size, grid_shape[0], grid_shape[1]), n = 2
        result:     VectorField of shape (batch_size, grid_shape[0], grid_shape[1]), n = 3
    '''
    assert coord.shape == result.shape, 'coord.shape != result.shape'
    grid_dx = 1. / coord.shape[1]
    grid_dy = 1. / coord.shape[2]
    for n, i, j in coord:
        global_start_x = ti.f32(i) * grid_dx
        global_start_y = ti.f32(j) * grid_dy
        global_x = global_start_x + coord[n, i, j][1] * grid_dx
        global_y = global_start_y + coord[n, i, j][0] * grid_dy

        xf = global_x * texture.shape[0] - 0.5
        yf = global_y * texture.shape[1] - 0.5

        xi = ti.floor(xf, ti.i32)
        x1i = xi + 1
        yi = ti.floor(yf, ti.i32)
        y1i = yi + 1
        wx0 = xf - xi
        wx1 = x1i - xf
        wy0 = yf - yi
        wy1 = y1i - yf

        # repeat padding
        xi = ti.max(0, xi)
        x1i = ti.min(x1i, texture.shape[0]-1)
        yi = ti.max(0, yi)
        y1i = ti.min(y1i, texture.shape[1]-1)

        xy00 = texture[xi, yi]
        xy01 = texture[xi, y1i]
        xy10 = texture[x1i, yi]
        xy11 = texture[xi, yi]
        result[n,i,j].fill(wx0 * wy0 * xy00 + wx0 * wy1 * xy01 + wx1 * wy0 * xy10 + wx1 * wy1 * xy11)

@ti.kernel
def sample_texture_from_block_2d_coord(
        texture: ti.template(),
        coord: ti.template(),
        result:ti.template()
    ):
    ''' Linear interpolation sampling.

    `coord` is coord in block, of shape (batch_size, grid_shape[0], grid_shape[1])

    Args:
        texture:    ScalrField of shape (w, h, c)
        coord:      VectorField of shape (batch_size, grid_shape[0], grid_shape[1]), n = 2
        result:     VectorField of shape (batch_size, grid_shape[0], grid_shape[1]), n = 3
    '''
    assert coord.shape == result.shape, 'coord.shape != result.shape'
    grid_dx = 1. / coord.shape[1]
    grid_dy = 1. / coord.shape[2]
    for n, i, j in coord:
        global_start_x = ti.f32(i) * grid_dx
        global_start_y = ti.f32(j) * grid_dy
        global_x = global_start_x + coord[n, i, j][1] * grid_dx
        global_y = global_start_y + coord[n, i, j][0] * grid_dy

        xf = global_x * texture.shape[0] - 0.5
        yf = global_y * texture.shape[1] - 0.5

        xi = ti.floor(xf, ti.i32)
        x1i = xi + 1
        yi = ti.floor(yf, ti.i32)
        y1i = yi + 1
        wx0 = xf - xi
        wx1 = x1i - xf
        wy0 = yf - yi
        wy1 = y1i - yf

        # repeat padding
        xi = ti.max(0, xi)
        x1i = ti.min(x1i, texture.shape[0]-1)
        yi = ti.max(0, yi)
        y1i = ti.min(y1i, texture.shape[1]-1)

        for c in ti.static(range(3)):
            xy00 = texture[xi, yi, c]
            xy01 = texture[xi, y1i, c]
            xy10 = texture[x1i, yi, c]
            xy11 = texture[xi, yi, c]
            result[n,i,j][c] = wx0 * wy0 * xy00 + wx0 * wy1 * xy01 + wx1 * wy0 * xy10 + wx1 * wy1 * xy11

@ti.kernel
def flatten_2d_grid_ouput(x: ti.u32, y: ti.u32, input_vf: ti.template(), output_vf: ti.template()):
    ''' Flastten NN 2D array output from 3D to 2D field.

    Args:
        x:          block_size[0]
        y:          block_size[1]
        input_vf:   VectorField of shape (x * y, grid_shape[0], grid_shape[1])
        output_vf:  VectorField of shape (x * grid_shape[0], y * grid_shape[1])
    '''
    assert x * y == input_vf.shape[0], f'x * y {x*y}, input_vf.shape[0] {input_vf.shape[0]}'
    for i, j in output_vf:
        grid_xn = ti.i32(i // x)
        grid_xm = i % x
        grid_yn = ti.i32(j // y)
        grid_ym = j % y
        batch_i = ti.i32(grid_ym * x + grid_xm)
        # 2d vector can be fit to 3d rgb
        for c in ti.static(range(output_vf.n)):
            output_vf[i, j][c] = 0.
        for c in ti.static(range(input_vf.n)):
            output_vf[i, j][c] = input_vf[batch_i, grid_xn, grid_yn][c]
        
@ti.kernel
def flatten_2d_grid_ouput_bw(x: ti.u32, y: ti.u32, input_vf: ti.template(), output_vf: ti.template()):
    ''' Flastten NN 2D array output from 3D to 2D field.

    Args:
        x:          block_size[0]
        y:          block_size[1]
        input_vf:   VectorField of shape (x * y, grid_shape[0], grid_shape[1])
        output_vf:  VectorField of shape (x * grid_shape[0], y * grid_shape[1])
    '''
    assert x * y == input_vf.shape[0], f'x * y {x*y}, input_vf.shape[0] {input_vf.shape[0]}'
    for i, j in output_vf:
        grid_xn = ti.i32(i // x)
        grid_xm = i % x
        grid_yn = ti.i32(j // y)
        grid_ym = j % y
        batch_i = ti.i32(grid_ym * x + grid_xm)
        output_vf[i, j] = input_vf[batch_i, grid_xn, grid_yn]

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

class Timer:
    def __init__(self, name) -> None:
        self.fname = name

    def __enter__(self):
        self.time = time.time()
        print(f'Timing function {self.fname}')
    
    def __exit__(self, type, value, trace):
        elapsed_time = time.time() - self.time
        print(f"elasped_time {elapsed_time * 1e3} ms")