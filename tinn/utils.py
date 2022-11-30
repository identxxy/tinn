import taichi as ti

@ti.kernel
def meshgrid_coord_2d(input_vf: ti.template()):
    for I in ti.grouped(input_vf):
        input_vf[I][0] = (I[0] + 0.5) / input_vf.shape[0]
        input_vf[I][1] = (I[1] + 0.5) / input_vf.shape[1]

@ti.kernel
def generate_random_uniform_2d(input_vf: ti.template()):
    for I in ti.grouped(input_vf):
        input_vf[I][0] = ti.random(ti.f32)
        input_vf[I][1] = ti.random(ti.f32)

@ti.kernel
def sample_texture_from_coord_2d(texture: ti.template(), coord: ti.template(), result:ti.template()):
    ''' Linear interpolation sampling.

    Args:
        texture:    ScalrField of shape (w, h, c)
        coord:      VectorField of shape (2, ...)
        result:     VectorField of shape (c, ...)
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
