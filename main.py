import taichi as ti
import numpy as np
import json
import imageio
import time

import tinn

#### Settings ####
config_path = 'data/dev.json'
ref_img_path = 'data/images/albert-640.png'
batch_size = 2**10
n_training_steps = 1000
n_input_dims = 2    # 2D image coord
n_output_dims = 3   # RGB color

block_size = (128, 128) # each 8 x 8 pixel use a network.

vis_interval = 10
vis_scale = 1.
##################

#### Load ####
with open(config_path) as f:
    config = json.load(f)
# need to rotate the image to fit taichi
ref_img = imageio.imread(ref_img_path)[::-1].T
# crop the image to multiple of `block_size`
x_n_i = ref_img.shape[0] // block_size[0]
x_b_i = (ref_img.shape[0] % block_size[0]) // 2
y_n_i = ref_img.shape[1] // block_size[1]
y_b_i = (ref_img.shape[1] % block_size[1]) // 2
ref_img = ref_img[x_b_i:x_n_i * block_size[0], y_b_i: y_n_i * block_size[1]]

grid_shape = (x_n_i, y_n_i)     #### IMPORTANT

print(f"With the block_size {block_size}")
print(f"Crop the image to shape {ref_img.shape}")
print(f"** The `grid_shape` is {grid_shape}.")

# type conversion
if ref_img.dtype == np.uint8:
    ref_img = ref_img.astype(np.float32) / 255.
whc = ref_img.shape
if len(whc) == 2:   # black white
    ref_img = np.stack([ref_img, ref_img, ref_img], 2)
    whc = ref_img.shape
##################

ti.init(ti.cuda, device_memory_GB=8, debug=True)

# image
ref_img_ = ti.field(ti.f32, shape=whc)
ref_img_.from_numpy(ref_img)
vis_img = ti.Vector.field(3, ti.f32, shape=(whc[0], whc[1]))

train_io_shape = (batch_size, grid_shape[0], grid_shape[1])
training_target = ti.Vector.field(n_output_dims, ti.f32, shape=train_io_shape)
training_batch = ti.Vector.field(n_input_dims, ti.f32, shape=train_io_shape)

predict_io_shape = (block_size[0] * block_size[1], grid_shape[0], grid_shape[1])
prediction = ti.Vector.field(n_output_dims, ti.f32, shape=predict_io_shape)
inference_batch = ti.Vector.field(n_input_dims, ti.f32, shape=predict_io_shape)
tinn.utils.meshgrid_coord_2d(block_size[0], block_size[1], inference_batch)
# tinn.utils.sample_texture_from_coord_2d(ref_img_, inference_batch, prediction)

# model
loss = tinn.Loss(config['loss'])
optimizer = tinn.Optimizer(config['optimizer'])
# network = tinn.NetworkWithInputEncoding(n_input_dims, n_output_dims, config['encoding'], config['network'])
network = tinn.Network(n_input_dims, n_output_dims, config['network'], grid_shape)

trainer = tinn.Trainer(network, optimizer, loss)

# gui
vis_wh = (int(vis_scale * whc[0]), int(vis_scale * whc[1]))
window = ti.ui.Window('visualizer', vis_wh)
canvas = window.get_canvas()

prev_time = time.time()
n_trained_step = 0
is_end = False
while window.running:
    # train
    if n_trained_step < n_training_steps:
        n_trained_step += 1
        tinn.utils.generate_random_uniform_2d(training_batch)
        tinn.utils.sample_texture_from_coord_2d(ref_img_, training_batch, training_target)
        ctx = trainer.training_step(training_batch, training_target)

        # visualize
        if n_trained_step % vis_interval == 0:
            elapsed_time = time.time() - prev_time

            loss_val = ctx.loss[None]
            print(f'# Step {n_trained_step}\t Loss {loss_val:.4f} \t Time: {int(elapsed_time * 1e6)} us')

            network.inference(inference_batch, prediction)
            tinn.utils.flatten_2d_grid_ouput(block_size[0], block_size[1], prediction, vis_img)
            prev_time = time.time()

    # end train
    else:
        if not is_end:
            is_end = True
            print(f"End training {n_trained_step} steps.")

    canvas.set_image(vis_img)

    # debug
    cursor_x, cursor_y = window.get_cursor_pos()
    cursor_xi = int(cursor_x * whc[0])
    cursor_yi = int(cursor_y * whc[1])
    cursor_block_xni = cursor_xi // block_size[0]
    cursor_block_xmi = cursor_xi % block_size[0]
    cursor_block_yni = cursor_yi // block_size[1]
    cursor_block_ymi = cursor_yi % block_size[1]
    if window.is_pressed('p'):
        canvas.set_image(ref_img_)
        print(f'cursor: {cursor_xi} \t {cursor_yi}')
        coord = inference_batch[block_size[0] * cursor_block_ymi + cursor_block_xmi, cursor_block_xni, cursor_block_yni]
        print(f'inference_batch: {coord[0]:.4f} \t {coord[1]:.4f}')

    window.show()
