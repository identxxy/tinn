# Special version for Hackthon demo, add surprise.

import taichi as ti
import numpy as np
import json
import imageio
import time

import tinn

#### Settings ####
config_path = 'data/dev.json'
ref1_img_path = 'data/images/albert-fit.jpg'
ref0_img_path = 'data/images/hu.jpg'
batch_size = 2 ** 8
n_input_dims = 2    # 2D image coord
n_output_dims = 3   # RGB color

block_size = (16, 16) # each block_size pixel use a network.

vis_interval = 10
vis_scale = 5

change_ref_flag = 0
##################

#### Load ####
with open(config_path) as f:
    config = json.load(f)

# need to rotate the image to fit taichi
def read_ref_image_path(ref_img_path, block_size):
    ref_img = imageio.imread(ref_img_path)[::-1].T
    print(f"Read image of shape {ref_img.shape}")
    is_bw = len(ref_img.shape) == 2
    if not is_bw and ref_img.shape[0] == 3:
        ref_img = np.moveaxis(ref_img, 0, 2) 
        print(f"Rearrange image of shape {ref_img.shape}")
    # crop the image to multiple of `block_size`
    x_n_i = ref_img.shape[0] // block_size[0]
    x_b_i = (ref_img.shape[0] % block_size[0]) // 2
    y_n_i = ref_img.shape[1] // block_size[1]
    y_b_i = (ref_img.shape[1] % block_size[1]) // 2
    ref_img = ref_img[x_b_i:x_b_i + x_n_i * block_size[0], y_b_i: y_b_i + y_n_i * block_size[1]]

    grid_shape = (x_n_i, y_n_i)     #### IMPORTANT

    print(f"With the block_size {block_size}")
    print(f"Crop the image to shape {ref_img.shape}")
    print(f"** The `grid_shape` is {grid_shape}.")

    # type conversion
    if ref_img.dtype == np.uint8:
        ref_img = ref_img.astype(np.float32) / 255.
    if is_bw:   # black white
        ref_img = np.stack([ref_img, ref_img, ref_img], 2)
    return ref_img, grid_shape

ref_img, grid_shape = read_ref_image_path(ref0_img_path, block_size)
whc = ref_img.shape
##################

ti.init(ti.cuda, device_memory_GB=8, debug=True, random_seed=0, kernel_profiler = True)

# image
ref_img_ = ti.field(ti.f32, shape=whc)
ref_img_.from_numpy(ref_img)
vis_img_rgb = ti.Vector.field(3, ti.f32, shape=(whc[0], whc[1]))
vis_img_bw = ti.field(ti.f32, shape=(whc[0], whc[1]))
vis_img = vis_img_rgb

train_io_shape = (batch_size, grid_shape[0], grid_shape[1])
training_target = ti.Vector.field(n_output_dims, ti.f32, shape=train_io_shape)
training_batch = ti.Vector.field(n_input_dims, ti.f32, shape=train_io_shape)

predict_io_shape = (block_size[0] * block_size[1], grid_shape[0], grid_shape[1])
prediction = ti.Vector.field(n_output_dims, ti.f32, shape=predict_io_shape)
ground_truth = ti.Vector.field(n_output_dims, ti.f32, shape=predict_io_shape)
eval_losses = ti.field(ti.f32, shape=predict_io_shape)
inference_batch = ti.Vector.field(n_input_dims, ti.f32, shape=predict_io_shape)
tinn.utils.meshgrid_coord_2d(block_size[0], block_size[1], inference_batch)
tinn.utils.sample_texture_from_block_2d_coord(ref_img_, inference_batch, ground_truth)

# non-random train
# tinn.utils.meshgrid_coord_2d(block_size[0], block_size[1], training_batch)    # test
# tinn.utils.sample_texture_from_block_2d_coord(ref_img_, training_batch, training_target)  # 1 ms

mask = ti.field(ti.u8, shape=grid_shape)

# model
loss = tinn.Loss(config['loss'])
optimizer = tinn.Optimizer(config['optimizer'])
network = tinn.NetworkWithInputEncoding(n_input_dims, n_output_dims, config['encoding'], config['network'], grid_shape)
# network = tinn.Network(n_input_dims, n_output_dims, config['network'], grid_shape)

trainer = tinn.Trainer(network, optimizer, loss)

# gui
vis_wh = (int(vis_scale * whc[0]), int(vis_scale * whc[1]))
window = ti.ui.Window('visualizer', vis_wh)
gui = window.get_gui()
canvas = window.get_canvas()

prev_time = time.time()
n_trained_step = 0
is_end = False
while window.running:
    cursor_x, cursor_y = window.get_cursor_pos()
    cursor_xi = int(cursor_x * whc[0])
    cursor_yi = int(cursor_y * whc[1])
    cursor_block_xni = cursor_xi // block_size[0]
    cursor_block_xmi = cursor_xi % block_size[0]
    cursor_block_yni = cursor_yi // block_size[1]
    cursor_block_ymi = cursor_yi % block_size[1]

    gui.text("Press <space> to train all NN.")
    gui.text("Click a block to train that one NN.")

    # change a reference image
    if gui.button('change reference'):
        change_ref_flag = 1 - change_ref_flag
        if change_ref_flag == 1:
            ref_img, grid_shape = read_ref_image_path(ref1_img_path, block_size)
            ref_img_.from_numpy(ref_img)
            tinn.utils.sample_texture_from_block_2d_coord(ref_img_, inference_batch, ground_truth)
        else:
            ref_img, grid_shape = read_ref_image_path(ref0_img_path, block_size)
            ref_img_.from_numpy(ref_img)
            tinn.utils.sample_texture_from_block_2d_coord(ref_img_, inference_batch, ground_truth)

    # train
    if window.is_pressed(' '):
        selected_block = (cursor_block_xni, cursor_block_yni)
        n_trained_step += 1
        tinn.utils.generate_random_uniform_2d(training_batch)   # 1 ms
        tinn.utils.sample_texture_from_block_2d_coord(ref_img_, training_batch, training_target)  # 1 ms
        ctx = trainer.training_step_all(training_batch, training_target)
        loss_val = ctx.loss[None]
        print(f'# Step ALL NN {n_trained_step} \t loss: {loss_val:.4f}')

    if window.is_pressed(ti.ui.LMB):
        mask.fill(0)
        mask[cursor_block_xni, cursor_block_yni] = 1
        tinn.utils.generate_random_uniform_2d(training_batch)   # 1 ms
        tinn.utils.sample_texture_from_block_2d_coord(ref_img_, training_batch, training_target)  # 1 ms
        ctx = trainer.training_step_one(training_batch, training_target, mask)
        loss_val = ctx.loss[None]
        print(f'** Step NN grid id {cursor_block_xni} {cursor_block_yni} \t loss: {loss_val:.4f}')

    # visualize
    curr_time = time.time()
    if curr_time - prev_time > 0.1: # 10 FPS
        prev_time = curr_time
        network.inference_all(inference_batch, prediction)  # 1000 ms...
        tinn.utils.flatten_2d_grid_ouput(block_size[0], block_size[1], prediction, vis_img_rgb) # 1ms
        vis_img = vis_img_rgb

    if window.is_pressed('t'):
        if batch_size != block_size[0] * block_size[1]:
            print("Cannot visualize training_batch, batch_size not fit block_size")
        else:
            tinn.utils.flatten_2d_grid_ouput(block_size[0], block_size[1], training_batch, vis_img_rgb)
            vis_img = vis_img_rgb
    if window.is_pressed('g'):
        if batch_size != block_size[0] * block_size[1]:
            print("Cannot visualize training_target, batch_size not fit block_size")
        else:
            tinn.utils.flatten_2d_grid_ouput(block_size[0], block_size[1], training_target, vis_img_rgb)
            vis_img = vis_img_rgb
    if window.is_pressed('i'):
        tinn.utils.flatten_2d_grid_ouput(block_size[0], block_size[1], inference_batch, vis_img_rgb)
        vis_img = vis_img_rgb
    if window.is_pressed('r'):
        tinn.utils.flatten_2d_grid_ouput(block_size[0], block_size[1], ground_truth, vis_img_rgb)
        vis_img = vis_img_rgb

    if window.is_pressed('l'):
        loss.loss_all(1., prediction, ground_truth, eval_losses)
        tinn.utils.flatten_2d_grid_ouput_bw(block_size[0], block_size[1], eval_losses, vis_img_bw) # 1ms
        vis_img = vis_img_bw

    if window.is_pressed('p'):
        ti.profiler.print_kernel_profiler_info()

    canvas.set_image(vis_img)
    window.show()
