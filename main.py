import taichi as ti
import numpy as np
import json
import imageio
import time

import tinn

#### Settings ####
config_path = 'data/dev.json'
ref_img_path = 'data/images/albert.jpg'
batch_size = 2**16
n_training_steps = 1000
n_input_dims = 2    # 2D image coord
n_output_dims = 3   # RGB color

vis_interval = 100
vis_scale = 0.25
##################

#### Load ####
with open(config_path) as f:
    config = json.load(f)
# need to rotate the image to fit taichi
ref_img = imageio.imread(ref_img_path)[::-1].T
if ref_img.dtype == np.uint8:
    ref_img = ref_img.astype(np.float32) / 255.
whc = ref_img.shape
if len(whc) == 2:   # black white
    ref_img = np.stack([ref_img, ref_img, ref_img], 2)
    whc = ref_img.shape
##################

ti.init(ti.cuda, device_memory_GB=8, debug=True)

# image
ref_img_ = ti.field(ti.f32, whc)
ref_img_.from_numpy(ref_img)

training_target = ti.Vector.field(n_output_dims, ti.f32, shape=batch_size)
training_batch = ti.Vector.field(n_input_dims, ti.f32, shape=batch_size)

prediction = ti.Vector.field(n_output_dims, ti.f32, (whc[0], whc[1]))
inference_batch = ti.Vector.field(n_input_dims, ti.f32, (whc[0], whc[1]))
tinn.utils.meshgrid_coord_2d(inference_batch)
# tinn.utils.sample_texture_from_coord_2d(ref_img_, inference_batch, prediction)

# model
loss = tinn.Loss(config['loss'])
optimizer = tinn.Optimizer(config['optimizer'])
# network = tinn.NetworkWithInputEncoding(n_input_dims, n_output_dims, config['encoding'], config['network'])
network = tinn.Network(n_input_dims, n_output_dims, config['network'])

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
            prev_time = time.time()

    # end train
    else:
        if not is_end:
            is_end = True
            print(f"End training {n_trained_step} steps.")

    canvas.set_image(prediction)

    # debug
    cursor_x, cursor_y = window.get_cursor_pos()
    cursor_xi = int(cursor_x * whc[0])
    cursor_yi = int(cursor_y * whc[1])
    if window.is_pressed('p'):
        canvas.set_image(ref_img_)
        print(f'cursor: {cursor_xi} \t {cursor_yi}')
        coord = inference_batch[cursor_xi, cursor_yi]
        print(f'inference_batch: {coord[0]:.4f} \t {coord[1]:.4f}')

    window.show()
