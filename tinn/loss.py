import taichi as ti

class Loss:
    def __init__(self, json) -> None:
        self.otype = json['otype']
        self.kernel = loss_dict[self.otype.lower()]
    
##### loss functions ####
@ti.kernel
def l1(
        scale: ti.f32,
        prediction: ti.template(),
        target: ti.template(),
        values: ti.template()
    ):
    for I in ti.grouped(prediction):
        values[I] = (ti.abs(prediction[I] - target[I])).sum() * scale

@ti.kernel
def l2(
        scale: ti.f32,
        prediction: ti.template(),
        target: ti.template(),
        values: ti.template()
    ):
    for I in ti.grouped(prediction):
        values[I] = ((prediction[I] - target[I])**2).sum() * scale

@ti.kernel
def relative_l2(
        scale: ti.f32,
        prediction: ti.template(),
        target: ti.template(),
        values: ti.template()
    ):
    for I in ti.grouped(prediction):
        l2_loss = (prediction[I] - target[I])**2
        det = (prediction[I]**2 + 0.01)
        values[I] = (l2_loss / det).sum() * scale

loss_dict = {
    'l1': l1,
    'l2': l2,
    'relativel2': relative_l2
}
