import taichi as ti

@ti.data_oriented
class Loss:
    def __init__(self, json) -> None:
        self.otype = json['otype']
        self.loss_func = loss_dict[self.otype.lower()]
    
    @ti.kernel
    def loss_all(self,
            scale: ti.f32,
            prediction: ti.template(),
            target: ti.template(),
            values: ti.template()
        ):
        for I in ti.grouped(prediction):
            self.loss_func(scale, prediction[I], target[I], values[I])

    @ti.kernel
    def loss_one(self,
            scale: ti.f32,
            prediction: ti.template(),
            target: ti.template(),
            values: ti.template(),
            mask: ti.template()
        ):
        for I in ti.grouped(mask):
            if mask[I] == 1:
                for i in range(prediction.shape[0]):
                    self.loss_func(scale, prediction[i, I], target[i, I], values[i, I])

##### loss functions ####
@ti.func
def l1(
        scale: ti.f32,
        prediction: ti.template(),
        target: ti.template(),
        values: ti.template()
    ):
    values = (ti.abs(prediction - target)).sum() * scale

@ti.func
def l2(
        scale: ti.f32,
        prediction: ti.template(),
        target: ti.template(),
        values: ti.template()
    ):
    values = ((prediction - target)**2).sum() * scale

@ti.func
def relative_l2(
        scale: ti.f32,
        prediction: ti.template(),
        target: ti.template(),
        values: ti.template()
    ):
    l2_loss = ((prediction - target)**2).sum()
    det = (prediction**2).sum() + 1e-2
    values = l2_loss / det * scale

loss_dict = {
    'l1': l1,
    'l2': l2,
    'relativel2': relative_l2
}
