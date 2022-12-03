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
            at: ti.template()
        ):
        for I in ti.grouped(prediction):
            # ugly hack
            yes = 1
            for d in ti.static(range(at.shape[1])):
                if I[1+d] < at[0, d] or I[1+d] >= at[1, d]: # 0 is the batch_size
                    yes = 0
            if yes == 1:
                self.loss_func(scale, prediction[I], target[I], values[I])

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
