Try my best to implement some basic part of [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) using [Taichi](https://docs.taichi-lang.org/) language.

A repo for the first Taichi [Hackthon](https://forum.taichi-lang.cn/t/topic/3506).


---

```
团队名：钛牛牛

项目名：tinn

项目介绍：

1. 背景
Taichi语言提供了自动求导功能，理论上可以使用Taichi构建神经网络，本项目旨在使用Taichi实现一个简单的神经网络训练框架。

近年来，随着人们对神经网络的性能和灵活性的需求逐渐提高，越来越多的研究人员开始自己编写cuda kernel来加速经典深度学习框架，甚至抛弃基于Python的深度学习框架，转而使用cuda版本的实现。NVlabs发布的tiny-cuda-nn是最常用的cuda版本的神经网络框架，本项目将参考tiny-cuda-nn，使用Taichi实现一个简单的神经网络训练框架。

2. 完成的功能
整个tiny-cuda-nn的功能复杂，本次参赛项目只致力于使用Taichi实现一个简单例程: mlp_learning_an_image
即使用神经网络(MLP)学习一张照片。（详情参考tiny-cuda-nn)

3. 实现
    3.1 main.py
    训练脚本，读取图片与训练参数，使用GGUI可视化训练过程。

    3.2 tinn库
    实现以下类
        Loss
        Optimizer
        Network
        Trainer

Dependency:
    numpy
    imageio
    json

```
