Try my best to implement some basic part of [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) using [Taichi](https://docs.taichi-lang.org/) language.

A repo for the first Taichi [Hackthon](https://forum.taichi-lang.cn/t/topic/3506).


---


团队名：钛牛牛

项目名：tinn

项目介绍：

1. 背景
   
    Taichi语言提供了自动求导功能，理论上可以使用Taichi构建神经网络。本项目旨在使用Taichi实现一个简单的神经网络训练框架。

    近年来，随着人们对神经网络的性能和灵活性的需求逐渐提高，越来越多的研究人员开始自己编写cuda kernel来加速经典深度学习框架，
    甚至抛弃基于Python的深度学习框架，转而使用cuda版本的实现,。NVlabs发布的tiny-cuda-nn是最常用的cuda版本的神经网络框架。
    本项目将参考tiny-cuda-nn，使用Taichi实现一个简单的神经网络训练框架。

2. 动机 
 
    由于Taichi的优势在于其并行计算的灵活性，而且可以很方便的与主流深度学习框架如PyTorch进行交互，
所以本项目的设计目标并不是使用Taichi实现类似PyTorch的功能，而是希望可以更加灵活。
整个项目参考了更为底层的tiny-cuda-nn库的架构，但由于Taichi的语言特性，与tiny-cuda-nn有不同的设计侧重。

    **具体来说，本项目的优势在于构建大量微型神经网络，而不是单个大型神经网络。**

    大量的微型神经网络可以使得模型的梯度传播更加灵活，比如构建动态的计算图(compute graph)。
    举个例子，在可导渲染中，用户可以使用光线步进(ray marching)来选取特定区域的神经网络进行优化。

2. 完成的功能
   
    整个tiny-cuda-nn的功能复杂，本次参赛项目只致力于使用Taichi实现一个简单例程: mlp_learning_an_image
    即使用神经网络(MLP)学习一张照片。（详情参考tiny-cuda-nn)

3. 实现
   
    在本项目中，网络参数的储存方式是AoS中的Struct，参数的本质是小型矩阵，使用MatrixField储存，
    由于Taichi在编译时会展开矩阵运算，所以当一层网络的输入输出维度`n_in_dims * n_out_dims > 32`时，
    Taichi会提示可能编译时间过长。这不是问题，这是本项目的feature。

    虽然将参数使用ScalarField储存，可以构建更宽的神经网络，如官方的[diff_sph](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/autodiff/diff_sph/diff_sph.py)例程（这个名字...谁能想到这个例程里面有神经网络呢...)，但其实更宽的网络可以考虑使用PyTorch实现，而且Taichi kernel可以直接操作Torch tensor的内存，无需拷贝，参考[blog](https://www.modb.pro/db/462148)。

    所以说，将参数使用MatrixField存储是很有意义的，使得用户可以方便地创建大量微型神经网络。

    - main.py
    
        训练脚本，读取图片与训练参数，使用GGUI可视化训练过程。

    - tinn库

        实现以下类  
        Loss  
        Optimizer  
        Network  
        Trainer  

Dependency:  
    numpy  
    imageio  
    json

---

