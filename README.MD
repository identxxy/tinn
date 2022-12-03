Try my best to implement some basic part of [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) using [Taichi](https://docs.taichi-lang.org/) language.

A repo for the first Taichi [Hackthon](https://forum.taichi-lang.cn/t/topic/3506).


---


团队名：钛牛牛

项目名：tinn

项目介绍：

1. 背景
   
    近年来，随着人们对神经网络的性能和灵活性的需求逐渐提高，越来越多的研究人员开始自己编写cuda kernel来加速经典深度学习框架，甚至抛弃基于Python的深度学习框架，转而使用cuda版本的实现,。NVlabs发布的tiny-cuda-nn是最常用的cuda版本的神经网络框架。  
    
    本项目将参考tiny-cuda-nn的框架，基于Taichi的autodiff机制，实现一个简单的神经网络训练框架。

2. 动机 
 
    由于Taichi的优势在于其并行计算的灵活性，而且可以很方便的与主流深度学习框架如PyTorch进行交互，所以本项目的设计目标并不是使用Taichi实现类似PyTorch的功能，而是希望可以更加灵活。在参考更为底层的tiny-cuda-nn库的架构的同时，由于Taichi的语言特性，与tiny-cuda-nn有不同的设计侧重。

    **具体来说，本项目的优势在于构建大量微型神经网络，而不是单个大型神经网络。**


    大量的微型神经网络可以使得模型的梯度传播更加灵活，类似于构建动态的计算图(compute graph)，在很多场景中可以解决梯度稀疏的问题。举个例子，在可导渲染中，用户可以使用光线步进(ray marching)的策略来选取特定区域的神经网络进行优化。

    > 单个大型神经网络的使用场景包括图片识别，NLP等经典深度学习场景。  
    > 大量微型神经网络的使用场景包括NeRF加速([KiloNeRF](https://creiser.github.io/kilonerf/))，集群多智能体强化学习(Swarm Multi-agent Reinforcement Learning (我瞎编的...)，比如给蚁群装上钛牛牛，让每个蚂蚁可以简单思考。 

3. 完成的功能
   
    整个tiny-cuda-nn的功能复杂，性能优化与硬件架构紧密相关，使用Taichi语言难以达到相似的性能，因此，本次参赛项目只致力于使用Taichi实现其中一个简单例程: `mlp_learning_an_image`

    即使用神经网络(MLP)学习一张照片。（详情参考[tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn))

    完成的功能包括:

    - 损失函数 Loss: L1, L2, RelativeL2
    - 优化器 Optimizer: SGD
    - **神经网络阵列**：神经网络**阵列**训练/推断，神经网络**单个**训练/推断
    - 编码器 Encoding: Position Encoding(Frequency Encoding)


    > **神经网络阵列** 的操作是本项目的创新点之一。

4. 实现
   
    在本项目中，网络参数的储存方式是AoS中的Struct，参数的本质是小型矩阵，使用MatrixField储存。由于Taichi在编译时会展开矩阵运算，所以当一层网络的输入输出维度`n_in_dims * n_out_dims > 32`时，Taichi会提示可能编译时间过长。*这是本项目的feature ：)*

    虽然将参数使用ScalarField储存，可以构建更宽的神经网络，如官方的[diff_sph](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/autodiff/diff_sph/diff_sph.py)例程（这个名字...谁能想到这个例程里面有神经网络呢...)，但其实更宽的网络可以考虑使用PyTorch实现，而且Taichi kernel可以直接操作Torch tensor的内存，无需拷贝，参考[blog](https://www.modb.pro/db/462148)。

    因此，将参数使用MatrixField存储很有意义，使得用户可以方便地创建大量微型神经网络。

    由此带来的弊端除了上文提到的编译警告，还有编译时间。由于每一层神经网络都是单独的`field`，运行时会编译出很多`ti.kernel`，初次运行时需要一段时间编译。每个`field`在Python Scope的存储方式是list。

    项目主要内容：

    - main.py
    
        训练脚本，读取图片与训练参数，使用GGUI可视化训练过程。

    - tinn库

        实现以下类  
        Loss  
        Optimizer  
        Network  
        Trainer  
        Encoding  
        NetworkWithInputEncoding

5. 未来方向

    实现Adam优化器。  
    尝试更多编码(Encoding)。


Dependency:  
    numpy  
    imageio  
    json

---

