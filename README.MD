# tinn

A library to create a large number of micro-scale neural networks using [Taichi](https://docs.taichi-lang.org/) language.  

Framework is inspired by [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).

## Feature

The **parrallelization** is among **all the networks**, rather than the matrix computation inside one NN, between layers.  

All the matrix computation between nerual network layers are **unrolled** by Taichi at compile time, executing in all parrallel threads,
which means the maxtrix size cannot be too large.  
When the matrix size exceed 32, e.g. for a 6-input 6-output layer, which means 36 matrix elements (>32), Taichi will give out a compile warning.  

## Demo

Network config is located at `data/*.json`

### MLP learn an image


```
python mlp_learn_an_image.py
```

For simplication, settings are located at the head of `mlp_learn_an_image.py`.  

#### Control

|key/mouse| visualied field | description |
|-|-|-|
`<space>` | - |train all NN | 
click LMB | - | train the NN at cursor postion| 
`t` | train_batch | random xy coord [0, 1)|
`i` | inference_batch | meshgrid xy coord [0, 1) |
`g` | train_target | sampled pixel from reference |
`r` | reference_img | ground truth GT|
`l` | inference_loss | inference loss to GT |
`p` | - | print profiler info |
||||

