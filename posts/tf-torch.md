---
title: tensorflow与pytorch代码差异
mathjax: true
toc: true
categories:
  - 深度学习
date: 2020-07-05 20:40:33
tags:
- Tensorflow
- Pytorch
---
可能会长期更新,因为经常需要从`pytorch`偷代码翻译成`tensorflow`😑因此记录一下差异的地方.

<!--more-->

####   1. `torch`中`nn.Conv2d`的`groups`参数

`torch`中`groups`控制输入和输出之间的连接,`in_channels`和`out_channels`必须都可以被组整除.
- `groups=1` 传统的卷积方式.
- `groups=2` 等效于并排设置两个`conv`层，每个`conv`层看到一半的输入通道，并产生一半的输出通道，并且随后将它们都连接在一起.
- `groups=in_channels` 每个输入通道都有自己的滤波器.


等价写法:
```python
nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, 
          stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)

kl.DepthwiseConv2D(kernel_size=kernel_size,
                  strides=stride, padding='same', use_bias=False)
```

NOTE:

这里`pytorch`生成的卷积核`shape = [out_channel, 1, kh, kw]`
这里`tflite`生成的卷积核`shape = [1, kh, kw, out_channel]`


#### 2. `nn.AdaptiveAvgPool2d`与`kl.GlobalAveragePooling2D`

当`nn.AdaptiveAvgPool2d(1)`时和`kl.GlobalAveragePooling2D()`相同,但是注意`torch`的输出是保持`4`维的,而`tensorflow`不保持维度.

等价写法:
```python
x=nn.AdaptiveAvgPool2d(1)(x)
# -----------------------------
pool=kl.GlobalAveragePooling2D()
x=k.backend.expand_dims(k.backend.expand_dims(pool(x),1),1)
```

当然直接修改`GlobalAveragePooling2D`里,添加`keepdims=true`参数也可以.


#### tf.contrib.layers.layer_norm与tf.keras.LayerNorm与nn.LayerNorm

##### `tf.contrib.layers.layer_norm`

tf以前遗留代码还是挺蛋疼的。在`tf.contrib.layers.layer_norm`中，对于输入为`(4, 10, 10, 3)`的张量，是对`(h,w,c)`进行归一化处理，但是他的仿射系数默认只对`c`有效：
```python
x = tf.reshape(tf.range(4 * 3 * 10 * 10, dtype=tf.float32), (4, 10, 10, 3))
xout = tf_contrib.layers.layer_norm(x,
                                    center=True, scale=True,
                                    scope='layer_norm')
mean.shape = (4, 1, 1, 1) 
gamma.shape = (3,)
```


##### `tf.keras.LayerNorm`

`tf.keras.LayerNorm`我就属实不懂了，讲道理他的归一化是对`(h,w,c)`进行归一化处理，仿射系数对`c`有效，但是输出归一化结果是`400=4×10x10`，这就很奇怪了，他默认的特征维度是`-1`，但是看起来却没有干`LayerNorm`应该做的事情，反而把`batch`维度也归一化了，**但是**在最终测试输出的时候发现结果是符合预期的。。属实不理解。

```python
inputs_np = tf.convert_to_tensor(
    np.arange(4 * 3 * 10 * 10).reshape((4, 10, 10, 3)), dtype=tf.float32)
inputs = k.Input((10, 10, 3), batch_size=None)
lm = k.layers.LayerNormalization()
lm.weights
lm_out = lm(inputs)
md = k.Model(inputs, lm_out) 
scale.shape # (3,)
mean.shape # (400,1)

lm_out_np = md(inputs_np)
lm_out_np = lm_out_np.numpy()
np.mean(lm_out_np[0, ...]) # -3.8146972e-08
np.var(lm_out_np[0, ...]) # 0.9985023
```



##### `nn.LayerNorm`

`nn.LayerNorm`是对`(c,h,w)`进行归一化处理，仿射系数对`c,h,w`有效，但有个非常蛋疼的问题就是，他没有办法复现老版本`tf`的行为，即只用`c`作为仿射系数，如果开启仿射会导致参数非常大。。。


```python
inputs = torch.tensor(np.arange(4 * 3 * 10 * 10).reshape((4, 3, 10, 10)), dtype=torch.float32)
lm = nn.LayerNorm([3, 10, 10], elementwise_affine=True)
ln_out = lm(inputs)
lm.weight.shape # torch.Size([3, 10, 10])
```

我继续检查他的源码,在`aten/src/ATen/native/layer_norm.h`中，将输入维度分为`M*N`，按照我们上面的做法即`M=4,N=3*10*10`。
然后进入cuda代码`aten/src/ATen/native/cuda/layer_norm_kernel.cu`利用`RowwiseMomentsCUDAKernel`计算均值与方差：
```cpp
template <typename T>
void LayerNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    T eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  DCHECK_EQ(X.numel(), M * N);
  DCHECK(!gamma.defined() || gamma.numel() == N);
  DCHECK(!beta.defined() || beta.numel() == N);
  const T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.data_ptr<T>() : nullptr;
  T* Y_data = Y->data_ptr<T>();
  T* mean_data = mean->data_ptr<T>();
  T* rstd_data = rstd->data_ptr<T>();
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  RowwiseMomentsCUDAKernel<T>
      <<<M, cuda_utils::kCUDABlockReduceNumThreads, 0, cuda_stream>>>(
          N, eps, X_data, mean_data, rstd_data);
  LayerNormForwardCUDAKernel<T><<<M, kCUDANumThreads, 0, cuda_stream>>>(
      N, X_data, mean_data, rstd_data, gamma_data, beta_data, Y_data);
  AT_CUDA_CHECK(cudaGetLastError());
}
```

接下来我们检查一下`group norm`，首先给定`group`，他将模型输入分为`N,C,HxW`。在`aten/src/ATen/native/cuda/group_norm_kernel.cu`中，当`group=1`的时候，`D=C/G=C`，`N×G=N`,也就是`group=1`的是等同于`layer norm`，并且此时他的可变化参数为`C`，可以用来等效`tf.contrib.layers.layer_norm`。

```cpp
template <typename T>
void GroupNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    T eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  using T_ACC = acc_type<T, true>;
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  TORCH_CHECK(!beta.defined() || beta.numel() == C);
  if (N == 0) {
    return;
  }
  const int64_t G = group;
  const int64_t D = C / G;
  const T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.data_ptr<T>() : nullptr;
  T* Y_data = Y->data_ptr<T>();
  T* mean_data = mean->data_ptr<T>();
  T* rstd_data = rstd->data_ptr<T>();
  const auto kAccType = X.scalar_type() == kHalf ? kFloat : X.scalar_type();
  Tensor a = at::empty({N, C}, X.options().dtype(kAccType));
  Tensor b = at::empty({N, C}, X.options().dtype(kAccType));
  T_ACC* a_data = a.data_ptr<T_ACC>();
  T_ACC* b_data = b.data_ptr<T_ACC>();
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  RowwiseMomentsCUDAKernel<T>
      <<<N * G, cuda_utils::kCUDABlockReduceNumThreads, 0, cuda_stream>>>(
          D * HxW, eps, X_data, mean_data, rstd_data);
  int64_t B = (N * C + kCUDANumThreads - 1) / kCUDANumThreads;
  ComputeFusedParamsCUDAKernel<T><<<B, kCUDANumThreads, 0, cuda_stream>>>(
      N, C, G, mean_data, rstd_data, gamma_data, beta_data, a_data, b_data);
  if (HxW < kCUDANumThreads) {
    B = (N * C * HxW + kCUDANumThreads - 1) / kCUDANumThreads;
    GroupNormForwardSimpleCUDAKernel<T><<<B, kCUDANumThreads, 0, cuda_stream>>>(
        N, C, HxW, X_data, a_data, b_data, Y_data);
  } else {
    GroupNormForwardCUDAKernel<T><<<N * C, kCUDANumThreads, 0, cuda_stream>>>(
        HxW, X_data, a_data, b_data, Y_data);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}
```