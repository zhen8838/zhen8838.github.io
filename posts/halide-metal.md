---
title: halide metal 初体验
mathjax: true
toc: true
categories:
  - 编译器
date: 2023-04-09 20:54:36
tags:
- Halide
- DSL
---

买了m2 mac pro之后, 一直想把m2的计算能力应用起来, 发现还是halide的功能比较完备, 支持metal后端, 所以尝试一下.

<!--more-->


# 0. Setup

我使用的是halide 14.0, 编译好之后配置好python bindings.

# 1. PyHalide CodeGen

简单写了一个代码, 验证metal后端的指令生成有效且正确.

⚠️ : halide本身没有导出`ParamMap`的接口, 我这里简单添加了一下, 后面有机会考虑做一下pr.

我对于metal gpu架构了解的比较少, 没太理解halide为什么对于metal后端调度也是和cuda gpu类似, 因为Apple的简单介绍里面并没有提到有block的层级.

```python
import halide as hl
import numpy as np
host = hl.get_host_target().with_feature(hl.TargetFeature.Metal).with_feature(hl.TargetFeature.Debug)
assert hl.host_supports_target_device(host)

brighter = hl.Func("brighter")
x, y = hl.Var("x"), hl.Var("y")

input = hl.ImageParam(hl.Float(32), 2, "input")
input_value = hl.Buffer(hl.Float(32), [16, 16])
input_value.fill(1.0)

# Define the hl.Func.
brighter[x, y] = input[x, y] * 3

# Schedule it.
xo, yo, xi, yi = hl.Var("xo"), hl.Var("yo"), hl.Var("xi"), hl.Var("yi")

brighter.gpu_tile(x, y, xo, yo, xi, yi, 8, 8)
brighter.print_loop_nest()
brighter.compile_jit(host)
brighter.compile_to_file("brighter", [hl.Argument(input)], "brighter", host)

# test the schedule
reference_output = hl.Buffer(hl.Float(32), [16, 16])
brighter.realize([reference_output], host, hl.ParamMap([hl.ParamMapping(input, input_value)]))
reference_output.copy_to_host()
for i in range(16):
  for j in range(16):
    assert reference_output[j, i] == 3.0
```

输出:
```sh
halide_metal_device_malloc (user_context: 0x0, buf: 0x16d8e3f48)
    allocating buffer(0, 0x0, 0x104268800, 1, uint8, {0, 8, 1}, {0, 8, 8}, {0, 3, 64})
Metal - Allocating: MTLCreateSystemDefaultDevice
Metal - Allocating: new_command_queue
    Time: 4.000000e-03 ms
halide_metal_copy_to_device dev = 0x124041160 metal_buffer = 0x124058e10 host = 0x104268800
Time for halide_metal_copy_to_device: 1.070000e-01 ms
halide_metal_device_free called on buf 0x16d8e3f48 device is 4899213664
    Time: 1.883300e-02 ms
produce brighter:
  gpu_block y.yo<Default_GPU>:
    gpu_block x.xo<Default_GPU>:
      gpu_thread y.yi in [0, 7]<Default_GPU>:
        gpu_thread x.xi in [0, 7]<Default_GPU>:
          brighter(...) = ...
Entering Pipeline brighter
Target: arm-64-osx-debug-jit-metal-user_context
 Input Buffer input: buffer(0, 0x0, 0x114008280, 1, float32, {0, 16, 1}, {0, 16, 16})
 Input (void *) __user_context: 0x16d8e2b40
 Output Buffer brighter: buffer(0, 0x0, 0x127077e80, 0, float32, {0, 16, 1}, {0, 16, 16})
Caching compiled kernel: 0x133e4dc00 id 2 context 0x1248a6e00
Time for halide_metal_initialize_kernels: 4.945000e-01 ms
halide_metal_device_malloc (user_context: 0x16d8e2b40, buf: 0x112b511a0)
    allocating buffer(0, 0x0, 0x127077e80, 0, float32, {0, 16, 1}, {0, 16, 16})
    Time: 2.945900e-02 ms
halide_metal_device_malloc (user_context: 0x16d8e2b40, buf: 0x113e307b0)
    allocating buffer(0, 0x0, 0x114008280, 1, float32, {0, 16, 1}, {0, 16, 16})
    Time: 6.083000e-03 ms
halide_metal_copy_to_device dev = 0x133e52d10 metal_buffer = 0x112b9f2c0 host = 0x114008280
Time for halide_metal_copy_to_device: 9.870800e-02 ms
Metal - supports setBytes
Total args size is 44 and with padding, size is 44
Setting shared memory length to 0
Dispatching threadgroups (number 0) blocks(2, 2, 1) threads(8, 8, 1)
Time for halide_metal_device_run: 6.777500e-01 ms
Exiting Pipeline brighter
Time for halide_metal_copy_to_host: 6.170410e-01 ms
halide_metal_device_free called on buf 0x113e307b0 device is 5165624592
    Time: 1.229200e-02 ms
halide_metal_device_free called on buf 0x112b511a0 device is 5165682976
    Time: 4.417000e-03 ms
```

# 2. Cpp Using Generated Kernel

接下来再写一个cpp代码调用一下生成好的kernel.

```cpp
#include "brighter.h"
#include <HalideBuffer.h>
#include <iostream>

int main(int argc, char **argv) {
    int dim = 16;
    size_t length = (size_t)dim * dim;
    float a[length];
    for (size_t i = 0; i < length; i++) {
        a[i] = 1;
        std::cout << a[i] << ", ";
    }
    std::cout << std::endl;
    float b[length];
    Halide::Runtime::Buffer<float, 2> input_buffer(a, dim, dim);
    input_buffer.set_host_dirty(); // for async buffer
    Halide::Runtime::Buffer<float, 2> output_buffer(b, dim, dim);
    brighter(input_buffer, output_buffer);
    output_buffer.copy_to_host(); // copy to host
    for (size_t i = 0; i < length; i++) {
        std::cout << b[i] << ", ";
    }
    std::cout << std::endl;
    return 0;
}
```

编译运行:
```sh
❯ clang++ -std=c++17 -stdlib=libc++ -Iout/build/debug/include/ -fno-objc-arc -framework Metal -framework Foundation -framework MetalKit brighter.o main.cpp && ./a.out
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
Entering Pipeline brighter
Target: arm-64-osx-debug-metal
 Input Buffer input: buffer(0, 0x0, 0x16b35a770, 1, float32, {0, 16, 1}, {0, 16, 16})
 Output Buffer brighter: buffer(0, 0x0, 0x16b35a370, 0, float32, {0, 16, 1}, {0, 16, 16})
Metal - Allocating: MTLCreateSystemDefaultDevice
Metal - Allocating: new_command_queue
Caching compiled kernel: 0x143611bd0 id 2 context 0x14480ca00
Time for halide_metal_initialize_kernels: 6.449170e-01 ms
halide_copy_to_device validating input buffer: buffer(0, 0x0, 0x16b35a370, 0, float32, {0, 16, 1}, {0, 16, 16})
halide_device_malloc validating input buffer: buffer(0, 0x0, 0x16b35a370, 0, float32, {0, 16, 1}, {0, 16, 16})
halide_device_malloc: target device interface 0x104acb260
halide_metal_device_malloc (user_context: 0x0, buf: 0x16b35abe8)
    allocating buffer(0, 0x0, 0x16b35a370, 0, float32, {0, 16, 1}, {0, 16, 16})
    Time: 8.459000e-03 ms
halide_copy_to_device 0x16b35abe8 skipped (host is not dirty)
halide_copy_to_device validating input buffer: buffer(0, 0x0, 0x16b35a770, 1, float32, {0, 16, 1}, {0, 16, 16})
halide_device_malloc validating input buffer: buffer(0, 0x0, 0x16b35a770, 1, float32, {0, 16, 1}, {0, 16, 16})
halide_device_malloc: target device interface 0x104acb260
halide_metal_device_malloc (user_context: 0x0, buf: 0x16b35ac60)
    allocating buffer(0, 0x0, 0x16b35a770, 1, float32, {0, 16, 1}, {0, 16, 16})
    Time: 3.375000e-03 ms
halide_copy_to_device 0x16b35ac60 host is dirty
halide_copy_to_device 0x16b35ac60 calling copy_to_device()
halide_metal_copy_to_device dev = 0x1436206f0 metal_buffer = 0x143621d00 host = 0x16b35a770
Time for halide_metal_copy_to_device: 1.747500e-01 ms
Metal - supports setBytes
Total args size is 44 and with padding, size is 44
Setting shared memory length to 0
Dispatching threadgroups (number 0) blocks(2, 2, 1) threads(8, 8, 1)
Time for halide_metal_device_run: 7.601670e-01 ms
Exiting Pipeline brighter
halide_copy_to_host validating input buffer: buffer(5425404128, 0x104acb260, 0x16b35a370, 2, float32, {0, 16, 1}, {0, 16, 16})
copy_to_host_already_locked 0x16b35abe8 dev_dirty is true
Time for halide_metal_copy_to_host: 4.510420e-01 ms
3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
halide_device_free validating input buffer: buffer(5425404128, 0x104acb260, 0x16b35a370, 0, float32, {0, 16, 1}, {0, 16, 16})
halide_metal_device_free called on buf 0x16b35abe8 device is 5425404128
    Time: 1.316700e-02 ms
halide_device_free validating input buffer: buffer(5425465072, 0x104acb260, 0x16b35a770, 0, float32, {0, 16, 1}, {0, 16, 16})
halide_metal_device_free called on buf 0x16b35ac60 device is 5425465072
    Time: 4.042000e-03 ms
```

# 3. PyHalide with numpy

如果把halide升级到15.x,那么可以无缝的python利用使用halide jit的kernel了. 这里说的无缝就是不需要对numpy的array做任何操作就可以直接传入halide kernel. 本来halide buffer是最内层为最低维度, 而numpy是最外层是最低维度, 现在在调用PyCallable的时候他已经可以做到自动转换了.

```python
import halide as hl
import numpy as np
host = hl.get_host_target().with_feature(hl.TargetFeature.Metal).with_feature(hl.TargetFeature.Debug)
assert hl.host_supports_target_device(host)


class MatMulPipeLine:
  def __init__(self) -> None:
    self.M, self.K, self.N = 4096, 4096, 2
    inputLhs = hl.ImageParam(hl.Float(32), 2, "inputLhs")
    inputRhs = hl.ImageParam(hl.Float(32), 2, "inputRhs")
    output = hl.Func("output")
    (m, n) = hl.Var("m"), hl.Var("n")
    k = hl.RDom([hl.Range(0, self.K)], "k")

    output[n, m] = 0.0
    output[n, m] += inputLhs[k.x, m] * inputRhs[n, k.x]

    self.matmul = output.compile_to_callable([inputLhs, inputRhs], target=host)

  def test_correctness(self):
    a = np.random.rand(self.M, self.K).astype(np.float32)
    b = np.random.rand(self.K, self.N).astype(np.float32)
    reference_output = np.matmul(a, b)
    halide_output = np.zeros_like(reference_output, dtype=np.float32)
    self.matmul(a, b, halide_output)
    np.allclose(reference_output, halide_output)


matmul = MatMulPipeLine()
matmul.test_correctness()
```


# 3. Metal 算子调度

对于metal的架构我并不太熟悉,比较熟悉的是npu架构, 我测试了一个简单的schedule之后, 发现性能比不调度还慢3倍. 后面发现是gpu这种架构没法在kernel里面load, 必须得在一开始的时候设定好buffer.

再然后我仔细调研了一下metal编程, 发现metal其实比较想象中的复杂, 主要是halide其实没有提供足够多个schedule api让每个循环映射到不同的硬件层次上, 所以想要高性能还是得自己来, 或者魔改halide. 对于cpu这种内存层次不多的硬件, halide所提供的描述能力倒是足够的. 
```cpp
kernel void inspector(
                  device const float* X                        [[buffer(0)]],
                  device float* result                         [[buffer(1)]],
                  device uint* store                           [[buffer(2)]],
                  uint thread_position_in_grid             ****    [[thread_position_in_grid]],
                  uint threads_per_grid                        [[threads_per_grid]],
                  uint dispatch_quadgroups_per_threadgroup     [[dispatch_quadgroups_per_threadgroup]],
                  uint dispatch_simdgroups_per_threadgroup     [[dispatch_simdgroups_per_threadgroup]], 
                  uint dispatch_threads_per_threadgroup        [[dispatch_threads_per_threadgroup]], 
                  uint grid_origin                             [[grid_origin]], 
                  uint grid_size                               [[grid_size]], 
                  uint quadgroup_index_in_threadgroup          [[quadgroup_index_in_threadgroup]], 
                  uint quadgroups_per_threadgroup              [[quadgroups_per_threadgroup]], 
                  uint simdgroup_index_in_threadgroup          [[simdgroup_index_in_threadgroup]], 
                  uint simdgroups_per_threadgroup              [[simdgroups_per_threadgroup]], 
                  uint thread_execution_width                  [[thread_execution_width]], 
                  uint thread_index_in_quadgroup               [[thread_index_in_quadgroup]], 
                  uint thread_index_in_simdgroup               [[thread_index_in_simdgroup]], 
                  uint thread_index_in_threadgroup             [[thread_index_in_threadgroup]], 
                  uint thread_position_in_threadgroup          [[thread_position_in_threadgroup]], 
                  uint threadgroup_position_in_grid            [[threadgroup_position_in_grid]],
                  uint threadgroups_per_grid                   [[threadgroups_per_grid]], 
                  uint threads_per_simdgroup                   [[threads_per_simdgroup]], 
                  uint threads_per_threadgroup                 [[threads_per_threadgroup]])
```