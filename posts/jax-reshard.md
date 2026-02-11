---
title: 探究jax reshard优化
toc: true
mathjax: true
categories:
  - 编译器
date: 2025-10-15 12:05:10
tags:
- 分布式
---

Google在分布式系统上有非常深厚的积累，本文主要尝试检查jax的行为来探究数据重分布`reshard`算子的优化方案。

<!--more-->


```python
import os
# os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '5'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
# os.environ['ABSL_VLOG_LEVEL'] = '4'
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, AxisType, get_abstract_mesh, reshard
```

配置当前环境下的device/mesh：


```python
jax.config.update('jax_num_cpu_devices', 8)
# jax.config.update('jax_logging_level', 'DEBUG')
mesh = jax.make_mesh((2, 4), ("X", "Y"), axis_types=(AxisType.Explicit, AxisType.Explicit))
jax.set_mesh(mesh)

print(f"Current mesh is: {get_abstract_mesh()}")
```

```cpp
    Current mesh is: AbstractMesh('X': 2, 'Y': 4, axis_types=(Explicit, Explicit), device_kind=cpu, num_cores=None)
```


# resharding `[M @ X,N] -> [M @ Y, N @ X]`

这是两个维度都需要发生变化的例子，数据在M上需要再次切分发送到Y上，然后在N上需要进行切分。


```python
@jax.jit
def reshard_1(x):
  y = reshard(x, P('Y', 'X'))
  return y
```


```python
x = jnp.ones((2048, 2048), jnp.float32, out_sharding=P('X', None))
compilation_args = {
    'xla_dump_to': 'tmp/reshard_1',
    'xla_dump_hlo_pass_re' : '.*'
}
traced_1 = reshard_1.trace(x)
lowered_1 = traced_1.lower()
compiled_1 = lowered_1.compile(compilation_args)
```

lower之后可以查看hlo的ir:


```python
print(lowered_1.as_text())
```

```cpp
    module @jit_reshard_1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
      sdy.mesh @mesh = <["X"=2, "Y"=4]>
      func.func public @main(%arg0: tensor<2048x2048xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"X"}, {}]>}) -> (tensor<2048x2048xf32> {jax.result_info = "result", sdy.sharding = #sdy.sharding<@mesh, [{"Y"}, {"X"}]>}) {
        %0 = sdy.sharding_constraint %arg0 <@mesh, [{"Y"}, {"X"}]> : tensor<2048x2048xf32>
        return %0 : tensor<2048x2048xf32>
      }
    }
```


compile之后可以参考spmd/mpmd的ir：


```python
print(compiled_1.as_text())
```

```cpp
    HloModule jit_reshard_1, is_scheduled=true, entry_computation_layout={(f32[1024,2048]{1,0})->f32[512,1024]{1,0}}, num_partitions=8
    
    %fused_computation (param_0: f32[1024,2048], param_1.1: s32[8], param_2.3: u32[], param_3.3: s32[8], param_4.2: s32[8]) -> f32[512,1024] {
      %param_0 = f32[1024,2048]{1,0} parameter(0)
      %param_4.2 = s32[8]{0} parameter(4)
      %param_2.3 = u32[] parameter(2)
      %dynamic-slice.8 = s32[1]{0} dynamic-slice(%param_4.2, %param_2.3), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_1)/reshard" }
      %param_3.3 = s32[8]{0} parameter(3)
      %dynamic-slice.7 = s32[1]{0} dynamic-slice(%param_3.3, %param_2.3), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_1)/reshard" }
      %subtract.3 = s32[1]{0} subtract(%dynamic-slice.8, %dynamic-slice.7), metadata={op_name="jit(reshard_1)/reshard" }
      %bitcast.3 = s32[] bitcast(%subtract.3), metadata={op_name="jit(reshard_1)/reshard" }
      %param_1.1 = s32[8]{0} parameter(1)
      %dynamic-slice.6 = s32[1]{0} dynamic-slice(%param_1.1, %param_2.3), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_1)/reshard" }
      %bitcast.2 = s32[] bitcast(%dynamic-slice.6), metadata={op_name="jit(reshard_1)/reshard" }
      ROOT %dynamic-slice.5 = f32[512,1024]{1,0} dynamic-slice(%param_0, %bitcast.3, %bitcast.2), dynamic_slice_sizes={512,1024}, metadata={op_name="jit(reshard_1)/reshard" }
    }
    
    ENTRY %main.0_spmd (param: f32[1024,2048]) -> f32[512,1024] {
      %partition-id = u32[] partition-id()
      %param = f32[1024,2048]{1,0} parameter(0), sharding={devices=[2,1,4]<=[8] last_tile_dim_replicate}, metadata={op_name="x"}
      %constant.3 = s32[8]{0} constant({0, 0, 512, 512, 1024, 1024, 1536, 1536}), metadata={op_name="jit(reshard_1)/reshard" }
      %constant.4 = s32[8]{0} constant({0, 1024, 0, 1024, 0, 1024, 0, 1024}), metadata={op_name="jit(reshard_1)/reshard" }
      %constant.5 = s32[8]{0} constant({0, 0, 0, 0, 1024, 1024, 1024, 1024}), metadata={op_name="jit(reshard_1)/reshard" }
      %bitcast_dynamic-slice_fusion = f32[512,1024]{1,0} fusion(%param, %constant.4, %partition-id, %constant.5, %constant.3), kind=kLoop, calls=%fused_computation, metadata={op_name="jit(reshard_1)/reshard" }, backend_config={"outer_dimension_partitions":["4"]}
      ROOT %collective-permute = f32[512,1024]{1,0} collective-permute(%bitcast_dynamic-slice_fusion), channel_id=1, source_target_pairs={{0,0},{1,4},{2,1},{3,5},{4,2},{5,6},{6,3},{7,7}}, metadata={op_name="jit(reshard_1)/reshard" }
    }
```
    



## IR解读

检查spmd的ir，先查看module信息：
```cpp
HloModule jit_reshard_1, is_scheduled=true, entry_computation_layout={(f32[1024,2048]{1,0})->f32[512,1024]{1,0}}, num_partitions=8
```

这里有一个`num_partitions`是定义程序所执行的分区数量，在hlo中还有一个`num_replicas`定义的是副本数， 我理解`num_replicas`是用于描述数据并行的，虽然在sharding的描述中看起来数据并行也是切分，但是他们属于不同的数据了，所以hlo这里会有一些特殊优化存在。


然后查看主函数参数为:
```cpp
%param = f32[1024,2048]{1,0} parameter(0), sharding={devices=[2,1,4]<=[8] last_tile_dim_replicate}, metadata={op_name="x"}
```
这表示了在spmd的ir下，并不会使用global view来表示分布式的数据。 同时这里`f32[1024, 2048]{1, 0}`中花括号部分表示的是数据的[layout](https://openxla.org/xla/hlo_to_thunks#layout_assignment_passes)， 就表明此时的param为列主序。



然后查看下面的三个constant和函数`bitcast_dynamic-slice_fusion`调用， 并将其简化:
```cpp
%partition-id = u32[] partition-id()
%slice_8 = dynamic-slice({0,   0 , 512, 512 , 1024, 1024, 1536, 1536}, %partition-id)
%slice_7 = dynamic-slice({0,   0 ,  0 ,  0  , 1024, 1024, 1024, 1024}, %partition-id)
%slice_6 = dynamic-slice({0, 1024,  0 , 1024,   0 , 1024,   0 , 1024}, %partition-id)
%sub_3 = subtract(%slice_8, %slice_7)
%slice_5 = f32[512,1024]{1,0} dynamic-slice(%param_0, %sub_3, %slice_6)
```
可以发现xla通过分析reshard，把需要进行数据传输的范围进行了提前计算，一共8个设备并行，然后每个设备上对应的slice 范围为：
```python
0 => dynamic_slice(input=f32[1024, 2048], start=(0 - 0  =       0 ,      0 ), size=(512, 1024))
1 => dynamic_slice(input=f32[1024, 2048], start=(0 - 0  =       0 ,    1024), size=(512, 1024))
2 => dynamic_slice(input=f32[1024, 2048], start=(512 - 0 =     512,      0 ), size=(512, 1024))
3 => dynamic_slice(input=f32[1024, 2048], start=(512 - 0 =     512,    1024), size=(512, 1024))
4 => dynamic_slice(input=f32[1024, 2048], start=(1024 - 1024 =  0 ,      0 ), size=(512, 1024))
5 => dynamic_slice(input=f32[1024, 2048], start=(1024 - 1024 =  0 ,    1024), size=(512, 1024))
6 => dynamic_slice(input=f32[1024, 2048], start=(1536 - 1024 = 512,      0 ), size=(512, 1024))
7 => dynamic_slice(input=f32[1024, 2048], start=(1536 - 1024 = 512,    1024), size=(512, 1024))
```

因为mesh的维度 `Y = 2X`, 所以上述操作在每个设备上对数据在M，N维度在此切分即可得到分布为`[M @ Y,N @ X]`的数据`f32[512,1024]`, 但数据的位置还是错乱的，因此还需要调用一次高性能的primitive: `collective-permute`，数据设备上交错之后就可以得到最终所需要的分布式布局：

```cpp
ROOT %collective-permute = f32[512,1024]{1,0} collective-permute(%bitcast_dynamic-slice_fusion), channel_id=1, source_target_pairs={{0,0},{1,4},{2,1},{3,5},{4,2},{5,6},{6,3},{7,7}}, metadata={op_name="jit(reshard_1)/reshard"}
```


# resharding `[M @ Y,N] -> [M @ X, N @ Y]`

上一个例子中由于X小于Y导致他可以使用本地tile进行计算，这次交换切分的位置进行测试。




```python
@jax.jit
def reshard_2(x):
  y = reshard(x, P('X', 'Y'))
  return y
```


```python
x = jnp.ones((2048, 2048), jnp.float32, out_sharding=P('Y', None))
compilation_args = {
    'xla_dump_to': 'tmp/reshard_2',
    'xla_dump_hlo_pass_re' : '.*'
}
traced_2 = reshard_2.trace(x)
lowered_2 = traced_2.lower()
compiled_2 = lowered_2.compile(compilation_args)
```


```python
print(compiled_2.as_text())
```

```cpp
    HloModule jit_reshard_2, is_scheduled=true, entry_computation_layout={(f32[512,2048]{1,0})->f32[1024,512]{1,0}}, num_partitions=8
    
    %fused_computation (param_0.3: f32[512,1,512], param_1: f32[512,1,512]) -> f32[1024,512] {
      %param_0.3 = f32[512,1,512]{2,1,0} parameter(0)
      %param_1 = f32[512,1,512]{2,1,0} parameter(1)
      %concatenate.1 = f32[512,2,512]{2,1,0} concatenate(%param_0.3, %param_1), dimensions={1}, metadata={op_name="jit(reshard_2)/reshard" }
      %transpose.1 = f32[2,512,512]{2,0,1} transpose(%concatenate.1), dimensions={1,0,2}, metadata={op_name="jit(reshard_2)/reshard" }
      %copy.1 = f32[2,512,512]{2,1,0} copy(%transpose.1), metadata={op_name="jit(reshard_2)/reshard" }
      ROOT %bitcast.3 = f32[1024,512]{1,0} bitcast(%copy.1), metadata={op_name="jit(reshard_2)/reshard" }
    }
    
    %fused_computation.1 (param_0.6: f32[512,2048], param_1.2: s32[8], param_2.3: u32[]) -> f32[512,1,512] {
      %param_0.6 = f32[512,2048]{1,0} parameter(0)
      %constant.18 = s32[] constant(0), metadata={op_name="jit(reshard_2)/reshard" }
      %param_1.2 = s32[8]{0} parameter(1)
      %param_2.3 = u32[] parameter(2)
      %dynamic-slice.10 = s32[1]{0} dynamic-slice(%param_1.2, %param_2.3), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_2)/reshard" }
      %bitcast.5 = s32[] bitcast(%dynamic-slice.10), metadata={op_name="jit(reshard_2)/reshard" }
      %dynamic-slice.9 = f32[512,1024]{1,0} dynamic-slice(%param_0.6, %constant.18, %bitcast.5), dynamic_slice_sizes={512,1024}, metadata={op_name="jit(reshard_2)/reshard" }
      %bitcast.4 = f32[512,2,512]{2,1,0} bitcast(%dynamic-slice.9), metadata={op_name="jit(reshard_2)/reshard" }
      ROOT %slice.2 = f32[512,1,512]{2,1,0} slice(%bitcast.4), slice={[0:512], [1:2], [0:512]}, metadata={op_name="jit(reshard_2)/reshard" }
    }
    
    %fused_computation.2 (param_0.9: f32[512,2048], param_1.4: s32[8], param_2.7: u32[]) -> f32[512,1,512] {
      %param_0.9 = f32[512,2048]{1,0} parameter(0)
      %constant.19 = s32[] constant(0), metadata={op_name="jit(reshard_2)/reshard" }
      %param_1.4 = s32[8]{0} parameter(1)
      %param_2.7 = u32[] parameter(2)
      %dynamic-slice.12 = s32[1]{0} dynamic-slice(%param_1.4, %param_2.7), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_2)/reshard" }
      %bitcast.7 = s32[] bitcast(%dynamic-slice.12), metadata={op_name="jit(reshard_2)/reshard" }
      %dynamic-slice.11 = f32[512,1024]{1,0} dynamic-slice(%param_0.9, %constant.19, %bitcast.7), dynamic_slice_sizes={512,1024}, metadata={op_name="jit(reshard_2)/reshard" }
      %bitcast.6 = f32[512,2,512]{2,1,0} bitcast(%dynamic-slice.11), metadata={op_name="jit(reshard_2)/reshard" }
      ROOT %slice.3 = f32[512,1,512]{2,1,0} slice(%bitcast.6), slice={[0:512], [0:1], [0:512]}, metadata={op_name="jit(reshard_2)/reshard" }
    }
    
    ENTRY %main.0_spmd (param: f32[512,2048]) -> f32[1024,512] {
      %partition-id = u32[] partition-id()
      %param = f32[512,2048]{1,0} parameter(0), sharding={devices=[4,1,2]<=[2,4]T(1,0) last_tile_dim_replicate}, metadata={op_name="x"}
      %constant.4 = s32[8]{0} constant({0, 0, 0, 0, 1024, 1024, 1024, 1024}), metadata={op_name="jit(reshard_2)/reshard" }
      %bitcast_slice_fusion.1 = f32[512,1,512]{2,1,0} fusion(%param, %constant.4, %partition-id), kind=kLoop, calls=%fused_computation.2, metadata={op_name="jit(reshard_2)/reshard" }, backend_config={"outer_dimension_partitions":["4"]}
      %bitcast_slice_fusion = f32[512,1,512]{2,1,0} fusion(%param, %constant.4, %partition-id), kind=kLoop, calls=%fused_computation.1, metadata={op_name="jit(reshard_2)/reshard" }, backend_config={"outer_dimension_partitions":["4"]}
      %all-to-all.1 = (f32[512,1,512]{2,1,0}, f32[512,1,512]{2,1,0}) all-to-all(%bitcast_slice_fusion.1, %bitcast_slice_fusion), channel_id=1, replica_groups=[4,2]<=[2,2,2]T(1,0,2), metadata={op_name="jit(reshard_2)/reshard" }
      %get-tuple-element.2 = f32[512,1,512]{2,1,0} get-tuple-element(%all-to-all.1), index=0
      %get-tuple-element.3 = f32[512,1,512]{2,1,0} get-tuple-element(%all-to-all.1), index=1
      %copy_bitcast_fusion = f32[1024,512]{1,0} fusion(%get-tuple-element.2, %get-tuple-element.3), kind=kLoop, calls=%fused_computation, metadata={op_name="jit(reshard_2)/reshard" }, backend_config={"outer_dimension_partitions":["4"]}
      ROOT %collective-permute = f32[1024,512]{1,0} collective-permute(%copy_bitcast_fusion), channel_id=2, source_target_pairs={{0,0},{1,1},{4,2},{5,3},{2,4},{3,5},{6,6},{7,7}}, metadata={op_name="jit(reshard_2)/reshard" }
    }
```
    


这次就可以发现xla对这种情况拆分为了两次的ccl操作，首先切分出小块后进行`all-to-all`，合并后再执行`collective-permute`得到最终的分布。

# resharding `[M, N @ (X,T)] -> [M @ (D,Y,X,T), N]`

测试一个比较复杂的例子


```python
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, AxisType, get_abstract_mesh, reshard

jax.config.update('jax_num_cpu_devices', 1 * 2 * 8 * 4 * 4)
mesh = jax.make_mesh((1, 2, 8, 4, 4), ('C', 'D', 'Y', 'X', 'T'),
                     axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit))
jax.set_mesh(mesh)

print(f"Current mesh is: {get_abstract_mesh()}")
```

```cpp
    Current mesh is: AbstractMesh('C': 1, 'D': 2, 'Y': 8, 'X': 4, 'T': 4, axis_types=(Explicit, Explicit, Explicit, Explicit, Explicit), device_kind=cpu, num_cores=None)
```



```python
@jax.jit
def reshard_3(x):
  y = reshard(x, P(('D', 'Y', 'X', 'T'), None))
  return y

x = jnp.ones((2048, 2048), jnp.float32, out_sharding=P(None, ('X', 'T')))
compilation_args = {
    'xla_dump_to': 'tmp/reshard_3',
    'xla_dump_hlo_pass_re' : '.*'
}

traced_3 = reshard_3.trace(x)
lowered_3 = traced_3.lower()
compiled_3 = lowered_3.compile(compilation_args)
print(compiled_3.as_text())
```


```cpp

    HloModule jit_reshard_3, is_scheduled=true, entry_computation_layout={(f32[2048,128]{1,0})->f32[8,2048]{1,0}}, num_partitions=256
    
    %fused_computation (param_0.2: f32[1,16,8,128]) -> f32[8,2048] {
      %param_0.2 = f32[1,16,8,128]{3,2,1,0} parameter(0)
      %transpose.1 = f32[1,8,16,128]{3,1,2,0} transpose(%param_0.2), dimensions={0,2,1,3}, metadata={op_name="jit(reshard_3)/reshard" }
      %copy.1 = f32[1,8,16,128]{3,2,1,0} copy(%transpose.1), metadata={op_name="jit(reshard_3)/reshard" }
      ROOT %bitcast.3 = f32[8,2048]{1,0} bitcast(%copy.1), metadata={op_name="jit(reshard_3)/reshard" }
    }
    
    %fused_computation.1 (param_0.5: f32[2048,128], param_1.2: s32[256], param_2.1: u32[]) -> f32[1,1,8,128] {
      %param_0.5 = f32[2048,128]{1,0} parameter(0)
      %param_1.2 = s32[256]{0} parameter(1)
      %param_2.1 = u32[] parameter(2)
      %dynamic-slice.10 = s32[1]{0} dynamic-slice(%param_1.2, %param_2.1), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.5 = s32[] bitcast(%dynamic-slice.10), metadata={op_name="jit(reshard_3)/reshard" }
      %constant.22 = s32[] constant(0), metadata={op_name="jit(reshard_3)/reshard" }
      %dynamic-slice.9 = f32[128,128]{1,0} dynamic-slice(%param_0.5, %bitcast.5, %constant.22), dynamic_slice_sizes={128,128}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.4 = f32[1,16,8,128]{3,2,1,0} bitcast(%dynamic-slice.9), metadata={op_name="jit(reshard_3)/reshard" }
      ROOT %slice.16 = f32[1,1,8,128]{3,2,1,0} slice(%bitcast.4), slice={[0:1], [15:16], [0:8], [0:128]}, metadata={op_name="jit(reshard_3)/reshard" }
    }
    
    %fused_computation.2 (param_0.8: f32[2048,128], param_1.5: s32[256], param_2.3: u32[]) -> f32[1,1,8,128] {
      %param_0.8 = f32[2048,128]{1,0} parameter(0)
      %param_1.5 = s32[256]{0} parameter(1)
      %param_2.3 = u32[] parameter(2)
      %dynamic-slice.12 = s32[1]{0} dynamic-slice(%param_1.5, %param_2.3), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.7 = s32[] bitcast(%dynamic-slice.12), metadata={op_name="jit(reshard_3)/reshard" }
      %constant.23 = s32[] constant(0), metadata={op_name="jit(reshard_3)/reshard" }
      %dynamic-slice.11 = f32[128,128]{1,0} dynamic-slice(%param_0.8, %bitcast.7, %constant.23), dynamic_slice_sizes={128,128}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.6 = f32[1,16,8,128]{3,2,1,0} bitcast(%dynamic-slice.11), metadata={op_name="jit(reshard_3)/reshard" }
      ROOT %slice.17 = f32[1,1,8,128]{3,2,1,0} slice(%bitcast.6), slice={[0:1], [14:15], [0:8], [0:128]}, metadata={op_name="jit(reshard_3)/reshard" }
    }
    
    %fused_computation.3 (param_0.11: f32[2048,128], param_1.8: s32[256], param_2.5: u32[]) -> f32[1,1,8,128] {
      %param_0.11 = f32[2048,128]{1,0} parameter(0)
      %param_1.8 = s32[256]{0} parameter(1)
      %param_2.5 = u32[] parameter(2)
      %dynamic-slice.14 = s32[1]{0} dynamic-slice(%param_1.8, %param_2.5), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.9 = s32[] bitcast(%dynamic-slice.14), metadata={op_name="jit(reshard_3)/reshard" }
      %constant.24 = s32[] constant(0), metadata={op_name="jit(reshard_3)/reshard" }
      %dynamic-slice.13 = f32[128,128]{1,0} dynamic-slice(%param_0.11, %bitcast.9, %constant.24), dynamic_slice_sizes={128,128}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.8 = f32[1,16,8,128]{3,2,1,0} bitcast(%dynamic-slice.13), metadata={op_name="jit(reshard_3)/reshard" }
      ROOT %slice.18 = f32[1,1,8,128]{3,2,1,0} slice(%bitcast.8), slice={[0:1], [13:14], [0:8], [0:128]}, metadata={op_name="jit(reshard_3)/reshard" }
    }
    
    %fused_computation.4 (param_0.14: f32[2048,128], param_1.11: s32[256], param_2.7: u32[]) -> f32[1,1,8,128] {
      %param_0.14 = f32[2048,128]{1,0} parameter(0)
      %param_1.11 = s32[256]{0} parameter(1)
      %param_2.7 = u32[] parameter(2)
      %dynamic-slice.16 = s32[1]{0} dynamic-slice(%param_1.11, %param_2.7), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.11 = s32[] bitcast(%dynamic-slice.16), metadata={op_name="jit(reshard_3)/reshard" }
      %constant.25 = s32[] constant(0), metadata={op_name="jit(reshard_3)/reshard" }
      %dynamic-slice.15 = f32[128,128]{1,0} dynamic-slice(%param_0.14, %bitcast.11, %constant.25), dynamic_slice_sizes={128,128}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.10 = f32[1,16,8,128]{3,2,1,0} bitcast(%dynamic-slice.15), metadata={op_name="jit(reshard_3)/reshard" }
      ROOT %slice.19 = f32[1,1,8,128]{3,2,1,0} slice(%bitcast.10), slice={[0:1], [12:13], [0:8], [0:128]}, metadata={op_name="jit(reshard_3)/reshard" }
    }
    
    %fused_computation.5 (param_0.17: f32[2048,128], param_1.14: s32[256], param_2.9: u32[]) -> f32[1,1,8,128] {
      %param_0.17 = f32[2048,128]{1,0} parameter(0)
      %param_1.14 = s32[256]{0} parameter(1)
      %param_2.9 = u32[] parameter(2)
      %dynamic-slice.18 = s32[1]{0} dynamic-slice(%param_1.14, %param_2.9), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.13 = s32[] bitcast(%dynamic-slice.18), metadata={op_name="jit(reshard_3)/reshard" }
      %constant.26 = s32[] constant(0), metadata={op_name="jit(reshard_3)/reshard" }
      %dynamic-slice.17 = f32[128,128]{1,0} dynamic-slice(%param_0.17, %bitcast.13, %constant.26), dynamic_slice_sizes={128,128}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.12 = f32[1,16,8,128]{3,2,1,0} bitcast(%dynamic-slice.17), metadata={op_name="jit(reshard_3)/reshard" }
      ROOT %slice.20 = f32[1,1,8,128]{3,2,1,0} slice(%bitcast.12), slice={[0:1], [11:12], [0:8], [0:128]}, metadata={op_name="jit(reshard_3)/reshard" }
    }
    
    %fused_computation.6 (param_0.20: f32[2048,128], param_1.17: s32[256], param_2.11: u32[]) -> f32[1,1,8,128] {
      %param_0.20 = f32[2048,128]{1,0} parameter(0)
      %param_1.17 = s32[256]{0} parameter(1)
      %param_2.11 = u32[] parameter(2)
      %dynamic-slice.20 = s32[1]{0} dynamic-slice(%param_1.17, %param_2.11), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.15 = s32[] bitcast(%dynamic-slice.20), metadata={op_name="jit(reshard_3)/reshard" }
      %constant.27 = s32[] constant(0), metadata={op_name="jit(reshard_3)/reshard" }
      %dynamic-slice.19 = f32[128,128]{1,0} dynamic-slice(%param_0.20, %bitcast.15, %constant.27), dynamic_slice_sizes={128,128}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.14 = f32[1,16,8,128]{3,2,1,0} bitcast(%dynamic-slice.19), metadata={op_name="jit(reshard_3)/reshard" }
      ROOT %slice.21 = f32[1,1,8,128]{3,2,1,0} slice(%bitcast.14), slice={[0:1], [10:11], [0:8], [0:128]}, metadata={op_name="jit(reshard_3)/reshard" }
    }
    
    %fused_computation.7 (param_0.23: f32[2048,128], param_1.20: s32[256], param_2.13: u32[]) -> f32[1,1,8,128] {
      %param_0.23 = f32[2048,128]{1,0} parameter(0)
      %param_1.20 = s32[256]{0} parameter(1)
      %param_2.13 = u32[] parameter(2)
      %dynamic-slice.22 = s32[1]{0} dynamic-slice(%param_1.20, %param_2.13), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.17 = s32[] bitcast(%dynamic-slice.22), metadata={op_name="jit(reshard_3)/reshard" }
      %constant.28 = s32[] constant(0), metadata={op_name="jit(reshard_3)/reshard" }
      %dynamic-slice.21 = f32[128,128]{1,0} dynamic-slice(%param_0.23, %bitcast.17, %constant.28), dynamic_slice_sizes={128,128}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.16 = f32[1,16,8,128]{3,2,1,0} bitcast(%dynamic-slice.21), metadata={op_name="jit(reshard_3)/reshard" }
      ROOT %slice.22 = f32[1,1,8,128]{3,2,1,0} slice(%bitcast.16), slice={[0:1], [9:10], [0:8], [0:128]}, metadata={op_name="jit(reshard_3)/reshard" }
    }
    
    %fused_computation.8 (param_0.26: f32[2048,128], param_1.23: s32[256], param_2.15: u32[]) -> f32[1,1,8,128] {
      %param_0.26 = f32[2048,128]{1,0} parameter(0)
      %param_1.23 = s32[256]{0} parameter(1)
      %param_2.15 = u32[] parameter(2)
      %dynamic-slice.24 = s32[1]{0} dynamic-slice(%param_1.23, %param_2.15), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.19 = s32[] bitcast(%dynamic-slice.24), metadata={op_name="jit(reshard_3)/reshard" }
      %constant.29 = s32[] constant(0), metadata={op_name="jit(reshard_3)/reshard" }
      %dynamic-slice.23 = f32[128,128]{1,0} dynamic-slice(%param_0.26, %bitcast.19, %constant.29), dynamic_slice_sizes={128,128}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.18 = f32[1,16,8,128]{3,2,1,0} bitcast(%dynamic-slice.23), metadata={op_name="jit(reshard_3)/reshard" }
      ROOT %slice.23 = f32[1,1,8,128]{3,2,1,0} slice(%bitcast.18), slice={[0:1], [8:9], [0:8], [0:128]}, metadata={op_name="jit(reshard_3)/reshard" }
    }
    
    %fused_computation.9 (param_0.29: f32[2048,128], param_1.26: s32[256], param_2.17: u32[]) -> f32[1,1,8,128] {
      %param_0.29 = f32[2048,128]{1,0} parameter(0)
      %param_1.26 = s32[256]{0} parameter(1)
      %param_2.17 = u32[] parameter(2)
      %dynamic-slice.26 = s32[1]{0} dynamic-slice(%param_1.26, %param_2.17), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.21 = s32[] bitcast(%dynamic-slice.26), metadata={op_name="jit(reshard_3)/reshard" }
      %constant.30 = s32[] constant(0), metadata={op_name="jit(reshard_3)/reshard" }
      %dynamic-slice.25 = f32[128,128]{1,0} dynamic-slice(%param_0.29, %bitcast.21, %constant.30), dynamic_slice_sizes={128,128}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.20 = f32[1,16,8,128]{3,2,1,0} bitcast(%dynamic-slice.25), metadata={op_name="jit(reshard_3)/reshard" }
      ROOT %slice.24 = f32[1,1,8,128]{3,2,1,0} slice(%bitcast.20), slice={[0:1], [7:8], [0:8], [0:128]}, metadata={op_name="jit(reshard_3)/reshard" }
    }
    
    %fused_computation.10 (param_0.32: f32[2048,128], param_1.29: s32[256], param_2.19: u32[]) -> f32[1,1,8,128] {
      %param_0.32 = f32[2048,128]{1,0} parameter(0)
      %param_1.29 = s32[256]{0} parameter(1)
      %param_2.19 = u32[] parameter(2)
      %dynamic-slice.28 = s32[1]{0} dynamic-slice(%param_1.29, %param_2.19), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.23 = s32[] bitcast(%dynamic-slice.28), metadata={op_name="jit(reshard_3)/reshard" }
      %constant.31 = s32[] constant(0), metadata={op_name="jit(reshard_3)/reshard" }
      %dynamic-slice.27 = f32[128,128]{1,0} dynamic-slice(%param_0.32, %bitcast.23, %constant.31), dynamic_slice_sizes={128,128}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.22 = f32[1,16,8,128]{3,2,1,0} bitcast(%dynamic-slice.27), metadata={op_name="jit(reshard_3)/reshard" }
      ROOT %slice.25 = f32[1,1,8,128]{3,2,1,0} slice(%bitcast.22), slice={[0:1], [6:7], [0:8], [0:128]}, metadata={op_name="jit(reshard_3)/reshard" }
    }
    
    %fused_computation.11 (param_0.35: f32[2048,128], param_1.32: s32[256], param_2.21: u32[]) -> f32[1,1,8,128] {
      %param_0.35 = f32[2048,128]{1,0} parameter(0)
      %param_1.32 = s32[256]{0} parameter(1)
      %param_2.21 = u32[] parameter(2)
      %dynamic-slice.30 = s32[1]{0} dynamic-slice(%param_1.32, %param_2.21), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.25 = s32[] bitcast(%dynamic-slice.30), metadata={op_name="jit(reshard_3)/reshard" }
      %constant.32 = s32[] constant(0), metadata={op_name="jit(reshard_3)/reshard" }
      %dynamic-slice.29 = f32[128,128]{1,0} dynamic-slice(%param_0.35, %bitcast.25, %constant.32), dynamic_slice_sizes={128,128}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.24 = f32[1,16,8,128]{3,2,1,0} bitcast(%dynamic-slice.29), metadata={op_name="jit(reshard_3)/reshard" }
      ROOT %slice.26 = f32[1,1,8,128]{3,2,1,0} slice(%bitcast.24), slice={[0:1], [5:6], [0:8], [0:128]}, metadata={op_name="jit(reshard_3)/reshard" }
    }
    
    %fused_computation.12 (param_0.38: f32[2048,128], param_1.35: s32[256], param_2.23: u32[]) -> f32[1,1,8,128] {
      %param_0.38 = f32[2048,128]{1,0} parameter(0)
      %param_1.35 = s32[256]{0} parameter(1)
      %param_2.23 = u32[] parameter(2)
      %dynamic-slice.32 = s32[1]{0} dynamic-slice(%param_1.35, %param_2.23), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.27 = s32[] bitcast(%dynamic-slice.32), metadata={op_name="jit(reshard_3)/reshard" }
      %constant.33 = s32[] constant(0), metadata={op_name="jit(reshard_3)/reshard" }
      %dynamic-slice.31 = f32[128,128]{1,0} dynamic-slice(%param_0.38, %bitcast.27, %constant.33), dynamic_slice_sizes={128,128}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.26 = f32[1,16,8,128]{3,2,1,0} bitcast(%dynamic-slice.31), metadata={op_name="jit(reshard_3)/reshard" }
      ROOT %slice.27 = f32[1,1,8,128]{3,2,1,0} slice(%bitcast.26), slice={[0:1], [4:5], [0:8], [0:128]}, metadata={op_name="jit(reshard_3)/reshard" }
    }
    
    %fused_computation.13 (param_0.41: f32[2048,128], param_1.38: s32[256], param_2.25: u32[]) -> f32[1,1,8,128] {
      %param_0.41 = f32[2048,128]{1,0} parameter(0)
      %param_1.38 = s32[256]{0} parameter(1)
      %param_2.25 = u32[] parameter(2)
      %dynamic-slice.34 = s32[1]{0} dynamic-slice(%param_1.38, %param_2.25), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.29 = s32[] bitcast(%dynamic-slice.34), metadata={op_name="jit(reshard_3)/reshard" }
      %constant.34 = s32[] constant(0), metadata={op_name="jit(reshard_3)/reshard" }
      %dynamic-slice.33 = f32[128,128]{1,0} dynamic-slice(%param_0.41, %bitcast.29, %constant.34), dynamic_slice_sizes={128,128}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.28 = f32[1,16,8,128]{3,2,1,0} bitcast(%dynamic-slice.33), metadata={op_name="jit(reshard_3)/reshard" }
      ROOT %slice.28 = f32[1,1,8,128]{3,2,1,0} slice(%bitcast.28), slice={[0:1], [3:4], [0:8], [0:128]}, metadata={op_name="jit(reshard_3)/reshard" }
    }
    
    %fused_computation.14 (param_0.44: f32[2048,128], param_1.41: s32[256], param_2.27: u32[]) -> f32[1,1,8,128] {
      %param_0.44 = f32[2048,128]{1,0} parameter(0)
      %param_1.41 = s32[256]{0} parameter(1)
      %param_2.27 = u32[] parameter(2)
      %dynamic-slice.36 = s32[1]{0} dynamic-slice(%param_1.41, %param_2.27), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.31 = s32[] bitcast(%dynamic-slice.36), metadata={op_name="jit(reshard_3)/reshard" }
      %constant.35 = s32[] constant(0), metadata={op_name="jit(reshard_3)/reshard" }
      %dynamic-slice.35 = f32[128,128]{1,0} dynamic-slice(%param_0.44, %bitcast.31, %constant.35), dynamic_slice_sizes={128,128}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.30 = f32[1,16,8,128]{3,2,1,0} bitcast(%dynamic-slice.35), metadata={op_name="jit(reshard_3)/reshard" }
      ROOT %slice.29 = f32[1,1,8,128]{3,2,1,0} slice(%bitcast.30), slice={[0:1], [2:3], [0:8], [0:128]}, metadata={op_name="jit(reshard_3)/reshard" }
    }
    
    %fused_computation.15 (param_0.47: f32[2048,128], param_1.44: s32[256], param_2.29: u32[]) -> f32[1,1,8,128] {
      %param_0.47 = f32[2048,128]{1,0} parameter(0)
      %param_1.44 = s32[256]{0} parameter(1)
      %param_2.29 = u32[] parameter(2)
      %dynamic-slice.38 = s32[1]{0} dynamic-slice(%param_1.44, %param_2.29), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.33 = s32[] bitcast(%dynamic-slice.38), metadata={op_name="jit(reshard_3)/reshard" }
      %constant.36 = s32[] constant(0), metadata={op_name="jit(reshard_3)/reshard" }
      %dynamic-slice.37 = f32[128,128]{1,0} dynamic-slice(%param_0.47, %bitcast.33, %constant.36), dynamic_slice_sizes={128,128}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.32 = f32[1,16,8,128]{3,2,1,0} bitcast(%dynamic-slice.37), metadata={op_name="jit(reshard_3)/reshard" }
      ROOT %slice.30 = f32[1,1,8,128]{3,2,1,0} slice(%bitcast.32), slice={[0:1], [1:2], [0:8], [0:128]}, metadata={op_name="jit(reshard_3)/reshard" }
    }
    
    %fused_computation.16 (param_0.50: f32[2048,128], param_1.47: s32[256], param_2.31: u32[]) -> f32[1,1,8,128] {
      %param_0.50 = f32[2048,128]{1,0} parameter(0)
      %param_1.47 = s32[256]{0} parameter(1)
      %param_2.31 = u32[] parameter(2)
      %dynamic-slice.40 = s32[1]{0} dynamic-slice(%param_1.47, %param_2.31), dynamic_slice_sizes={1}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.35 = s32[] bitcast(%dynamic-slice.40), metadata={op_name="jit(reshard_3)/reshard" }
      %constant.37 = s32[] constant(0), metadata={op_name="jit(reshard_3)/reshard" }
      %dynamic-slice.39 = f32[128,128]{1,0} dynamic-slice(%param_0.50, %bitcast.35, %constant.37), dynamic_slice_sizes={128,128}, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast.34 = f32[1,16,8,128]{3,2,1,0} bitcast(%dynamic-slice.39), metadata={op_name="jit(reshard_3)/reshard" }
      ROOT %slice.31 = f32[1,1,8,128]{3,2,1,0} slice(%bitcast.34), slice={[0:1], [0:1], [0:8], [0:128]}, metadata={op_name="jit(reshard_3)/reshard" }
    }
    
    ENTRY %main.0_spmd (param: f32[2048,128]) -> f32[8,2048] {
      %partition-id = u32[] partition-id()
      %param = f32[2048,128]{1,0} parameter(0), sharding={devices=[1,16,16]<=[16,16]T(1,0) last_tile_dim_replicate}, metadata={op_name="x"}
      %constant.3 = s32[256]{0} constant({...}), metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast_slice_fusion.15 = f32[1,1,8,128]{3,2,1,0} fusion(%param, %constant.3, %partition-id), kind=kLoop, calls=%fused_computation.16, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast_slice_fusion = f32[1,1,8,128]{3,2,1,0} fusion(%param, %constant.3, %partition-id), kind=kLoop, calls=%fused_computation.1, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast_slice_fusion.1 = f32[1,1,8,128]{3,2,1,0} fusion(%param, %constant.3, %partition-id), kind=kLoop, calls=%fused_computation.2, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast_slice_fusion.2 = f32[1,1,8,128]{3,2,1,0} fusion(%param, %constant.3, %partition-id), kind=kLoop, calls=%fused_computation.3, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast_slice_fusion.3 = f32[1,1,8,128]{3,2,1,0} fusion(%param, %constant.3, %partition-id), kind=kLoop, calls=%fused_computation.4, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast_slice_fusion.4 = f32[1,1,8,128]{3,2,1,0} fusion(%param, %constant.3, %partition-id), kind=kLoop, calls=%fused_computation.5, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast_slice_fusion.5 = f32[1,1,8,128]{3,2,1,0} fusion(%param, %constant.3, %partition-id), kind=kLoop, calls=%fused_computation.6, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast_slice_fusion.6 = f32[1,1,8,128]{3,2,1,0} fusion(%param, %constant.3, %partition-id), kind=kLoop, calls=%fused_computation.7, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast_slice_fusion.7 = f32[1,1,8,128]{3,2,1,0} fusion(%param, %constant.3, %partition-id), kind=kLoop, calls=%fused_computation.8, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast_slice_fusion.8 = f32[1,1,8,128]{3,2,1,0} fusion(%param, %constant.3, %partition-id), kind=kLoop, calls=%fused_computation.9, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast_slice_fusion.9 = f32[1,1,8,128]{3,2,1,0} fusion(%param, %constant.3, %partition-id), kind=kLoop, calls=%fused_computation.10, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast_slice_fusion.10 = f32[1,1,8,128]{3,2,1,0} fusion(%param, %constant.3, %partition-id), kind=kLoop, calls=%fused_computation.11, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast_slice_fusion.11 = f32[1,1,8,128]{3,2,1,0} fusion(%param, %constant.3, %partition-id), kind=kLoop, calls=%fused_computation.12, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast_slice_fusion.12 = f32[1,1,8,128]{3,2,1,0} fusion(%param, %constant.3, %partition-id), kind=kLoop, calls=%fused_computation.13, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast_slice_fusion.13 = f32[1,1,8,128]{3,2,1,0} fusion(%param, %constant.3, %partition-id), kind=kLoop, calls=%fused_computation.14, metadata={op_name="jit(reshard_3)/reshard" }
      %bitcast_slice_fusion.14 = f32[1,1,8,128]{3,2,1,0} fusion(%param, %constant.3, %partition-id), kind=kLoop, calls=%fused_computation.15, metadata={op_name="jit(reshard_3)/reshard" }
      %all-to-all.1 = (f32[1,1,8,128]{3,2,1,0}, f32[1,1,8,128]{3,2,1,0}, f32[1,1,8,128]{3,2,1,0}, f32[1,1,8,128]{3,2,1,0}, f32[1,1,8,128]{3,2,1,0}, /*index=5*/f32[1,1,8,128]{3,2,1,0}, f32[1,1,8,128]{3,2,1,0}, f32[1,1,8,128]{3,2,1,0}, f32[1,1,8,128]{3,2,1,0}, f32[1,1,8,128]{3,2,1,0}, /*index=10*/f32[1,1,8,128]{3,2,1,0}, f32[1,1,8,128]{3,2,1,0}, f32[1,1,8,128]{3,2,1,0}, f32[1,1,8,128]{3,2,1,0}, f32[1,1,8,128]{3,2,1,0}, /*index=15*/f32[1,1,8,128]{3,2,1,0}) all-to-all(%bitcast_slice_fusion.15, %bitcast_slice_fusion.14, %bitcast_slice_fusion.13, %bitcast_slice_fusion.12, %bitcast_slice_fusion.11, /*index=5*/%bitcast_slice_fusion.10, %bitcast_slice_fusion.9, %bitcast_slice_fusion.8, %bitcast_slice_fusion.7, %bitcast_slice_fusion.6, /*index=10*/%bitcast_slice_fusion.5, %bitcast_slice_fusion.4, %bitcast_slice_fusion.3, %bitcast_slice_fusion.2, %bitcast_slice_fusion.1, /*index=15*/%bitcast_slice_fusion), channel_id=1, replica_groups=[16,16]<=[256], metadata={op_name="jit(reshard_3)/reshard" }
      %get-tuple-element.2 = f32[1,1,8,128]{3,2,1,0} get-tuple-element(%all-to-all.1), index=0
      %get-tuple-element.3 = f32[1,1,8,128]{3,2,1,0} get-tuple-element(%all-to-all.1), index=1
      %get-tuple-element.4 = f32[1,1,8,128]{3,2,1,0} get-tuple-element(%all-to-all.1), index=2
      %get-tuple-element.5 = f32[1,1,8,128]{3,2,1,0} get-tuple-element(%all-to-all.1), index=3
      %get-tuple-element.6 = f32[1,1,8,128]{3,2,1,0} get-tuple-element(%all-to-all.1), index=4
      %get-tuple-element.7 = f32[1,1,8,128]{3,2,1,0} get-tuple-element(%all-to-all.1), index=5
      %get-tuple-element.8 = f32[1,1,8,128]{3,2,1,0} get-tuple-element(%all-to-all.1), index=6
      %get-tuple-element.9 = f32[1,1,8,128]{3,2,1,0} get-tuple-element(%all-to-all.1), index=7
      %get-tuple-element.10 = f32[1,1,8,128]{3,2,1,0} get-tuple-element(%all-to-all.1), index=8
      %get-tuple-element.11 = f32[1,1,8,128]{3,2,1,0} get-tuple-element(%all-to-all.1), index=9
      %get-tuple-element.12 = f32[1,1,8,128]{3,2,1,0} get-tuple-element(%all-to-all.1), index=10
      %get-tuple-element.13 = f32[1,1,8,128]{3,2,1,0} get-tuple-element(%all-to-all.1), index=11
      %get-tuple-element.14 = f32[1,1,8,128]{3,2,1,0} get-tuple-element(%all-to-all.1), index=12
      %get-tuple-element.15 = f32[1,1,8,128]{3,2,1,0} get-tuple-element(%all-to-all.1), index=13
      %get-tuple-element.16 = f32[1,1,8,128]{3,2,1,0} get-tuple-element(%all-to-all.1), index=14
      %get-tuple-element.17 = f32[1,1,8,128]{3,2,1,0} get-tuple-element(%all-to-all.1), index=15
      %concatenate = f32[1,16,8,128]{3,2,1,0} concatenate(%get-tuple-element.2, %get-tuple-element.3, %get-tuple-element.4, %get-tuple-element.5, %get-tuple-element.6, /*index=5*/%get-tuple-element.7, %get-tuple-element.8, %get-tuple-element.9, %get-tuple-element.10, %get-tuple-element.11, /*index=10*/%get-tuple-element.12, %get-tuple-element.13, %get-tuple-element.14, %get-tuple-element.15, %get-tuple-element.16, /*index=15*/%get-tuple-element.17), dimensions={1}, metadata={op_name="jit(reshard_3)/reshard" }
      ROOT %copy_bitcast_fusion = f32[8,2048]{1,0} fusion(%concatenate), kind=kLoop, calls=%fused_computation, metadata={op_name="jit(reshard_3)/reshard" }
    }
```


这里通过拆分为细粒度的tile后进行all-to-all的方式数据交换，最后再把一个节点上多个tile合并起来得到结果。

# resharding `[M @ D, N @ (X,Y)] -> [M, N @ (D,Y,X,T)]`


```python
@jax.jit
def reshard_4(x):
  y = reshard(x, P(None, ('D', 'Y', 'X', 'T')))
  return y

x = jnp.ones((2048, 2048), jnp.float32, out_sharding=P('D', ('X', 'Y')))
compilation_args = {
    'xla_dump_to': 'tmp/reshard_4',
    'xla_dump_hlo_pass_re' : '.*'
}

traced_4 = reshard_4.trace(x)
lowered_4 = traced_4.lower()
compiled_4 = lowered_4.compile(compilation_args)
print(compiled_4.as_text())
```


```cpp
    HloModule jit_reshard_4, is_scheduled=true, entry_computation_layout={(f32[1024,64]{1,0})->f32[2048,8]{1,0}}, num_partitions=256
    
    %fused_computation (param_0: f32[2048,2048], param_1.1: u32[]) -> f32[2048,8] {
      %param_0 = f32[2048,2048]{1,0} parameter(0)
      %constant.6 = s32[] constant(0), metadata={op_name="jit(reshard_4)/reshard" }
      %param_1.1 = u32[] parameter(1)
      %convert.1 = s32[] convert(%param_1.1), metadata={op_name="jit(reshard_4)/reshard" }
      %constant.5 = s32[] constant(8), metadata={op_name="jit(reshard_4)/reshard" }
      %multiply.2 = s32[] multiply(%convert.1, %constant.5), metadata={op_name="jit(reshard_4)/reshard" }
      ROOT %dynamic-slice.2 = f32[2048,8]{1,0} dynamic-slice(%param_0, %constant.6, %multiply.2), dynamic_slice_sizes={2048,8}, metadata={op_name="jit(reshard_4)/reshard" }
    }
    
    ENTRY %main.0_spmd (param: f32[1024,64]) -> f32[2048,8] {
      %partition-id = u32[] partition-id(), metadata={op_name="jit(reshard_4)/reshard" }
      %param = f32[1024,64]{1,0} parameter(0), sharding={devices=[2,32,4]<=[2,8,4,4]T(0,2,1,3) last_tile_dim_replicate}, metadata={op_name="x"}
      %copy = f32[1024,64]{0,1} copy(%param), sharding={devices=[2,32,4]<=[2,8,4,4]T(0,2,1,3) last_tile_dim_replicate}, metadata={op_name="x"}, backend_config={"outer_dimension_partitions":["2"]}
      %all-gather = f32[1024,2048]{0,1} all-gather(%copy), channel_id=1, replica_groups=[8,32]<=[2,8,4,4]T(0,3,2,1), dimensions={1}, use_global_device_ids=true, metadata={op_name="jit(reshard_4)/reshard" }
      %copy.1 = f32[1024,2048]{1,0} copy(%all-gather), metadata={op_name="jit(reshard_4)/reshard" }, backend_config={"outer_dimension_partitions":["4"]}
      %all-gather.1 = f32[2048,2048]{1,0} all-gather(%copy.1), channel_id=2, replica_groups=[128,2]<=[2,8,4,4]T(2,1,3,0), dimensions={0}, use_global_device_ids=true, metadata={op_name="jit(reshard_4)/reshard" }
      ROOT %multiply_dynamic-slice_fusion = f32[2048,8]{1,0} fusion(%all-gather.1, %partition-id), kind=kLoop, calls=%fused_computation, metadata={op_name="jit(reshard_4)/reshard" }
    }
    
    


    W1015 18:20:08.668874 12979952 spmd_partitioner.cc:645] [SPMD] Involuntary full rematerialization. The compiler cannot go from sharding {devices=[2,32,4]<=[2,8,4,4]T(0,2,1,3) last_tile_dim_replicate} to {devices=[1,256]<=[256]} efficiently for HLO operation %param = f32[1024,64]{1,0} parameter(0), sharding={devices=[2,32,4]<=[2,8,4,4]T(0,2,1,3) last_tile_dim_replicate}, metadata={op_name="x"}. As the last resort, SPMD will replicate the tensor and then partition it to obtain the target sharding, which is inefficient. This issue will be fixed by Shardy partitioner in the future, which is tracked in b/433785288. Contact Shardy or XLA team for help.
```


这个例子xla无法handle了，他只能每个节点完全gather到所有数据之后重新切分。 

# spmd-partitioning pass

上面我对编译过程进行了dump，然后通过翻阅IR的变化定位到了reshard的具体优化pass为`spmd-partitioning`.  在deepwiki上可以看到对它的[初步分析](https://deepwiki.com/openxla/xla/7.1-spmd-partitioning#spmd-partitioning)， 这里再对其做一些记录。

这个pass最主要的逻辑在于[ReshardNoCache](https://github.com/openxla/xla/blob/b253947e64fef974e2a8b9209f97ce8a807f4296/xla/service/spmd/spmd_partitioner.cc#L511), 内部主要以模式识别分发为主：

```cpp
 if (CanReshardWithCollectivePermute(sharding(), target)) {
    return ReshardWithCollectivePermute(target);
  }

  if (auto src_tgt_dims =
          GetReshardAllToAllSourceTargetDims(sharding(), target)) {
    return ReshardWithAllToAll(target, *src_tgt_dims);
  }

  if (!target.IsTileMaximal() && sharding().ReplicateOnLastTileDim()) {
    auto try_reshard = ReshardFromPartialReplicateWithDynamicSlice(target);
/* or */ try_reshard = ReshardPartialReplicateWithAllToAll(target);
  }

  if (!sharding().IsTileMaximal() && target.ReplicateOnLastTileDim()) {
    auto try_reshard = ReshardToPartialReplicateWithAllGather(target);
/* or */ try_reshard = ReshardPartialReplicateWithAllToAll(target);
  }

  if (!sharding().IsReplicated()) {
    if (!target.IsReplicated()) {
      if (sharding().IsTiled() && target.IsTiled()) {
        auto reshard = TryComplexReshardHandling(target);
      }
    }
  }
```

1. `ReshardWithCollectivePermute`, 通过检查source sharding在当前tile或切分后tile是否满足[CollectivePermute](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#collective_permute)的spec， 他这里的约束就是src/dst的rank id只能出现一次。

2. `ReshardWithAllToAll`, 这也是检查当前tile或切分后tile是否满足[all_to_all](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#all_to_all)。

3. `ReshardPartialReplicateWithAllToAll`, 是一个更加特化的调用all_to_all的分支，支持最后一个维度为复制的情况下进行优化

4. `TryComplexReshardHandling`， 最后不支持的情况是都转移到这里，这里的逻辑是先reshape distributed tensor，然后去match target sharding的tile，检查是否有机会调用ccl primitive，如果不支持只能走all gather + slice的分支了。



