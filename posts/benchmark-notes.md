---
title: benchmark的经验与技巧
mathjax: true
toc: true
categories:
  - 边缘计算
date: 2024-08-08 16:32:50
tags:
-   踩坑经验
---

为了公平对比性能都不是一件容易的事情. 各个框架的runtime都可能存在一些不同配置, 需要把他们安排到统一基准线去对比才有意义.

<!--more-->

# TVM

## meta schedule 禁用多线程

```python
from tvm import meta_schedule as ms

# disable parallel when num cores = 1.
rules = ms.ScheduleRule.create('llvm')
newrules = []
for rule in rules:
  if isinstance(rule, ms.schedule_rule.ParallelizeVectorizeUnroll):
    newrules.append(ms.schedule_rule.ParallelizeVectorizeUnroll(-1, 64, [0, 16, 64, 512], True))
  else:
    newrules.append(rule.clone())
mutators = ms.Mutator.create('llvm')
newmutators = []
for m in mutators:
  if isinstance(m, ms.mutator.MutateParallel):
    newmutators.append(ms.mutator.MutateParallel(-1))
  else:
    newmutators.append(m.clone())
sg = ms.space_generator.PostOrderApply(sch_rules=newrules)

database = ms.tune_tir(
    mod=prim_func,
    target="llvm --num-cores=1",
    max_trials_global=64,
    num_trials_per_iter=64,
    work_dir="./tune_tmp",
    num_tuning_cores=4,
    space=sg,
)

sch = ms.tir_integration.compile_tir(database, prim_func, "llvm --num-cores=1")
```

# numpy

## numpy 禁止多线程

# iree

## iree benchmark

基本流程

```sh
iree-compile \
  iree/matmul.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=host \
  --iree-llvmcpu-target-cpu-features=host \
  --iree-llvmcpu-loop-interleaving \
  --iree-llvmcpu-slp-vectorization \
  --iree-llvmcpu-loop-unrolling \
  --iree-llvmcpu-loop-vectorization \
  --compile-mode=std \
  -o out/iree/matmul.vmbf

iree-benchmark-module \
  --module=out/iree/matmul.vmbf \
  --device=local-task \
  --task_topology_cpu_ids=0,1,2,3 \
  --function=abs \
  --input=1x1024x2048xf32=2 \
  --input=1x2048x512xf32=1
```

--iree-hal-target-backends:
  - cuda
  - llvm-cpu,    - cpu target 存在下面这些类型
    - aarch64    - AArch64 (little endian)
    - aarch64_32 - AArch64 (little endian ILP32)
    - aarch64_be - AArch64 (big endian)
    - arm        - ARM
    - arm64      - ARM64 (little endian)
    - arm64_32   - ARM64 (little endian ILP32)
    - armeb      - ARM (big endian)
    - riscv32    - 32-bit RISC-V
    - riscv64    - 64-bit RISC-V
    - thumb      - Thumb
    - thumbeb    - Thumb (big endian)
    - wasm32     - WebAssembly 32-bit
    - wasm64     - WebAssembly 64-bit
    - x86        - 32-bit X86: Pentium-Pro and above
    - x86-64     - 64-bit X86: EM64T and AMD64
  - metal-spirv
  - rocm
  - vmvx
  - vmvx-inline
  - vulkan-spirv

--iree-benchmark-module --list_devices
  - local-sync://   synchronous, single-threaded driver that executes work inline
  - local-task://  asynchronous, multithreaded driver built on IREE's "task" system


# 系统性能峰值测试

## 带宽性能测试

暂时没有找到测试l1,l2 cache带宽的工具, 但是有[lm_bench](https://github.com/intel/lmbench.git)可以测试带宽的各种指标. 他提供了传统的bw mem:

bw_mem_cp [ -P <并行度> ] [ -W <热身次数> ] [ -N <重复次数> ] 大小 rd|wr|rdwr|cp|fwr|frd|bzero|bcopy [对齐]

描述

bw_mem 分配两倍指定内存量，将其归零，然后将前半部分复制到后半部分。将每秒移动的兆字节数作为结果进行报告。
大小规范可能以“k”或“m”结尾，表示千字节 (* 1024) 或兆字节 (* 1024 * 1024)。

输出

输出格式为 CB"%0.2f %.2f\n", 兆字节，兆字节每秒，即
8.00 25.33

bw_mem 中有九种不同的内存基准。它们各自测量读取、写入或复制数据的方法略有不同。

1. rd,  测量将数据读入处理器的时间。它计算整数值数组的总和。它每次访问四个字。
2. wr,  测量将数据写入内存的时间。它为整数值数组的每个内存分配一个常量值。它每次访问四个字。
3. rdwr 测量将数据读入内存然后将数据写入同一内​​存位置的时间。对于数组中的每个元素，它将当前值添加到运行总和中，然后再为元素分配新的（常量）值。它每次访问四个字。
4. cp   测量将数据从一个位置复制到另一个位置的时间。它执行数组复制：dest[i] = source[i]。它每次访问四个字。
5. frd  测量将数据读入处理器的时间。它计算整数值数组的总和。
6. fwr  测量将数据写入内存的时间。它为整数值数组的每个内存分配一个常量值。
7. fcp  测量将数据从一个位置复制到另一个位置的时间。它执行数组复制：dest[i] = source[i]。
8. bzero  测量系统清零内存的速度。
9. bcopy  测量系统复制数据的速度。

内存利用率

此基准测试最多可将请求的内存移动三倍。Bcopy 将使用 2-3 倍的内存带宽：从源读取一次，然后写入目标。写入通常会导致缓存行读取，然后在稍后的某个时间点写回缓存行。如果处理器架构实现了“加载缓存行”和“存储缓存行”指令（以及“getcachelinesize”），内存利用率可能会减少 1/3。

首先测到频率为
```sh
/usr/lib/lmbench/bin/x86_64-linux-gnu/mhz
2448 MHz, 0.4085 nanosec clock
```
再测l1大小的读取
```sh
/usr/lib/lmbench/bin/x86_64-linux-gnu/bw_mem 32K rd -N=10
0.032000 100634.84
```
然后计算byte/cycle:

$$
\begin{aligned}
\frac{Byte}{Hz} = \frac{MB\times 1024 \times 1024}{S} \div \frac{MHz\times 10^6}{S} = 43 B/Hz
\end{aligned}
$$

计算写入带宽, 这里就勉强算一半吧.
```sh
/usr/lib/lmbench/bin/x86_64-linux-gnu/bw_mem 32K wr -N=10
0.032000 56970.93
```


