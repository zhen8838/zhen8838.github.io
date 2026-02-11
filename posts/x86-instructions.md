---
title: x86指令集使用汇总
mathjax: true
toc: true
categories:
  - 体系结构
date: 2021-07-05 12:59:12
tags:
  - 指令集
---

一些x86的硬件指令集相关信息。

<!--more-->

# x86指令集时间顺序

```sh
MMX(1996) 3DNow!(1998) SSE(1999) SSE2(2001) SSE3(2004) SSSE3(2006) SSE4(2006) SSE5(2007) AVX(2008) F16C(2009) XOP(2009) FMA(FMA4: 2011, FMA3: 2012) AVX2(2013) AVX-512(2015) 
```
现在一般支持AVX2的基本上都支持FMA的，FMA是乘加融合(fuse mul add)的意思，这个对于卷积啊还有其他操作还是很重要的。


# 函数多架构分发

[gcc中的静态函数分发](https://gcc.gnu.org/onlinedocs/gcc-4.9.0/gcc/Function-Multiversioning.html)可以用`__attribute__ ((target ("default")))`等前缀对函数进行修饰。

[gcc中动态函数分发](https://gcc.gnu.org/onlinedocs/gcc-4.8.2/gcc/X86-Built-in-Functions.html)，通过判断cpu型号来执行各种函数。


# 其他体系结构优化

## 给出硬件切换最少的指令集

对于一些专有npu的指令,一些内存搬运需要跳stride,如果我们是连续的内存,让指令内部硬件执行最少的跳转次数是最好的.
```
    shape_n : 8
    shape_c : 3
    shape_h : 1
    shape_w : 48
    |
    v
    shape_n : 1
    shape_c : 1
    shape_h : 1
    shape_w : 576
```