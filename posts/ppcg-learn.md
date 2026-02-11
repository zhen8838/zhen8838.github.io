---
title: ppcg 学习
mathjax: true
toc: true
categories:
  - 编译器
date: 2023-05-31 16:22:15
tags:
- 多面体模型
---

学习一个ppcg的整体流程与细节.

<!--more-->

# ppcg启动

ppcg编译好之后生成的ppcg并不是elf文件, 而是一个shell脚本, 实际上需要执行./libs/ppcg.

# ppcg代码转换

1. 转换的核心逻辑是callback, 注册在clang的parser中, 在parser完成后自动进行转换.


# isl打印调试

1. isl是通过宏构造了一堆`isl_${type}_dump`和`isl_${type}_to_str`的函数. 比如isl_schedule_dump/isl_union_set_dump/isl_set_dump等等.