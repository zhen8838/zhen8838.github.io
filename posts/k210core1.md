---
title: k210中使用core1
date: 2019-04-24 12:30:39
categories:
  - 边缘计算
tags:
-   K210
-   踩坑经验
---

记录一下在k210中使用`core1`遇到的错误.

<!--more-->

# 1.无法结束kpu_run_kmodel

我想把`kpu task`放在第二个核去运行,但是一直无法结束这个任务,然后发现原来第二个核也要添加中断使能`sysctl_enable_irq()`...