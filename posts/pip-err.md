---
title: 安装conda之后pip执行全局的问题
categories:
  - 编程语言
date: 2019-03-21 11:31:10
tags:
-   踩坑经验
-   Python
---

今天我要装3个`Tensorflow`但是我发现每次装一个就卸载前面那个,我很奇怪,然后就发现了这个大问题!

<!--more-->

# 问题描述

我是发现我`which`出来的`pip`和执行的`pip`不是同一个`pip`:
```sh
(tf1.12) ➜  TensorFlow2.0Tutorials-master which pip  
/home/zqh/miniconda3/envs/tf1.12/bin/pip
(tf1.12) ➜  TensorFlow2.0Tutorials-master pip --version              
pip 19.0.3 from /home/zqh/.local/lib/python3.6/site-packages/pip (python 3.6)
```
让我很难受啊.

# 解决方案
直接卸载全局的`pip`! 然后都用虚拟环境.记得卸载前先关闭虚拟环境.
```sh
➜  gitio python3 -m pip uninstall pip
Uninstalling pip-19.0.3:
  Would remove:
    /home/zqh/.local/bin/pip
    /home/zqh/.local/bin/pip3
    /home/zqh/.local/bin/pip3.6
    /home/zqh/.local/lib/python3.6/site-packages/pip-19.0.3.dist-info/*
    /home/zqh/.local/lib/python3.6/site-packages/pip/*
Proceed (y/n)? y
➜  gitio sudo python3 -m pip uninstall pip
```
