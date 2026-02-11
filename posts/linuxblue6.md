---
title: Raspi蓝牙:编程实现录音
categories:
  - 边缘计算
date: 2019-03-04 19:33:42
tags:
-   蓝牙
-   Linux
-   树莓派
---

经过之前一系列[复杂的配置](https://zhen8838.github.io/2019/03/02/pulseaudio-compile/),我搭建好了交叉编译`pulseaudio`的环境,现在我们可以调用相应的`api`去进行录音.

<!--more-->

# 使用
我这里使用的例程为[parec-simple.c](https://freedesktop.org/software/pulseaudio/doxygen/parec-simple_8c-example.html)

当配置好交叉编译环境之后直接执行`arm-linux-gnueabihf-gcc parec-simple.c -lpulse -lpulse-simple -o record`即可.

在树莓派中执行,并通过`ctrl+c`退出录制,接着开始播放~:
```sh
pi@raspberrypi:~ $ ./record > test
^C
pi@raspberrypi:~ $ pacat -pv test --format=s16le
Opening a playback stream with sample specification 's16le 2ch 44100Hz' and channel map 'front-left,front-right'.
Connection established.
Stream successfully created.
Buffer metrics: maxlength=4194304, tlength=352800, prebuf=349276, minreq=3528
Using sample spec 's16le 2ch 44100Hz', channel map 'front-left,front-right'.
Connected to device bluez_sink.E8_07_BF_E1_1B_02.headset_head_unit (index: 5, suspended: no).
Got EOF.
Stream started.
Playback stream drained.: 55360 usec.          
Draining connection to server.
```

# 解析

程序的基本结构如下:

1.  创建一个新的采样流(pa_simple_new)


2.  读取音频流(pa_simple_read)


3.  写出输出(write)


4.  释放流资源(pa_simple_free)
