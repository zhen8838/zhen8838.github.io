---
title: 解决Ubuntu下nvidia驱动导致画面撕裂
categories:
  - 工具使用
date: 2019-03-20 10:07:24
tags:
-   踩坑经验
---

我最近装了这个`RTX2060`,虽然`Tensorflow`是编译成功安装了,但是我用的时候发现我的屏幕画面撕裂的太厉害了,我滚动代码都会导致花屏,把我气死.今天终于解决了.

<!--more-->

# 解决方案

首先,什么强制同步啥的都没什么用.我的问题在于`Linux`的内核版本,如果是`4.15`内核的朋友赶紧升级一波:

```sh
sudo apt install linux-generic-hwe-18.04-edge
```

升级到`4.18`内核之后,重装了`nvidia 4.15`驱动.成功解决.

```
                          ./+o+-       zqh@pc
                  yyyyy- -yyyyyy+      OS: Ubuntu 18.04 bionic
               ://+//////-yyyyyyo      Kernel: x86_64 Linux 4.18.0-16-generic
           .++ .:/++++++/-.+sss/`      Uptime: 10m
         .:++o:  /++++++++/:--:/-      Packages: 2671
        o:+o+:++.`..```.-/oo+++++/     Shell: zsh 5.4.2
       .:+o:+o/.          `+sssoo+/    Resolution: 3000x1920
  .++/+:+oo+o:`             /sssooo.   DE: GNOME 
 /+++//+:`oo+o               /::--:.   WM: GNOME Shell
 \+/+o+++`o++o               ++////.   WM Theme: 
  .++.o+++oo+:`             /dddhhh.   GTK Theme: Adwaita [GTK2/3]
       .+.o+oo:.          `oddhhhh+    Icon Theme: Adwaita
        \+.++o+o``-````.:ohdhhhhh+     Font: Cantarell 11
         `:o+++ `ohhhhhhhhyo++os:      CPU: Intel Core i7-7700 @ 8x 4.2GHz [27.8°C]
           .o:`.syhhhhhhh/.oo++o`      GPU: GeForce RTX 2060
               /osyyyyyyo++ooo+++/     RAM: 2058MiB / 16003MiB
                   ````` +oo+++o\:    
                          `oo++.     
```
