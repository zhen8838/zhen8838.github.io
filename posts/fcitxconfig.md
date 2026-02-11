---
title: fcitx配置
categories:
  - 工具使用
mathjax: true
toc: true
date: 2018-11-26 15:50:37
tags:
- 踩坑经验
---

昨天刚刚重新装了`Ubuntu`，最舒服的一点就是`Nvida`的显卡驱动分分钟安装。但是也有一些不爽的，像这个`fcitx`的配置我又找了老半天。

<!--more-->

# 配置

我的都是使用右`CTRL`切换输入法的，所以我巨讨厌Windows ( 因为我改不了233


在`Ubuntu`下也会出现一些问题，不过都还好解决，就是`fcitx`不保存我的配置，所以需要去强行配置一下～

修改他的配置文件，把`switch_key`修改为`R_CTRL`,并且把权限都改成只读～
```sh
$ vi ~/.config/fcitx/config 
# 修改内容
$ sudo chmod 444 ~/.config/fcitx/config 
$ ll ~/.config/fcitx/config 
-r--r--r-- 1 zqh zqh 3038 11月 26 15:56 /home/zqh/.config/fcitx/config
```
