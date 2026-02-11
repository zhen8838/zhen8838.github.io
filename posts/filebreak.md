---
title: ntfs文件损坏
categories:
  - 操作系统
mathjax: true
toc: true
date: 2018-12-02 15:54:07
tags:
-   Linux
-   踩坑经验
---


今天真的是背，我的一个`md`文档莫名奇妙损坏了。导致我的`hexo`无法生成博客。

<!--more-->

# 问题出现

因为的是双系统，所以在两个系统中切换的时候，由于`Windows`的快速关机的~~bug~~特性，会导致在`Linux`那个硬盘只读，需要使用`ntfsfix`，我怀疑就是因为经常这样，导致我的那个文件被破坏了，并且在`Linux`下还无法删除。



![错误](./filebreak/1.png)





# 问题解决

我重启到`Windows`下，然后删除了这个文件，后来在群里面问了一波，需要再`Windows`关机的时候选择重启，这样`Windows`就不会使用快速关机了。我可以在重启的时候去`grub`关机。真麻烦。