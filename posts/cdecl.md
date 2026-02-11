---
title: C语义转换
date: 2018-09-03 23:42:24
tags: 
-   C
categories: 
-   工具使用
---

# 你有没有曾经对c语言的定义苦恼？

<!--more-->


今天发现一个有趣的工具：[cdecl](https://cdecl.org/)
点击进入即可。
这是一个可以将c语言的定义转成英语的小工具，对于一些看着头疼的定义直接可以给出解释。
例如：`int (*(*foo)(void ))[3]`
解释：`declare foo as pointer to function (void) returning pointer to array 3 of int`

但是经过我的测试也发现了一些不足，比如我拿了一个极端点的：`(*(void (*)())0)()`，程序就蒙了(ಡωಡ)。

我忽然有点想自己写个软件用它的接口，再搞个翻译，给一些人用，不是美滋滋？
噢对了，linux上可以直接安装这个～～