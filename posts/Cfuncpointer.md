---
title: C函数指针的使用
date: 2018-06-27 17:18:13
mathjax: true
toc: true
categories: 
-   编程语言
tags: 
-   C
---


今天发现有本C陷阱与缺陷，就打开看看，发现其中有个非常有意思的东西：
`(*(void(*)())0())`.作者介绍到，这一语句的作用是调用首地址为0的地址的程序。我对他进行了一些学习。

<!--more-->
##  函数指针
首先来一个简单的函数指针使用。

```c
void prt(void)
{
    printf("hello\n");
}

int main(int argc, char const *argv[])
{
    void (*pf)(void);
    pf=prt;
    pf();
    return 0;
}
```


我们将这个函数指针的定义做一些小小的改动，可以获得一个类型转换符。

##  类型转换符
改动定义如下。

```c
void (*pf)(void) ---> (void (*)()) 
```


从前面我们可以发现函数指针的赋值，其实也是地址的交互。那么我可以将一个地址强制转换为函数指针并使用吗？

```c
void prt(void)
{
    printf("hello\n");
}

int main(int argc, char const *argv[])
{
    void (*pf)(void);
    pf=prt;
    pf();
    
    int64_t addr=(int64_t)prt;
    pf=(void (*)())addr;
    pf();

    return 0;
}
```


输出如下：
```shell
☁  Desktop  gcc 1.c && ./a.out
hello
hello
```
显而易见这个强制转换是起作用的。

##  任意地址运行程序
那么现在可以做最后一步：

```c
void prt(void)
{
    printf("hello\n");
}

int main(int argc, char const *argv[])
{
    int64_t addr=(int64_t)prt;
    (*(void (*)())addr)();
    return 0;
}
```


输出：
```sh
☁  Desktop  gcc 1.c && ./a.out
hello
```
现在我们就可以运行任意地址的程序了~