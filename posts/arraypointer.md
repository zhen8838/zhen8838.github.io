---
title: C数组指针
date: 2018-06-28 11:03:14
mathjax: true
tags:
-   C
categories:
-   编程语言
---

## 定义

今天看c陷阱与缺陷，发现这个数组指针挺有意思的。
首先定义一个数组指针：`int (*p)[4]`。程序如下：

<!--more-->


```c
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char const *argv[])
{
    int calendar[3][4]={
    {1,2,3,4,},
    {5,6,7,8,},
    {9,10,11,12,},};
    int (*p)[4];
    int i;
    p=calendar;

    return 0;
}
```


通过`p=calendar`，这样`p`就指向了`calendar`第一个元素，也就是`calendar`的3个有着4个元素的元素的第一个元素。

## 例子

这里通过一系列的小例子去实验说明`p`的用法。

1.  例子1
    
    ```c
    printf("%d\n",calendar[2][2]);
    printf("%d\n",*(*(calendar+2)+2));
    printf("%d\n",*(*(p+2)+2));
    ```
    

    输出
    ```sh
    11
    11
    11
    ```
    这个例子还是挺好理解的，`p=calendar`那么他们的地址相同，用法相同即可寻找到对应的元素。
    当然要注意一点`*(calendar+2)`这样才能指向二维数组的第3行。

2.  例子2
    
    ```c
    printf("%d\n",calendar[2][2]);
    printf("%d\n",*(*calendar+10));
    printf("%d\n",*(*p+10));
    ```
    

    输出
    ```sh
    11
    11
    11
    ```
    
    ```c
    printf("%p\n",p);
    printf("%p\n",calendar);
    printf("%p\n",*p);
    printf("%p\n",*calendar);
    ```
    

    输出
    ```sh
    0x7ffdcddf8480
    0x7ffdcddf8480
    0x7ffdcddf8480
    0x7ffdcddf8480
    ```
    这里要注意虽然他们的指向的地址是一样的，但是只能使用`*(*p+10)`这样的形式，`*(p+10)`这个形式是不被允许的。
    这个应该是由于c语言中对于`&calendar==calendar`的定义，虽然编译器会提示warning，但是其返回值是成立的。

3.  例子3
    
    ```c
    printf("%d\n",(**p+10));
    printf("%d\n",**p);
    printf("%d\n",calendar[0][0]);
    ```
    

    输出
    ```sh
    11
    1
    1
    ```
    这个`**p`指向的就是二维数组第一个元素的值。