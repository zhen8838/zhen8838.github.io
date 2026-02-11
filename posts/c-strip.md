---
title: Linux下实现\r\n换行
categories:
- 操作系统
mathjax: true
toc: true
date: 2018-11-13 23:16:41
tags:
-  Linux
-  树莓派
---

今天在写`NB-iot`的程序的时候，发现一个很蛋疼的问题，程序会把接收到的数据`printf`出来，但是`linux`下`\n`就会换行，所以会出现大段的空白，需要解决一下。

<!--more-->

## 问题描述

当我执行程序时出现以下现象：
```sh
➜  build ./NB_test
Now Config is :
     tty_port  : /dev/ttyUSB0
     baud_rate : 9600
uart  init success
PI-->>NB: AT+CGATT=1
NB-->>PI:

OK

PI-->>NB: AT+CGATT?
NB-->>PI:

+CGATT:1



OK

PI-->>NB: AT+CEREG=1
NB-->>PI:

OK

PI-->>NB: AT+CEREG?
NB-->>PI:

+CEREG:1,1



OK

Connected to China Mobile Network.
q
good bye~
```

经过寻找，发现问题出现在：

```c
printf("PI-->>NB: %s\n", cmd);
```


此处是因为是发送给`NB`模组的`cmd`是`\r\n`的，所以`printf`的时候就会出现问题。



```c
printf("NB-->>PI: %s\n", UART_RX_BUF)
```


此处是因为`UART_RX_BUF`就是有段空白的，所以需要删除空白字符。


## 问题解决

为了清除多余的空格，我写了几个函数：

```c
/* @brief 去除字符串左边的空字符
 *
 * */
char *lstrip(char *str) {
    static char out[512]= {0};

    if (NULL == str) return NULL;

    char *tmp= str;
    int start= 0;

    while (isspace(*tmp++)) start++;
    int len= strlen(str) - start;

    for (int i= 0; i <= len; ++i) { out[i]= *(str + i + start); }

    return out;
}

/* @brief 去除字符串右边的空字符
 *
 * */
char *rstrip(char *str) {
    static char out[512]= {0};

    if (NULL == str) return NULL;
    char *tmp= str;
    int len= strlen(str) - 1;

    while (isspace(*(str + len))) len--;

    for (int i= 0; i <= len + 1; ++i) { out[i]= *(str + i); }
    out[len + 2]= '\0';

    return out;
}


/* @brief 去除字符串两边的空字符
 *
 * */
char *strip(char *str) {
    static char out[512]= {0};
    
    if (NULL == str) return NULL;

    char *tmp= str;
    int len= strlen(str) - 1;
    int start= 0;

    while (isspace(*tmp++)) start++;
    while (isspace(*(str + len))) len--;

    for (int i= 0; i <= len - start + 1; ++i) { out[i]= *(str + start + i); }
    out[len - start + 2]= '\0';
    
    return out;
}
```



然后在后面的程序中进行调用：


```c
if (isPrintf) printf("PI-->>NB: %s\n", rstrip(cmd));
if (isPrintf) printf("NB-->>PI: %s\n", strip(UART_RX_BUF));
```



输出结果为：

```sh
➜  build ./NB_test
Now Config is :
     tty_port  : /dev/ttyUSB0
     baud_rate : 9600
[ OK.  ] Uart Init
PI-->>NB: AT+CGATT=1
NB-->>PI: OK

PI-->>NB: AT+CGATT?
NB-->>PI: +CGATT:1



OK

PI-->>NB: AT+CEREG=1
NB-->>PI: OK

PI-->>NB: AT+CEREG?
NB-->>PI: +CEREG:1,1



OK

[ OK.  ] Connected to China Mobile Network.
```