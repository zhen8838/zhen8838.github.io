---
title: k210_双核测试
mathjax: true
toc: true
date: 2018-11-11 20:01:41
categories:
  - 边缘计算
tags:
-   K210
---


k210由两个核组成，我想测试一下两个核的运行情况。编写了以下程序：

<!--more-->


```c
/*
 * @Author: Zheng Qihang 
 * @Date: 2018-11-11 20:10:46 
 * @Last Modified by:   Zheng Qihang 
 * @Last Modified time: 2018-11-11 20:10:46 
 */

#include "bsp.h"
#include <stdio.h>

volatile int core0_value= 0;
volatile int core1_value= 0;
volatile int global_value= 0;

int core1_function(void *ctx) {
    uint64_t core= current_coreid();

    printf("Core %ld Hello world\n", core);
    while (1) {
        core1_value++;
        global_value++;
        msleep(500);
        if (core1_value % 5 == 0) {
            printf("core1 %d global %d\n", core1_value, global_value);
        }
    };
}

int main() {
    uint64_t core= current_coreid();
    printf("Core %ld Hello world\n", core);
    register_core1(core1_function, NULL);
    while (1) {
        core0_value++;
        global_value--;
        msleep(500);
        if (core0_value % 5 == 0) {
            printf("core0 %d global %d\n", core0_value, global_value);
        }
    };
}
```




# 编译运行

```sh
➜  build cmake .. -DPROJ=double_core && make && make clean
➜  build python3 isp.py -p /dev/ttyUSB0 -b 115200 double_core.bin

[INFO] Rebooting...
Core 0 Hello world
Core 1 Hello world
core0 5 global 0
core1 5 global -1
core0 10 global 0
core1 10 global -1
core0 15 global 0
core1 15 global -1
core0 20 global 0
core1 20 global -1
```


# 思考

发现两个核应该是以同一个速度进行运行的，并且是同时运行的，对全局变量具有相同的权限。但是对于同一个外设，同一时间的优先级还是未知的。