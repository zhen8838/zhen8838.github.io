---
title: k210_GPIO使用
date: 2018-11-02 19:37:53
categories:
  - 边缘计算
tags:
-   K210
---

听说学会了点灯就学会了一切2333

<!--more-->


# 1.看原理图
我的是绿色板子，观察原理图。`LED0`对应`IO12`，`LED1`对应`IO12`.

# 2.看手册
我看了一会，发现这个芯片有个很牛逼的东西`现场可编程 IO 阵列 (Field programmable IO Array)`，他可以自由映射内部的255个功能到外部48个io！真的强

但是这个芯片的缺憾是通用GPIO只有8个，从`FUNC_GPIO0`到`FUNC_GPIO7`,但是～他还有更强的**32个高速GPIO**
# 3.coding

1.  功能映射


    因为有`FPIOA`的存在，所以我们需要先将io映射到想要的功能：
    
    ```c
        /* 绿色板子IO12-->LED0 IO13-->LED1 */
        fpioa_set_function(12, FUNC_GPIO1);
        fpioa_set_function(13, FUNC_GPIO2);
    ```
    


2.  初始化GPIO

    
    ```c
    gpio_init();/* 初始化gpio */
    ```
    

3.  设置GPIO模式

    这里的`1`就是对应了`FUNC_GPIO1`～
    
    ```c
    /* typedef enum _gpio_drive_mode
    {
        GPIO_DM_INPUT,
        GPIO_DM_INPUT_PULL_DOWN,
        GPIO_DM_INPUT_PULL_UP,
        GPIO_DM_OUTPUT,
    } gpio_drive_mode_t; */
    gpio_set_drive_mode(1, GPIO_DM_OUTPUT); 
    ```
    


4.  设置状态
    这里没有什么好说的，`HIGH`对应`1`，`LOW`对应`0` 
    
    ```c
    gpio_pin_value_t value1= GPIO_PV_HIGH, value2= GPIO_PV_LOW;
    gpio_set_pin(1, value1);
    ```
    


5.  完整程序
    这里的**sleep**函数是sdk中自带的，这个自带的sleep有三种，完全满足我们的一般要求。
    
    ```c
    #include "fpioa.h"
    #include "gpio.h"
    #include <stdio.h>
    #include <unistd.h>


    int main(void) {
        /* 老板子io 11 12 是led */
        fpioa_set_function(12, FUNC_GPIO1);
        fpioa_set_function(13, FUNC_GPIO2);

        gpio_init();/* 初始化gpio */
        gpio_set_drive_mode(1, GPIO_DM_OUTPUT);
        gpio_set_drive_mode(2, GPIO_DM_OUTPUT);
        gpio_pin_value_t value1= GPIO_PV_HIGH, value2= GPIO_PV_LOW;
        gpio_set_pin(1, value1);
        gpio_set_pin(2, value2);
        while (1) {
            sleep(1);
            gpio_set_pin(1, value1= !value1);
            gpio_set_pin(2, value2= !value2);
        }
        return 0;
    }
    ```
    


# 4.编译运行
```sh
➜  build cmake .. -DPROJ=gpio_led && make
➜  build python3 isp.py -p /dev/ttyUSB0 -b 115200 gpio_led.bin
```
上电即可看到现象咯。