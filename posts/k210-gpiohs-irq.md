---
title: k210 高速gpio与中断
date: 2018-11-02 20:40:18
categories:
  - 边缘计算
tags:
-   K210
---


高速gpio拥有更快的反转能力，并且一共有32个io，足够我们使用。


<!--more-->


# K210外部中断
可以将任意的外部中断分配到cpu中。

# 代码

```c

#include "fpioa.h"
#include "gpiohs.h"
#include "sysctl.h"
#include <stdio.h>
#include <unistd.h>

int irq_flag= 1;
/* clang-format off */
#define PIN_LED     12
#define PIN_KEY     14
/* clang-format on */

void irq_gpiohs2(void *gp) {
    irq_flag= gpiohs_get_pin(2); /* 进入中断后读取KEY值 */

    printf("IRQ The PIN is %d\n", irq_flag);

    if (irq_flag) /* 设置LED状态 */
        gpiohs_set_pin(3, GPIO_PV_LOW);
    else
        gpiohs_set_pin(3, GPIO_PV_HIGH);
}

int main(void) {
    plic_init();         /* 初始化中断 */
    sysctl_enable_irq(); /* 使能系统中断 */

    fpioa_set_function(PIN_LED, FUNC_GPIOHS3); /* LED映射高速GPIO3 */
    gpiohs_set_drive_mode(3, GPIO_DM_OUTPUT);  /* 设置GPIOHS3 输出 */
    gpio_pin_value_t value= GPIO_PV_HIGH;      /* 初始化GPIO状态 */
    gpiohs_set_pin(3, value);                  /* GPIO状态设置 */

    fpioa_set_function(PIN_KEY, FUNC_GPIOHS2); /* KEY映射GPIOHS2 */
    gpiohs_set_drive_mode(2, GPIO_DM_INPUT);   /* 设置GPIOHS2 输入 */
    gpiohs_set_pin_edge(2, GPIO_PE_BOTH);      /* 设置双沿触发 */
    gpiohs_set_irq(2, 1, irq_gpiohs2); /* 设置回调函数，优先级1 */

    while (1) {
        sleep(1);
        if (irq_flag) { gpiohs_set_pin(3, value= !value); }
        int val= gpiohs_get_pin(2); /* 正常状态读取KEY值 */
        printf("The PIN is %d\n", val);
    }
    return 0;
}

``

`