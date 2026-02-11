---
title: stm32使用静态库
date: 2018-08-16 10:18:40
categories: 
-   边缘计算
tags: 
-   stm32
---


我这两天写程序的时候，使用了大量的宏定义去开启和关闭代码块。但是我发现每次我切换一个宏定义的时候就会将所以的stm32工程中的所有文件进行重新编译，耗费太长时间，效率不如之前用注释代码块的方式。所以我仔细看了看。发现每次编译stm32hal库的时间是最长的，那么我就可以将所有的hal库文件生成一个静态库，让编译的时候连接即可。

<!--more-->


# 生成静态库

我使用的是偷懒的方法。首先我的工程是stm32cubemx生成的，其中的makefile，直接可以进行make。编译之后，自动在工程的`build`目录下生成了一系列的.o .d文件。那么我的任务就是将这些.o文件合成一个静态库。

首先查看那些是hal库的.o文件
```sh
➜  build ls stm32l4xx_hal*.o
stm32l4xx_hal_adc_ex.o  stm32l4xx_hal_dma_ex.o    stm32l4xx_hal_flash.o          stm32l4xx_hal_i2c_ex.o  stm32l4xx_hal.o         stm32l4xx_hal_rcc_ex.o  stm32l4xx_hal_tim.o
stm32l4xx_hal_adc.o     stm32l4xx_hal_dma.o       stm32l4xx_hal_flash_ramfunc.o  stm32l4xx_hal_i2c.o     stm32l4xx_hal_pwr_ex.o  stm32l4xx_hal_rcc.o     stm32l4xx_hal_uart_ex.o
stm32l4xx_hal_cortex.o  stm32l4xx_hal_flash_ex.o  stm32l4xx_hal_gpio.o           stm32l4xx_hal_msp.o     stm32l4xx_hal_pwr.o     stm32l4xx_hal_tim_ex.o  stm32l4xx_hal_uart.o
```
要注意这里面**stm32l4xx_hal_msp.o**不能加入静态库！，因为这个文件是经常被修改的！所以我们先要删除这个文件
```sh
➜  build rm stm32l4xx_hal_msp.o
```
接下来生成静态库（多添加两个文件是因为我觉得这两个文件也不太会改变）：
```sh
➜  build arm-none-eabi-ar cr  libhal.a  stm32l4xx_hal*.o system_stm32l4xx.o startup_stm32l431xx.o
➜  build mv libhal.a ../
```
移动到`libhal.a`到与`makefile`同级目录后修改makefile

-   注释源文件

将c源文件和asm源文件中的库文件注释掉。
```makefile

######################################
# source
######################################
# C sources
C_SOURCES =  \
Src/main.c \
Src/gpio.c \
Src/adc.c \
Src/tim.c \
Src/usart.c \
Src/delay.c \
Hardware/BH1750/BH1750.c \
Hardware/DHT11/DHT11_BUS.c \
Hardware/GPS/gps.c \
Hardware/OLED/oled.c \
Src/stm32l4xx_it.c \
Src/stm32l4xx_hal_msp.c \
# Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_adc.c \
# Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_adc_ex.c \
# Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_tim.c \
# Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_tim_ex.c \
# Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_uart.c \
# Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_uart_ex.c \
# Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal.c \
# Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_i2c.c \
# Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_i2c_ex.c \
# Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_rcc.c \
# Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_rcc_ex.c \
# Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_flash.c \
# Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_flash_ex.c \
# Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_flash_ramfunc.c \
# Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_gpio.c \
# Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_dma.c \
# Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_dma_ex.c \
# Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_pwr.c \
# Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_pwr_ex.c \
# Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_cortex.c \
# Src/system_stm32l4xx.c  

# ASM sources
ASM_SOURCES =  \
# startup_stm32l431xx.s
```

-   添加连接库

在`LIBS`后面添加上静态库的名字，然后编译即可
```makefile
#######################################
# LDFLAGS
#######################################
# link script
LDSCRIPT = STM32L431RBTx_FLASH.ld

# libraries
LIBS = -lc -lm -lnosys libhal.a
LIBDIR = 
LDFLAGS = $(MCU) -specs=nano.specs -T$(LDSCRIPT) $(LIBDIR) $(LIBS) -Wl,-Map=$(BUILD_DIR)/$(TARGET).map,--cref -Wl,--gc-sections
```




# 效果

编译快了许多～～～

