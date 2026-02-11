---
title: linux stm32 开发
date: 2018-08-13 20:50:15
tags: 
-   Linux
-   stm32
categories: 
-   边缘计算
---


最近换了双系统，发现还是linux下面写程序爽。windows还是比较适合打游戏233.
这篇文章记录一下linux下开发stm32的一些东西。

<!--more-->



# 安装vscode以及交叉编译器

在[官网](https://code.visualstudio.com/Download)下载即可


# 安装交叉编译链

下载[地址](https://launchpad.net/gcc-arm-embedded/+download)
下载好后解压并且设置路径，可以看这篇[文章](https://blog.csdn.net/embbnux/article/details/17616809)中的安装交叉编译链部分。


# 安装stm32cubemx
下载[地址](https://www.st.com/content/st_com/en/products/development-tools/software-development-tools/stm32-software-development-tools/stm32-configurators-and-code-generators/stm32cubemx.html)
下载好后执行`××.linux`即可

# 安装jlink驱动

下载[地址](http://www.segger.com/jlink-software.html)
Ubuntu的话双击即可安装。
安装成功后运行如下出现信息说明成功，输入`q`退出。
```sh
➜  gitio JLinkExe
SEGGER J-Link Commander V6.12j (Compiled Feb 15 2017 18:03:38)
DLL version V6.12j, compiled Feb 15 2017 18:03:30

Connecting to J-Link via USB...O.K.
Firmware: J-Link ARM V8 compiled Jan 31 2018 18:34:52
Hardware version: V8.00
S/N: 20080643
License(s): RDI,FlashDL,FlashBP,JFlash,GDBFull
VTref = 3.371V


Type "connect" to establish a target connection, '?' for help
J-Link>
```

# 使用cubemx建立一个makefile工程

建立工程我就不赘述了。这里放一下vscode的配置
```json
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "${workspaceFolder}/Inc"
            ],
            "defines": [
                "USE_HAL_DRIVER",
                "STM32L431xx"
            ],
            "compilerPath": "/opt/gccStm32/bin/arm-none-eabi-gcc",
            "cStandard": "c99",
            "cppStandard": "c++11",
            "intelliSenseMode": "clang-x64"
        }
    ],
    "version": 4
}
```

# 编译

编译只需要`make`即可在`build`文件下生成`bin`和`hex`文件
```sh
➜  gcclight make
arm-none-eabi-gcc -c -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -DUSE_HAL_DRIVER-DSTM32L431xx -IInc -IHardware/BH1750 -IHardware/DHT11 -IHardware/GPS -IHardware/OLED -IDrivers/STM32L4xx_HAL_Driver/Inc -IDrivers/STM32L4xx_HAL_Driver/Inc/Legacy -IDrivers/CMSIS/Device/ST/STM32L4xx/Include -IDrivers/CMSIS/Include -Og -Wall -fdata-sections -ffunction-sections -g -gdwarf-2 -MMD -MP -MF"build/main.d" -Wa,-a,-ad,-alms=build/main.lst Src/main.c -o build/main.o
arm-none-eabi-gcc build/main.o build/gpio.o build/adc.o build/tim.o build/usart.o build/delay.o build/BH1750.o build/DHT11_BUS.o build/gps.o build/oled.o build/stm32l4xx_it.o build/stm32l4xx_hal_msp.o build/stm32l4xx_hal_adc.o build/stm32l4xx_hal_adc_ex.o build/stm32l4xx_hal_tim.o build/stm32l4xx_hal_tim_ex.o build/stm32l4xx_hal_uart.o build/stm32l4xx_hal_uart_ex.o build/stm32l4xx_hal.o build/stm32l4xx_hal_i2c.o build/stm32l4xx_hal_i2c_ex.o build/stm32l4xx_hal_rcc.o build/stm32l4xx_hal_rcc_ex.o build/stm32l4xx_hal_flash.o build/stm32l4xx_hal_flash_ex.o build/stm32l4xx_hal_flash_ramfunc.o build/stm32l4xx_hal_gpio.o build/stm32l4xx_hal_dma.o build/stm32l4xx_hal_dma_ex.o build/stm32l4xx_hal_pwr.o build/stm32l4xx_hal_pwr_ex.o build/stm32l4xx_hal_cortex.o build/system_stm32l4xx.o build/startup_stm32l431xx.o -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard-specs=nano.specs -TSTM32L431RBTx_FLASH.ld  -lc -lm -lnosys  -Wl,-Map=build/gcclight.map,--cref -Wl,--gc-sections -o build/gcclight.elf
arm-none-eabi-size build/gcclight.elf
   text    data     bss     dec     hex filename
  18160     120    2128   20408    4fb8 build/gcclight.elf
arm-none-eabi-objcopy -O ihex build/gcclight.elf build/gcclight.hex
arm-none-eabi-objcopy -O binary -S build/gcclight.elf build/gcclight.bin
```

# 烧录

连接到jlink
我使用如下脚本执行。
```sh
#!/usr/bin/expect
# 30s 超时
set timeout 30
# 执行
spawn JLinkExe

# 启动成功
expect "J-Link>"
# 连接
send "connect\r"

# 检测到默认设备
expect "<Default>:"
# 确认设备
send "\r"

# 选择连接方式
expect "TIF>"
# 确认连接方式
send "S\r"

# 选择速度
expect "Speed>"
# 确认速度
send "\r"

# 用于测试
# # 连接成功
# expect "J-Link>"
# # 重启
# send "rx 20\r"

# 烧录程序
# 如果连接成功
expect "identified."
# 烧录程序
send "loadbin build/gcclight.bin 0x8000000\r"

# 如果烧录成功
expect "O.K."
# 重启
send "rx 20\r"

# 如果重启成功
expect "J-Link>"
# 运行程序
send "g\r"


# 如果运行成功
expect "J-Link>"
# 退出jlink
send "qc\r"

# 将控制权交给用户
interact
```


终端输入如下

```sh
➜  gcclight ./install.sh
spawn JLinkExe
SEGGER J-Link Commander V6.12j (Compiled Feb 15 2017 18:03:38)
DLL version V6.12j, compiled Feb 15 2017 18:03:30

Connecting to J-Link via USB...O.K.
Firmware: J-Link ARM V8 compiled Jan 31 2018 18:34:52
Hardware version: V8.00
S/N: 20080643
License(s): RDI,FlashDL,FlashBP,JFlash,GDBFull
VTref = 3.371V


Type "connect" to establish a target connection, '?' for help
J-Link>connect
Please specify device / core. <Default>: STM32L431RB
Type '?' for selection dialog
Device>
Please specify target interface:
  J) JTAG (Default)
  S) SWD
TIF>S
Specify target interface speed [kHz]. <Default>: 4000 kHz
Speed>
Device "STM32L431RB" selected.


Found SWD-DP with ID 0x2BA01477
Found SWD-DP with ID 0x2BA01477
AP-IDR: 0x24770011, Type: AHB-AP
AHB-AP ROM: 0xE00FF000 (Base addr. of first ROM table)
Found Cortex-M4 r0p1, Little endian.
FPUnit: 6 code (BP) slots and 2 literal slots
CoreSight components:
ROMTbl 0 @ E00FF000
ROMTbl 0 [0]: FFF0F000, CID: B105E00D, PID: 000BB00C SCS
ROMTbl 0 [1]: FFF02000, CID: B105E00D, PID: 003BB002 DWT
ROMTbl 0 [2]: FFF03000, CID: B105E00D, PID: 002BB003 FPB
ROMTbl 0 [3]: FFF01000, CID: B105E00D, PID: 003BB001 ITM
ROMTbl 0 [4]: FFF41000, CID: B105900D, PID: 000BB9A1 TPIU
ROMTbl 0 [5]: FFF42000, CID: B105900D, PID: 000BB925 ETM
Cortex-M4 identified.
J-Link>loadbin build/gcclight.bin 0x8000000
Halting CPU for downloading file.
Downloading file [build/gcclight.bin]...
Comparing flash   [100%] Done.
Verifying flash   [100%] Done.
J-Link: Flash download: Flash download skipped. Flash contents already match
O.K.
J-Link>rx 20
Reset delay: 20 ms
Reset type NORMAL: Resets core & peripherals via SYSRESETREQ & VECTRESET bit.
J-Link>g
qc
J-Link>qc
```