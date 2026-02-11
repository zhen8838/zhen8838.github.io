---
title: OrangPi开启spi-dev
date: 2018-07-31 21:08:02
categories: 
-   边缘计算
tags:
-   Linux
-   香橙派
-   踩坑经验
---


我想在OrangePi Zero中打开Spidev，编写一些应用层的驱动。

<!--more-->


# 配置armbian-config

我首先打开`armbian-config`，进入hardware使能`spidev`以及`spidev-add-cs1`并重启。

# 发现问题

接着我发现在/dev/目录下没有spidev生成。我查看了许多。
发现他们的系统在`armbian-config`中使能了，就会去加载对应的dtbo。
我做如下查看:

```sh
root@pi:~# vi /boot/dtb-4.17.11-sunxi64/allwinner/overlay/sun50i-h5-spi-spidev.dtbo 
Ð^Mþí^@^@^C^L^@^@^@8^@^@^B¬^@^@^@(^@^@^@^Q^@^@^@^P^@^@^@^@^@^@^@`^@^@^Bt^@^@^@^@
^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^A^@^@^@^@^@^@^@^C^@^@^@^T^@^@^@^@allwinner,sun50i
-h5^@^@^@^@^Afragment@0^@^@^@^@^@^C^@^@^@       ^@^@^@^K/aliases^@^@^@^@^@^@^@^A
__overlay__^@^@^@^@^C^@^@^@^Q^@^@^@^W/soc/spi@1c68000^@^@^@^@^@^@^@^C^@^@^@^Q^@^
@^@^\/soc/spi@1c69000^@^@^@^@^@^@^@^B^@^@^@^B^@^@^@^Afragment@1^@^@^@^@^@^C^@^@^
@^D^@^@^@!ÿÿÿÿ^@^@^@^A__overlay__^@^@^@^@^C^@^@^@^D^@^@^@(^@^@^@^A^@^@^@^C^@^@^@
^D^@^@^@7^@^@^@^@^@^@^@^Aspidev^@^@^@^@^@^C^@^@^@^G^@^@^@^@spidev^@^@^@^@^@^C^@^
@^@     ^@^@^@Cdisabled^@^@^@^@^@^@^@^C^@^@^@^D^@^@^@J^@^@^@^@^@^@^@^C^@^@^@^D^@
^@^@N^@^OB@^@^@^@^B^@^@^@^B^@^@^@^B^@^@^@^Afragment@2^@^@^@^@^@^C^@^@^@^D^@^@^@!
ÿÿÿÿ^@^@^@^A__overlay__^@^@^@^@^C^@^@^@^D^@^@^@(^@^@^@^A^@^@^@^C^@^@^@^D^@^@^@7^
@^@^@^@^@^@^@^Aspidev^@^@^@^@^@^C^@^@^@^G^@^@^@^@spidev^@^@^@^@^@^C^@^@^@
^@^@^@Cdisabled^@^@^@^@^@^@^@^C^@^@^@^D^@^@^@J^@^@^@^@^@^@^@^C^@^@^@^D^@^@^@N^@^
OB@^@^@^@^B^@^@^@^B^@^@^@^B^@^@^@^A__fixups__^@^@^@^@^@^C^@^@^@^U^@^@^@^W/fragme
nt@1:target:0^@^@^@^@^@^@^@^C^@^@^@^U^@^@^@^\/fragment@2:target:0^@^@^@^@^@^@^@^
B^@^@^@^B^@^@^@ compatible^@target-path^@spi0^@spi1^@target^@#address-cells^@#si
ze-cells^@status^@reg^@spi-max-frequency^@
```
这里发现这里spidev跟着的状态是disable的。然后我又查看了他的源码：
`/linux-4.17.y/arch/arm64/boot/dts/allwinner/overlay/sun50i-h5-spi-spidev.dts`
```sh
/dts-v1/;
/plugin/;

/ {
	compatible = "allwinner,sun50i-h5";

	fragment@0 {
		target-path = "/aliases";
		__overlay__ {
			spi0 = "/soc/spi@1c68000";
			spi1 = "/soc/spi@1c69000";
		};
	};

	fragment@1 {
		target = <&spi0>;
		__overlay__ {
			#address-cells = <1>;
			#size-cells = <0>;
			spidev {
				compatible = "spidev";
				status = "disabled";
				reg = <0>;
				spi-max-frequency = <1000000>;
			};
		};
	};

	fragment@2 {
		target = <&spi1>;
		__overlay__ {
			#address-cells = <1>;
			#size-cells = <0>;
			spidev {
				compatible = "spidev";
				status = "disabled";
				reg = <0>;
				spi-max-frequency = <1000000>;
			};
		};
	};
};
```

我就开始认为没有打开使能才会没有产生`spidev`。但是在我又询问了论坛的人后才知道。

# 修改armbianEnv.txt

他们的[手册](https://docs.armbian.com/Hardware_Allwinner_overlays/)中写到：

`
param_* - overlay parameters
`

需要添加参数。。

我看了一会知道了他们的启动运行流程：

启动->读取`armbianEnv.txt-`>加载`overlay`参数对应的dtbo->继续读取`armbianEnv.txt`的参数项->根据参数来运行`sun50i-h5-fixup.scr`->将对应的外设使能

所以现在需要添加：这个两个参数使能spidev。

`
param_spidev_spi_bus=1
param_spidev_spi_cs=1
`

# 完成

最终可以看到：
```sh
root@pi:~# ls /dev/s
shm/       snd/       spidev1.1  stderr     stdin      stdout 
```

# 测试spi

我们使用linux中自带的`spidev_test.c`进行测试。
进入`/linux-4.17.y/tools/spi`中交叉编译：

```sh
➜  ~ cd /home/zqh/sources/linux-mainline/linux-4.17.y/tools/spi 
➜  spi git:(84d52eb0) ✗ aarch64-linux-gnu-gcc -o spidev_test spidev_test.c -lpthread -static
➜  spi git:(84d52eb0) ✗ ls
include  Makefile  spidev_fdx.c  spidev_test  spidev_test.c  spidev_test-in.o
```

将`spidev_test`拷贝入开发板中，并且短接`MISO`和`MOSI`。运行测试程序（记得加-v选项）。

```sh
root@pi:~# ./spidev_test -D /dev/spidev1.0 -v
spi mode: 0x0
bits per word: 8
max speed: 500000 Hz (500 KHz)
TX | FF FF FF FF FF FF 40 00 00 00 00 95 FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF F0 0D  | ......@....�..................�.
RX | FF FF FF FF FF FF 40 00 00 00 00 95 FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF F0 0D  | ......@....�..................�.
root@pi:~# ./spidev_test -D /dev/spidev1.0 -v -p helloworld
spi mode: 0x0
bits per word: 8
max speed: 500000 Hz (500 KHz)
TX | 68 65 6C 6C 6F 77 6F 72 6C 64 __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __  | helloworld
RX | 68 65 6C 6C 6F 77 6F 72 6C 64 __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __  | helloworld
```

