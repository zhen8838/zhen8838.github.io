---
title: k210-tool-chains mac m1编译
mathjax: true
toc: true
date: 2021-07-16 22:11:46
categories:
  - 边缘计算
tags:
-   K210
---

关于如何在apple m1 上编译k210 toolchains

<!--more-->

#  安装依赖

注意这里会遇到一个包解压不了的问题,那是因为作者把压缩包名写错了.. 装完之后把路径都export一下.
```sh
brew install gawk gnu-sed gmp mpfr libmpc isl zlib expat
```

还得装wget
```sh
brew install wget
```
然后又会碰到打不开文件的错误
```sh
==> Pouring libunistring-0.9.10.arm64_big_sur.bottle.tar.gz
tar: Error opening archive: Failed to open '/Users/lisa/Library/Caches/Homebrew/downloads/b666b30f757d05bf040dd947beab7c38d56a04b6a47552b726cf799a06cd8cf9--libunistring-0.9.10.arm64_big_sur.bottle.tar.gz'
Error: Failure while executing; `tar --extract --no-same-owner --file /Users/lisa/Library/Caches/Homebrew/downloads/b666b30f757d05bf040dd947beab7c38d56a04b6a47552b726cf799a06cd8cf9--libunistring-0.9.10.arm64_big_sur.bottle.tar.gz --directory /private/tmp/d20210717-75579-1fykoqn` exited with 1. Here's the output:
tar: Error opening archive: Failed to open '/Users/lisa/Library/Caches/Homebrew/downloads/b666b30f757d05bf040dd947beab7c38d56a04b6a47552b726cf799a06cd8cf9--libunistring-0.9.10.arm64_big_sur.bottle.tar.gz'
```
把那个文件名手动进行修改
```sh
mv /Users/lisa/Library/Caches/Homebrew/downloads/b68429257038e80dad7f5e906f26422d73b1a124cafb3a4f6d4d8aad2a96419c--libunistring--0.9.10.arm64_big_sur.bottle.tar.gz  /Users/lisa/Library/Caches/Homebrew/downloads/b666b30f757d05bf040dd947beab7c38d56a04b6a47552b726cf799a06cd8cf9--libunistring-0.9.10.arm64_big_sur.bottle.tar.gz
```

# 下载代码并编译

```
git clone --recursive https://github.com/kendryte/kendryte-gnu-toolchain
cd kendryte-gnu-toolchain
./configure --prefix=/usr/local/opt/kendryte-toolchain --with-cmodel=medany --with-arch=rv64imafc --with-abi=lp64f
sudo make -j
```

## readline编译问题

发现是readline编译出现错误
```sh
configure: creating ./config.status
config.status: creating Makefile
config.status: creating po/Makefile.in
config.status: creating config.h
config.status: executing depfiles commands
config.status: executing libtool commands
config.status: executing default-1 commands
config.status: executing default commands
make[1]: *** [all] Error 2
make: *** [stamps/build-binutils-newlib] Error 2
```
出错具体信息如下
```sh
/Users/lisa/Documents/kendryte-gnu-toolchain/riscv-binutils-gdb/readline/rltty.c:83:7: error: implicit declaration of function 'ioctl' is invalid in C99 [-Werror,-Wimplicit-function-declaration]
  if (ioctl (tty, TIOCGWINSZ, &w) == 0)
      ^
/Users/lisa/Documents/kendryte-gnu-toolchain/riscv-binutils-gdb/readline/rltty.c:720:3: error: implicit declaration of function 'ioctl' is invalid in C99 [-Werror,-Wimplicit-function-declaration]
  ioctl (fildes, TIOCSTART, 0);
  ^
/Users/lisa/Documents/kendryte-gnu-toolchain/riscv-binutils-gdb/readline/rltty.c:759:3: error: implicit declaration of function 'ioctl' is invalid in C99 [-Werror,-Wimplicit-function-declaration]
  ioctl (fildes, TIOCSTOP, 0);
```

一番查看发现虽然外面的config都设置了`-disable-werror`,但是readline里面的编译选项并没有继承.所以需要单独配置一下. 去`kendryte-gnu-toolchain/riscv-binutils-gdb/readline/Makefile.in`中把79行修改为如下,避免这个waring即可.
```make
LOCAL_DEFS = -Wno-implicit-function-declaration @LOCAL_DEFS@
```

## backend错误

链接的时候报错,这个问题好像是gcc不支持arm64的mac系统,因为那时候都还没出m1..
```sh
clang: warning: argument unused during compilation: '-no-pie' [-Wunused-command-line-argument]
Undefined symbols for architecture arm64:
  "_host_hooks", referenced from:
      c_common_no_more_pch() in c-pch.o
      toplev::main(int, char**) in libbackend.a(toplev.o)
      gt_pch_save(__sFILE*) in libbackend.a(ggc-common.o)
      gt_pch_restore(__sFILE*) in libbackend.a(ggc-common.o)
ld: symbol(s) not found for architecture arm64
clang: error: linker command failed with exit code 1 (use -v to see invocation)
```

参考这个[方案](https://github.com/riscv/riscv-gnu-toolchain/issues/800#issuecomment-808722775),在`riscv-gcc/gcc/config/host-darwin.c`中添加两行代码,以根据当前host重新生成.


# 完成

编译时间大约在4个小时左右,最后默认是安装到了opt目录下:

![](k210-tool-chains/m1-tool-chain.png)

我把编译好的包放在[云盘](https://drive.google.com/file/d/1fmlhRItYnrbEGe9OqK6C28O5UWQpEWmu/view?usp=sharing)中了,需要的可以下载.