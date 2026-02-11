---
title: libmpfr错误
categories:
  - 边缘计算
mathjax: true
toc: true
date: 2018-12-01 15:30:16
tags:
-   K210
-   踩坑经验
---

我重装了`Ubuntu`之后去编译`k210`的程序发现编译不了了。蛋疼。



<!--more-->


# 错误输出

```sh
➜  build make
[  3%] Building C object lib/CMakeFiles/kendryte.dir/bsp/entry.c.obj
/opt/kendryte-toolchain-7.2.0/bin/../libexec/gcc/riscv64-unknown-elf/7.2.0/cc1: error while
loading shared libraries: libmpfr.so.4: cannot open shared object file: No such file or dire
ctory
lib/CMakeFiles/kendryte.dir/build.make:62: recipe for target 'lib/CMakeFiles/kendryte.dir/bs
p/entry.c.obj' failed
make[2]: *** [lib/CMakeFiles/kendryte.dir/bsp/entry.c.obj] Error 1
CMakeFiles/Makefile2:122: recipe for target 'lib/CMakeFiles/kendryte.dir/all' failed
make[1]: *** [lib/CMakeFiles/kendryte.dir/all] Error 2
Makefile:83: recipe for target 'all' failed
make: *** [all] Error 2
```

发现这个他是说这个lib找不到了，我继续看输出

```sh
➜  ~ ldd /opt/kendryte-toolchain-7.2.0/bin/../libexec/gcc/riscv64-unknown-elf/7.2.0/cc1
	linux-vdso.so.1 (0x00007ffd235ea000)
	libmpc.so.3 => /usr/lib/x86_64-linux-gnu/libmpc.so.3 (0x00007f6ea5ba5000)
	libmpfr.so.4 => not found
	libgmp.so.10 => /usr/lib/x86_64-linux-gnu/libgmp.so.10 (0x00007f6ea5924000)
	libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f6ea5720000)
	libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007f6ea5503000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f6ea5165000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f6ea4d74000)
	/lib64/ld-linux-x86-64.so.2 (0x00007f6ea5dbd000)
	libmpfr.so.6 => /usr/lib/x86_64-linux-gnu/libmpfr.so.6 (0x00007f6ea4af4000)
```

这个`libmpfr.so.4`的确找不到，我看看`ldconfig`

```sh
➜  ~ ldconfig -p | grep mpfr
	libmpfr.so.6 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libmpfr.so.6
	libmpfr.so (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libmpfr.so
```

我尝试安装这个`libmpfr.so.4`

```sh
➜  ~ sudo apt-get install libmpfr4   
[sudo] password for zqh: 
Reading package lists... Done
Building dependency tree       
Reading state information... Done
E: Unable to locate package libmpfr4
```

没有这个包。md


# 解决

我决定直接链接两个包完事
```sh
sudo ln -s /usr/lib/x86_64-linux-gnu/libmpfr.so.6 /usr/lib/x86_64-linux-gnu/libmpfr.so.4
➜  x86_64-linux-gnu l | grep libmpfr 
-rw-r--r--   1 root root  1.1M 2月   8  2018 libmpfr.a
lrwxrwxrwx   1 root root    16 2月   8  2018 libmpfr.so -> libmpfr.so.6.0.1
lrwxrwxrwx   1 root root    38 12月  1 15:50 libmpfr.so.4 -> /usr/lib/x86_64-linux-gnu/libmpfr.so.6
lrwxrwxrwx   1 root root    16 11月 28 10:19 libmpfr.so.6 -> libmpfr.so.6.0.1
-rw-r--r--   1 root root  512K 2月   8  2018 libmpfr.so.6.0.1
```

然后编译成功..