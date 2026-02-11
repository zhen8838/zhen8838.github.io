---
title: 树莓派NB-IOT使用
date: 2018-11-04 14:13:17
categories:
-   边缘计算
tags:
-   树莓派
-   NB-IOT
---



我的同学给了我一个nb-iot的小[开发板](https://item.taobao.com/item.htm?spm=a1z10.1-c-s.w4004-12399990372.10.49d5520d7gojSP&id=576818997326)
,让我在树莓派上移植一个nb-iot的程序。

<!--more-->


# 1.配置交叉编译

1.  首先给树莓派烧系统。
    看看编译器版本是什么：

    ```sh
    pi@raspberrypi:~$ gcc -v
    gcc version 6.3.0 20170516 (Raspbian 6.3.0-18+rpi1+deb9u1)
    ```

2.  下载交叉编译器。

    百度一搜，去[官方](http://releases.linaro.org/components/toolchain/binaries/latest-5/arm-linux-gnueabihf/)下载`	gcc-linaro-5.5.0-2017.10-x86_64_arm-linux-gnueabihf.tar.xz`

3.  解压安装

    **注意：** 这里我`mv`是为了对这个文件夹**改名字**
    ```sh
    ➜  Downloads sudo tar -xvf gcc-linaro-5.5.0-2017.10-x86_64_arm-linux-gnueabihf.tar.xz -C /opt
    ➜  Downloads cd /opt/ 
    ➜  /opt sudo mv gcc-linaro-5.5.0-2017.10-x86_64_arm-linux-gnueabihf gccRaspPI
    ➜  /opt cd gccRaspPI/bin 
    ```
4.  设置环境变量

    **注意：** 我修改`zshrc`是因为我使用的是`zsh`，一般情况下使用的是`bash`
    ```sh
    ➜  bin realpath .
    /opt/gccRaspPI/bin
    ➜  bin sudo vi ~/.zshrc
    ```
    最后一行添加`export  PATH="$PATH:/opt/gccRaspPI/bin"`
    ```sh
    ➜  bin source ~/.zshrc           
    ➜  bin arm-linux-gnueabihf-gcc -v
    gcc version 5.5.0 (Linaro GCC 5.5-2017.10) 
    ```
    设置成功。

# 2.编译程序

先写个程序：

```c
#include <stdio.h>
int main(int argc, char const *argv[]) {
    printf("hello world\n");
    return 0;
}
```



编译：
```sh
➜  nb-proj arm-linux-gnueabihf-gcc main.c -o test
```

# 传输程序

利用`sftp`发送文件后：
```sh
pi@raspberrypi:~ $ ./test 
hello world
```
执行成功。