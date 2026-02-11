---
title: OrangePI蓝牙：搜索设备
date: 2018-06-19 20:35:09
tags: 
-   Linux
-   树莓派
-   蓝牙
categories: 
-   边缘计算 
---

上一篇[文章](https://zhen8838.github.io/2018/05/30/Linuxblue1/)讲述了我在OrangePi中开启蓝牙的过程，这一章来讲述我如何对蓝牙进行编程操作。

<!--more-->


## 开发板中使用bluez
1.  安装bluez
    我的OrangePi的板子上是bluez5.43版本,为了开发bluez需要安装一些必要的头文件.
    ```sh
    sudo apt-get install libbluetooth-dev
    ls /usr/include/bluetooth/
    bluetooth.h  bnep.h  cmtp.h  hci.h  hci_lib.h  hidp.h  l2cap.h  rfcomm.h  sco.h  sdp.h  sdp_lib.h
    ```
    安装后,我们就可以在通过添加一些头文件去调用bluez的api做一些事情了.
    
    ```c
    #include <bluetooth/bluetooth.h>
    #include<Bluetooth/hci.h>
    #include<Bluetooth/hci_lib.h>
    ```
    

2.  配置编译
    首先我使用外国老哥曾经写的文档中的代码做测试^[1]^(这个文档非常nice),这个一个扫描周围蓝牙设备的小程序.
    
    ```c
    #include <stdio.h>
    #include <stdlib.h>
    #include <unistd.h>
    #include <sys/socket.h>
    #include <bluetooth/bluetooth.h>
    #include <bluetooth/hci.h>
    #include <bluetooth/hci_lib.h>

    int main(int argc, char **argv)
    {
    inquiry_info *ii = NULL;
    int max_rsp, num_rsp;
    int dev_id, sock, len, flags;
    int i;
    char addr[19] = { 0 };
    char name[248] = { 0 };

    dev_id = hci_get_route(NULL);
    sock = hci_open_dev( dev_id );
    if (dev_id < 0 || sock < 0) {
        perror("opening socket");
        exit(1);
    }

    len  = 8;
    max_rsp = 255;
    flags = IREQ_CACHE_FLUSH;
    ii = (inquiry_info*)malloc(max_rsp * sizeof(inquiry_info));
    
    num_rsp = hci_inquiry(dev_id, len, max_rsp, NULL, &ii, flags);
    if( num_rsp < 0 ) perror("hci_inquiry");

    for (i = 0; i < num_rsp; i++) {
        ba2str(&(ii+i)->bdaddr, addr);
        memset(name, 0, sizeof(name));
        if (hci_read_remote_name(sock, &(ii+i)->bdaddr, sizeof(name), 
            name, 0) < 0)
        strcpy(name, "[unknown]");
        printf("%s  %s\n", addr, name);
    }

    free( ii );
    close( sock );
    return 0;
    }
    ```
    

    使用:
    ```sh
    root@H5:~# vi test.c
    root@H5:~# gcc -o simplescan test.c -lbluetooth
    root@H5:~# ./simplescan
    CC:29:F5:79:14:21  iPhone
    ``` 
## 交叉编译bluez
1.  下载bluez

    首先去[官方网址](http://www.bluez.org/release-of-bluez-5-43/)下载bluez,这里我还是选择使用5.43版本.
2.  解压配置
    这里参考网络文章^[2]^.
    将bluez5.43解压出来.并新建一个文件夹blue,用于存放安装的文件.
    ```sh
    mkdir blue
    cd bluez-5.43
    CC=/home/zqh/GccOrangPi/bin/aarch64-linux-gnu-gcc #指定交叉编译器
    ./configure --host=aarch64-linux-gnu --prefix=/home/zqh/Program/orangepi/blue  --disable-obex --enable-library
    ```
    这里如果配置检查出现错误那么参考这个文章^[3]^.
    这里有一个比较恶心的错误:
    ```sh
    configure: error: readline header files are required
    ```
    我安装了readline-dev还是会出现这个错误,这里就需要把这个头文件复制到交叉编译器的头文件目录下.
    ```sh
    sudo cp -r /usr/include/readline /home/zqh/GccOrangPi/aarch64-linux-gnu/libc/usr/include
    ```
3.  编译安装
    配置完成后开始编译以及安装
    ```sh
    make
    sudo make install
    cd /home/zqh/Program/orangepi/blue
    ```
    之后就可以把blue目录下的文件移植到开发板上.
    将/bin下所有文件，放到开发板/usr/bin
    include所有文件，放到开发板/usr/include
    lib所有文件，放到开发板/usr/lib
    sbin所有文件，放到开发板/usr/sbin
    当然bluez安装还是会把许多的文件安装在默认目录,比如/etc下,我先尝试看看会不会影响开发.
## 偷懒方法
其实orangepi可以自己apt-get到libbluetooth-dev,那么我们可以直接将板子上的动态链接库拉过来使用即可~**(注意编译动态链接库的编译器版本不同可能会出现错误,那么还是要全部手动编译一波,再安装进去)**
不过要是大家有什么更简单的方法,可以方便的交叉编译,可以告诉我~

## 使用bluez
1.  配置路径

    我这里使用的是cmake配置工程:
    ```sh
    # 设置工程名称
    project (TEST)
    # 设置可执行文件名称
    set(MY_TARGET t1)
    # 需要链接的动态链接库
    set(EXTRA_LIBS libpthread.so libbluetooth.so)
    # CMAKE最小版本
    cmake_minimum_required (VERSION 2.6)

    # 设置目标平台系统
    set(CMAKE_SYSTEM_NAME Linux)

    # 设置交叉编译库路径
    set(CMAKE_FIND_ROOT_PATH /home/zqh/GccOrangPi/)
    # set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)#只在交叉编译库路径中寻找
    # set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

    # 设置交叉编译器
    set(CMAKE_C_COMPILER /home/zqh/GccOrangPi/bin/aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER /home/zqh/GccOrangPi/bin/aarch64-linux-gnu-g++)
    set(CMAKE_C_EXTENSIONS "-lbluetooth -pipe -g -Wall -W -fPIE")
    set(CMAKE_CXX_EXTENSIONS "-lbluetooth -pipe -g -Wall -W -fPIE")
    #设置执行文件输出目录
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
    #设置库输出路径
    set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

    message("++++++++++++++Start Build+++++++++++++++++")

    # 添加头文件目录
    include_directories(${PROJECT_SOURCE_DIR}/usr/inc)
    include_directories("/home/zqh/Program/orangepi/blue/include")

    # 添加源文件目录
    aux_source_directory(${PROJECT_SOURCE_DIR}/usr/src USRSRC)


    # 添加子目录 子目录里面放一些别的编译好的模块
    #ADD_SUBDIRECTORY(src)

    # 链接库搜索路径
    link_directories("/home/zqh/GccOrangPi/" "/home/zqh/Program/orangepi/blue/lib")

    # 添加动态库
    link_libraries(${EXTRA_LIBS})

    # 添加可执行文件（可执行文件名 [配置] 源文件）
    add_executable(${MY_TARGET} ${USRSRC})

    # 执行文件链接属性
    TARGET_LINK_LIBRARIES(${MY_TARGET} ${EXTRA_LIBS})
    ```
1.  编译运行
    这里使用之前的代码.很简单:
    ```sh
    cd build
    cmake ..
    make
    ```
    然后发送可执行文件到开发板上,运行:
    ```sh
    root@H5:~# ./t1
    CC:29:F5:79:14:21  iPhone
    ```

## 参考资料
[1].http://people.csail.mit.edu/albert/bluez-intro/c404.html
[2].http://www.forwhat.cn/post-436.html
[3].https://blog.csdn.net/twy76/article/details/23851587


