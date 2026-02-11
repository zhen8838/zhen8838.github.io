---
title: cpp模板编译踩坑记
categories:
  - 编程语言
date: 2019-04-01 14:30:22
tags:
- 踩坑经验
- CPP
---

今天给我的东西写了个可变长模板log类,然后编译的时候踩了大坑.😤

<!--more-->

## 出现很多很多错误...

我开始的时候想将`#include`放到`namespace`的定义中去，然后直接炸....

出现很多类似的错误：
```sh
In file included from /opt/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/arm-linux-gnueabihf/include/c++/4.9.3/string:52:0,
                 from /opt/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/arm-linux-gnueabihf/include/c++/4.9.3/bits/locale_classes.h:40,
                 from /opt/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/arm-linux-gnueabihf/include/c++/4.9.3/bits/ios_base.h:41,
                 from /opt/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/arm-linux-gnueabihf/include/c++/4.9.3/ios:42,
                 from /opt/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/arm-linux-gnueabihf/include/c++/4.9.3/ostream:38,
                 from /opt/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/arm-linux-gnueabihf/include/c++/4.9.3/iostream:39,
                 from /home/zqh/Documents/raspi_blue/utils/inc/loger.h:13,
                 from /home/zqh/Documents/raspi_blue/utils/src/record.cpp:2:
/opt/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/arm-linux-gnueabihf/include/c++/4.9.3/bits/basic_string.h: In function 'long int Loger::std::stol(const string&, Loger::std::size_t*, int)':
```

## 猜测:

我怀疑是将`#include <iostream>`放在`namespace`中,就会导致`std`和别一些东西冲突..

## 出现多重定义的错误

我把定义放到外面之后编译出现
```sh
CMakeFiles/myrecord.dir/utils/src/sock.cpp.o: In function `Loger::show_list()':
sock.cpp:(.text+0x0): multiple definition of `Loger::show_list()'
CMakeFiles/myrecord.dir/utils/src/record.cpp.o:record.cpp:(.text+0x0): first defined here
CMakeFiles/myrecord.dir/xfly/src/quickasr.cpp.o: In function `Loger::show_list()':
quickasr.cpp:(.text+0x0): multiple definition of `Loger::show_list()'
CMakeFiles/myrecord.dir/utils/src/record.cpp.o:record.cpp:(.text+0x0): first defined here
CMakeFiles/myrecord.dir/usr/src/main.cpp.o: In function `Loger::show_list()':
main.cpp:(.text+0x1cc): multiple definition of `Loger::show_list()'
CMakeFiles/myrecord.dir/utils/src/record.cpp.o:record.cpp:(.text+0x0): first defined here
collect2: error: ld returned 1 exit status
CMakeFiles/myrecord.dir/build.make:172: recipe for target '../bin/myrecord' failed
make[2]: *** [../bin/myrecord] Error 1
CMakeFiles/Makefile2:67: recipe for target 'CMakeFiles/myrecord.dir/all' failed
make[1]: *** [CMakeFiles/myrecord.dir/all] Error 2
Makefile:83: recipe for target 'all' failed
make: *** [all] Error 2
```

## 猜测

这个我找了半天没找到相关描述..但是我在那个函数前面加了`inline`就莫名的好了 😂



最终代码:

```cpp
/*
 * @Author: Zheng Qihang
 * @Date: 2019-03-11 12:47:23
 * @Last Modified by: Zheng Qihang
 * @Last Modified time: 2019-04-01 15:01:21
 */
#ifndef _LOGER_
#define _LOGER_

#include <cstdlib>
#include <iostream>

namespace Loger {

/* clang-format off */
#define NONE                 "\e[0m"
#define BLACK                "\e[0;30m"
#define L_BLACK              "\e[1;30m"
#define RED                  "\e[0;31m"
#define L_RED                "\e[1;31m"
#define GREEN                "\e[0;32m"
#define L_GREEN              "\e[1;32m"
#define BROWN                "\e[0;33m"
#define YELLOW               "\e[1;33m"
#define BLUE                 "\e[0;34m"
#define L_BLUE               "\e[1;34m"
#define PURPLE               "\e[0;35m"
#define L_PURPLE             "\e[1;35m"
#define CYAN                 "\e[0;36m"
#define L_CYAN               "\e[1;36m"
#define GRAY                 "\e[0;37m"
#define WHITE                "\e[1;37m"

#define BOLD                 "\e[1m"
#define UNDERLINE            "\e[4m"
#define BLINK                "\e[5m"
#define REVERSE              "\e[7m"
#define HIDE                 "\e[8m"
#define CLEAR                "\e[2J"
#define CLRLINE              "\r\e[K"

#define ERROR_C              "\e[0;31m[  ERR   ] \e[0m"
#define OK_C                 "\e[0;32m[   OK   ] \e[0m"
#define DEBUG_C              "\e[0;34m[ DEBUG  ] \e[0m"
#define INFO_C               "\e[0;35m[  INFO  ] \e[0m"
#define WAIT_C               "\e[0;33m[  WAIT  ] \e[0m"
/* clang-format on */

/**
 * @brief 处理边界条件
 *
 */
inline void show_list() { std::cout << std::endl; }

/**
 * @brief 处理参数列表
 *
 * @tparam T
 * @tparam Args
 * @param var 参数1
 * @param args 变长参数
 */
template<typename T, typename... Args> void show_list(const T &var, const Args &... args) {
    std::cout << var;
    show_list(args...);
}

template<typename... Args> void info(const Args &... args) {
    std::cout << INFO_C;
    show_list(args...);
}

template<typename... Args> void err(const Args &... args) {
    std::cout << ERROR_C;
    show_list(args...);
}

template<typename... Args> void errexit(const Args &... args) {
    std::cout << ERROR_C;
    show_list(args...);
    exit(-1);
}
} // namespace Loger

#endif
```


