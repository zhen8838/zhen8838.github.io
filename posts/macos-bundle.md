---
title: macos中bundle的使用
mathjax: true
toc: true
categories:
  - 操作系统
date: 2024-03-13 22:06:19
tags:
  - cmake
  - CPP
---


研究一下在macos中如何编译bundle文件并动态加载并运行.

<!--more-->

使用gcc编译macos程序的时候, 可以通过`-bundle`选项来指示, 编译出来就是一个包含资源和可执行代码的包. 使用 -bundle 选项时, 还需要指定一个入口点, 通常是 main 函数. 它的好处是使得开发者能够创建动态加载的代码模块, 这些模块可以包含可执行代码和资源, 并且可以被其他应用程序在运行时加载和使用.



## 简单例子

来自苹果开源库的[dyld测试例子](https://opensource.apple.com/source/dyld/dyld-852.2/unit-tests/test-cases/)

bundle.c
```cpp
#include <stdbool.h>
#include <stdio.h>

// test to see if bss section is properly expanded 

static int mydata[1000000];

bool checkdata()
{
  printf("check data!\n");
	return ( mydata[500000] == 0 );
}

```

main.c
```cpp
#include <Availability.h>
#include <mach-o/dyld.h>
#include <stdbool.h>
#include <stdio.h>

typedef bool (*CheckFunc)();

int main() {
// these APIs are only available on Mac OS X - not iPhone OS
#if __MAC_OS_X_VERSION_MIN_REQUIRED
    NSObjectFileImage ofi;
    if (NSCreateObjectFileImageFromFile("test.bundle", &ofi) !=
        NSObjectFileImageSuccess) {
        // FAIL("NSCreateObjectFileImageFromFile failed");
        return 1;
    }

    NSModule mod = NSLinkModule(ofi, "test.bundle", NSLINKMODULE_OPTION_NONE);
    if (mod == NULL) {
        // FAIL("NSLinkModule failed");
        return 1;
    }

    NSSymbol sym = NSLookupSymbolInModule(mod, "_checkdata");
    if (sym == NULL) {
        // FAIL("NSLookupSymbolInModule failed");
        return 1;
    }

    CheckFunc func = NSAddressOfSymbol(sym);
    if (!func()) {
        // FAIL("NSAddressOfSymbol failed");
        return 1;
    }

    if (!NSUnLinkModule(mod, NSUNLINKMODULE_OPTION_NONE)) {
        // FAIL("NSUnLinkModule failed");
        return 1;
    }

    if (!NSDestroyObjectFileImage(ofi)) {
        // FAIL("NSDestroyObjectFileImage failed");
        return 1;
    }
#endif
    printf("funck\n");
    // PASS("bundle-basic");
    return 0;
}
```

执行
```sh
clang bundle.c -bundle -o test.bundle
clang main.c
./a.out
```

## 使用cmake


注意cmake的`MACOS_BUNDLE`和链接选项的bundle并不是一个东西. 所以用那个选项是无法得到正确的结果的.
```sh
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# 设置项目名称和版本
project(test.bundle VERSION 1.0)

# 指定 C++ 标准
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加一个 macOS bundle 可执行文件
add_executable(test.bundle bundle.c)
target_link_options(test.bundle PUBLIC -bundle)
# 如果有需要，可以链接库
# target_link_libraries(test.bundle SomeLibrary)

# 包含 CMake 的 macOS 应用程序 bundle 模块
# include(CMakeOSXBundleInfo)
```

## 一些问题

### 1. link问题

我在正式的项目本来是使用`nostdlib static`来编译的, 然后现在链接选项添加了`-bundle`后就遇到这个问题:

```sh
ld: warning: ignoring -e, not used for output type
0  0x1041d2f2c  __assert_rtn + 72
1  0x1041a7984  ld::AtomFileConsolidator::addAtomFile(ld::AtomFile const*, ld::AtomFile const*, bool) + 4952
2  0x1041b34e4  ld::AtomFileConsolidator::addAtomFile(ld::AtomFile const*) + 148
3  0x1041d1430  ld::pass::stubs(ld::Options const&, ld::AtomFileConsolidator&) + 1464
4  0x1041bad60  ld::AtomFileConsolidator::resolve() + 12744
5  0x104142b40  main + 9308
ld: Assertion failed: (slot < _sideTableBuffer.size()), function addAtom, file AtomFileConsolidator.cpp, line 2278.
clang: error: linker command failed with exit code 1 (use -v to see invocation)
```

搜索之后发现是linker升级的原因, 可以通过`-ld_classic`来避免.

### 2. 全局变量冲突的问题

我现在是同时添加多个bundle的模块, 然后调用每个模块中的函数, 目前不同的模板都是存在相同名字的全局变量, 这导致在执行的过程中同一个程序的不同的函数中取到的全局变量的地址是不同的, 从而出现问题. 我改成dylib的形式, 然后使用dlopen去加载函数, 也会出现同样的问题, 所以我怀疑这个问题就是和mac os的底层实现有关. 但是又不能一次只加载一个代码, 这样的话每个kernel启动前都需要load一次可执行文件, 性能就炸了.

尝试使用类似下面这种静态函数的方式来规避全局变量, 发现也不行:
```cpp
nncase_runtime_cpu_mt_t *g_cpu_mt(nncase_runtime_cpu_mt_t *from) {
    static nncase_runtime_cpu_mt_t *g_mt;
    if (from != NULL) {
        g_mt = from;
    }
    return g_mt;
}
```

目前的总结:

1. 非extern
   1. 加static, 两个cpp中的全局变量不是同一个.
   2. 不加static, 两个cpp中的会自己定义两次全局变量, 然后编译的过程中报错多重定义.
2. extern
   1. 不论是在哪个cpp里面实现, 加载多个程序之后都会出现冲突.
3. 同名函数中静态变量, 同样他也会被优化成类似全局变量的形式然后出现混乱.
4. 给全局变量进行编号也无效, 实际上可能是因为实例化出来的c++函数在全局符号表中发生了冲突, 导致kernel0中执行了kernel7的tensor copy.