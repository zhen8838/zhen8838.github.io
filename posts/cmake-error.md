---
title: cmake踩坑&爬坑
mathjax: true
toc: true
categories:
  - 编程语言
date: 2021-04-25 00:55:13
tags:
- cmake
- CPP
- 踩坑经验
---

最近实习了，每天都要和cmake打交道，记录一些时常要用的东西和遇到的问题。


<!--more-->


## cmake 最小模板

```
cmake_minimum_required(VERSION 3.9)
project(xxx)
 
 
#设定编译参数
set(CMAKE_CXX_STANDARD 11)
 
#设定源码列表.cpp
aux_source_directory(<dir> <variable>)
#比如:aux_source_directory(${CMAKE_SOURCE_DIR} DIR)  

 
#设定头文件路径
include_directories(../include/)
#include_directories("路径1"  “路径2”...)
 
 
#设定链接库的路径（一般使用第三方非系统目录下的库）
link_directories(../build/)
#link_directories("路径1"  “路径2”...)
 
 
#添加子目录,作用相当于进入子目录里面，展开子目录的CMakeLists.txt
#同时执行，子目录中的CMakeLists.txt一般是编译成一个库，作为一个模块
#在父目录中可以直接引用子目录生成的库
#add_subdirectory(math)
 
 
#生成动/静态库
#add_library(动/静态链接库名称  SHARED/STATIC(可选，默认STATIC)  源码列表)
#可以单独生成多个模块
 
 
#生成可执行文件
add_executable(myLevealDB   ${SOURCE_FILES} )
#比如：add_executable(hello_world    ${SOURCE_FILES})
 
 
target_link_libraries(xxx pthred glog)#就是g++ 编译选项中-l后的内容，不要有多余空格
 
# ADD_CUSTOM_COMMAND( #执行shell命令
#           TARGET myLevelDB 
#           POST_BUILD #在目标文件myLevelDBbuild之后，执行下面的拷贝命令，还可以选择PRE_BUILD命令将
# 会在其他依赖项执行前执行  PRE_LINK命令将会在其他依赖项执行完后执行  POST_BUILD命令将会在目# 标构建完后执行。
#           COMMAND cp ./myLevelDB  ../
# )
```

## add_subdirectory

用`add_subdirectory`可以很方便的控制是否编译子目录，比如我们将测试代码放到`test`中，在根目录的cmakelist里面选择是否编译单元测试文件。


## 利用通配符找到源文件


```
file(GLOB SOURCE_FILES ${CMAKE_SOURCE_DIR}/test_*.cpp)
add_executable(cpp_test ${SOURCE_FILES})
```

## macro使用


| 变量  | 说明                                                 |
| ----- | ---------------------------------------------------- |
| ARGV# | # 是一个下标，0 指向第一个参数，累加                 |
| ARGV  | 所有的定义时要求传入的参数                           |
| ARGN  | 定义时要求传入的参数以外的参数列表                   |
| ARGC  | 传入的实际参数的个数，也就是调用函数是传入的参数个数 |


当要收集额外参数的时候，不能直接在宏的入口处写参数，因为那里的参数是必须传入的，只能直接传入然后用if进行解析。
```
macro(add_code name)
    if(${ARGC} GREATER 2)
        set(gen_name ${ARGV1})
        set(gen_variable ${ARGV2})
    else()
        set(gen_name ${name})
        set(gen_variable "")
    endif()
    add_custom_target(gen_${name} 
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gen_all -g halide_${name} -n halide_${gen_name} -o ${CMAKE_CURRENT_SOURCE_DIR}/kernels -e c_header,assembly,schedule,stmt_html target=host-no_asserts-no_bounds_query "${gen_variable}")
    add_dependencies(gen_${name} gen_all)
endmacro()
```

## 空格被转义

我想输入一个参数为`"kernel_width=3 kernel_height=3"`然后执行命令，但是我发现执行命令的时候变成了这样：
```
kernel_width=3\ kernel_height=3
```

这个是因为cmake为了区别于他多个变量赋值的问题，我们可以用两个方式解决
```
set(testfiles "test1" "test2") \\ 1. 直接多个变量赋值
set(testfiles "kernel_width=3;kernel_height=3") \\ 2. 使用分号隔开
```

## 检查正确性

有时候我们可能需要检查cmake的内部变量是不是有错误，要加message比较麻烦，我们可以这样：
```
make VERBOSE=1 
```

## 为某个文件添加依赖，自定义生成


使用custom command，然后把生成的src添加到一个新的的target中，然后目标target依赖这个target。
要注意，我们生成的源文件不能直接加到目标target中，因为一开始没有生成代码，会导致cmake config出错，得找找方法去解决一下。
```cmake
add_custom_command(
  OUTPUT testData.cpp
  COMMAND reswrap 
  ARGS    testData.src > testData.cpp
  DEPENDS testData.src 
)
add_custom_target(evaluator_k510.update SOURCES ${KERNEL_SRCS})
add_dependencies(evaluator_k510 evaluator_k510.update)
```

如果是在没有依赖，那么可以自己加个`dummy.cpp`然后强制把需要生成的文件给添加上


## cmake默认target输出位置

在编译时通过`add_executable()`生成的可执行文件，以及利用`add_library( SHARED )`的文件，默认输出到`CMAKE_RUNTIME_OUTPUT_DIRECTORY`，通常是`build/bin`目录。


## cmake导出pseudo target，并进行调用

我首先用`add_custom_command`生成了一个lib，但是这个lib不是cmake构建的，所以不能直接被cmake调用，并且如果他是一个lib，必须要先写个`add_library`制作一个伪目标，然后再把他真实的lib添加上去，然后才能正常使用这个target。 
```
add_library(hkg_${os_name}_runtime INTERFACE)
target_link_libraries(hkg_${os_name}_runtime INTERFACE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/halide_runtime_${os_name}.${suffix} -ldl)
add_library(hkg::${os_name}_runtime ALIAS hkg_${os_name}_runtime)
```


更加复杂的例子在这里`https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#interface-libraries`


## cmake安装target设置其名字

当我们设置自定义目标的时候，用别名`ALIAS`可以指定在整个构建过程中的target的使用名字，但是如果这个target需要导出（也就是被第三方库使用时），他的名字还得使用`set_target_properties`设置一下。
```cmake
add_library(Halide::Generator ALIAS Halide_Generator)
set_target_properties(Halide_Generator PROPERTIES EXPORT_NAME Generator)
```

## cmake安装目录路径

cmake真的处处是坑，如果想install一个目录下的所有东西，那么必须这样写:
```cmake
    install(DIRECTORY include/
            DESTINATION include)
```
如果少了一个`/`安装后就会出现`include/include`的情况

## cmake函数返回值

cmake里面是没有函数返回值的，所以必须要在函数中设置变量的值，同时指定他的scope，同时还需要注意设置变量的写法。或者说传递变量的名字然后用${name}的方法完成了c中类似指针的功能🤣。

```cmake
FUNCTION(MY_FUNC_WITH_RET ret)
    # The following line works by accident if the name of variable in the parent
    # is the same as in the function
    SET(ret "in function" PARENT_SCOPE)
    # This is the correct way to get the variable name passed to the function
    SET(${ret} "in function" PARENT_SCOPE)
ENDFUNCTION()

SET(ret "before function")
MY_FUNC_WITH_RET(ret)
MESSAGE("output from function = ${ret}")
```

## cmake写入多行include

还是利用configure_file，不过需要自己手动处理比较多的东西

简单测试，在.h.in文件中添加`@linux_include_list@`块之后执行cmake，就可以得到正确的include了。
```
cmake_minimum_required(VERSION 3.20)
project(fuck)
set(base_name "conv2d")
set(os_name "linux")
set(header_list "")
list(APPEND feature_list 
    "avx512" 
    "avx2" 
    "sse41" 
    "bare")
list(APPEND full_feature_list 
    "-sse41-avx-f16c-fma-avx2-avx512" 
    "-avx-avx2-f16c-fma-sse41"
    "-avx-f16c-sse41"
    "")

foreach(feature full_feature IN ZIP_LISTS feature_list full_feature_list)
    message("${feature} ${full_feature}")
    list(APPEND header_list "${base_name}-${os_name}-${feature}.h")
endforeach()

set(linux_include_list "")
foreach(header IN LISTS header_list)
    set(linux_include_list "${linux_include_list}\r\n#include \"${header}\"")
endforeach()


# set(linux_include_list ${header_list})
set(windows_include_list ${header_list})
set(osx_include_list ${header_list})



configure_file(src/conv2d.h.in ${CMAKE_SOURCE_DIR}/src/conv2d.h)
```

## cmake获取target 的library path

我遇到了一个奇怪的bug，就是使用`add_library`添加了一组link的对象接口之后，被链接的库没有被正确写入到target中，但是cmake又无法调试，所以只能从target中获取属性再打印出来看看，因为我的link library是interface的模式，所以要获取interface的property
```
get_target_property(OUT hkg_linux_src INTERFACE_LINK_LIBRARIES)
message(STATUS ${OUT})
```

然后我就发现了问题，即我们在编写`target_link_libraries`的时候，所有的字符串都是raw模式的，也就是说我们不需要加上双引号，我原来的写法如下
```
target_link_libraries(hkg_linux_src INTERFACE $<BUILD_INTERFACE:"${LINUX_SRCS}">
    $<INSTALL_INTERFACE:"${LINUX_SRCS}">)
```
则他的输出为
```
$<BUILD_INTERFACE:"/home/workspace/kernels_generator/include/hkg/generated_kernels/halide_conv2d_7x7_linux_bare.o">
```
而这里面的字符串是原封不动的写入到target_link_libraries中的，此时多了的双引号就是引起错误的根源了。

所以需要这样写，就可以避免这个问题了。
```
target_link_libraries(hkg_linux_src INTERFACE $<BUILD_INTERFACE:${LINUX_SRCS}>
    $<INSTALL_INTERFACE:${LINUX_SRCS}>)
```

## undefined reference to `pthread_create'

这个问题也老奇怪了，明明添加了`-lphread`但是还是报错，发现要使用`-phread`才可以。



## macos中的rpath问题

RPATH就是可执行文件寻找他动态链接库的路径,在linux中,大多数命令都是以以下顺序去搜索动态链接库的

1.  RPATH
2.  LD_LIBRARY_PATH
3.  RUNPATH
4.  /etc/ld.so.conf 
5.  builtin directories

通常我们设置LD_LIBRARY_PATH为给可执行程序提供库的路径,但是如果有同时存在多个版本的库都在路径中,同时我们还需要链接不同版本的库,那应该怎么办? 这时候就需要设置RPATH或者RUNPATH,提供一个指定的路径/

但是macos的情况有些不同,他的连接器 dyld 使用每个 dylib 的完整路径来定位依赖的动态库。

可执行文件中用完整路径记录他的依赖库:
```sh
❯ otool -L bin/can_use_target.generator
bin/can_use_target.generator:
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1292.100.5)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 905.6.0)
```



## conan支持riscv

我今天才发现Conan build的时候没有按照编译器选择对应的arch,然后看了一下,他们官方居然都没有做rv的支持. 他可选的setting中没有rv这个平台. 最终的cmake中的修改如下:
```cmake
    set(TARGET_ARCH x86_64)
    if(ENABLE_K510_RUNTIME AND BUILDING_TEST)
        # NOTE you need add `riscv64` in `~/.conan/settings.yml` `arch, arch_build, arch_target` item.
        # refer from https://github.com/conan-io/cmake-conan/issues/307
        set(TARGET_ARCH "riscv64")
    endif()
    
    conan_cmake_run(BASIC_SETUP
                    CONANFILE conanfile-runtime.txt
                    SETTINGS compiler.cppstd=17
                    BUILD missing
                    ARCH ${TARGET_ARCH}
                    ENV CC=${CMAKE_C_COMPILER}
                    ENV CXX=${CMAKE_CXX_COMPILER}
                    ENV CFLAGS=${CMAKE_C_FLAGS}
                    ENV CXXFLAGS=${CMAKE_CXX_FLAGS})
    include(${CMAKE_BINARY_DIR}/conan_paths.cmake)
```
