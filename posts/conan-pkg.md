---
title: Conan使用汇总
mathjax: true
toc: true
categories:
  - 编程语言
date: 2021-06-22 11:36:16
tags:
-   CPP
-   Conan
-   踩坑经验
---

记录使用conan时遇到的各种问题.

<!--more-->

# 为第三方库进行打包

1.  构建一个新pkg

```sh
conan new Clang/12.0.0 -t 
```

2.  修改conanfile.py

直接去`https://github.com/conan-io/conan-center-index/blob/master/recipes/`下面参考各种打包的流程即可

3.  下载代码到本地调试构建


注意每次build的时候最好把build目录的CmakeCache.txt删除了再开始，不然有时候你以为你成功，但其实只是偶尔的cache是正确的。

```sh
conan export . Clang/12.0.0@demo/testing # 修改代码后需要export才能生效 
conan install . --install-folder build_x86_64 -s arch=x86_64 # 如果我们的当前开发的代码需要依赖别的conan包，可以先把别的包install到对应目录 -s可以修改配置，默认配置在conan config下面
conan source . --install-folder build_x86_64 --source-folder src  # 调用source方法下载源码并做修改
# 可以分别调用build的三个阶段使用conan对源码进行编译
conan build . --build-folder build_x86_64 --source-folder src --configure
conan build . --build-folder build_x86_64 --source-folder src --build
conan build . --build-folder build_x86_64 --source-folder src --install
# conan build . --build-folder build_x86_64 --source-folder src # 也可以一步到位
conan package . --build-folder build_x86_64 --package-folder=package_x86 # 打包
# 打包之后还需要同步到local cache中才能被其他包使用，调用这个命令可以自动打包
conan export-pkg . Clang/12.0.0@demo/testing --build-folder=build_x86_64 -s arch=x86
```

4.  打包上传云端

如果上面的步骤之后，本地去include对应的代码可以正常编译，那么就可以导出pkg了
```sh
conan create .
```


## 注意点

1.  如果需要修改第三方库的源码，可以用patch的形式，调用conan的方法进行修改。

通常都是要在第三方库中添加如下信息的，因为你所有的依赖都是交给conan了，而不是有的通过本机，有的通过Conan。conan会给所有的第三方库都生成一个findxxx.cmake，通过conan_basic_setup去执行，执行完毕后，find_package就可以找到来自与Conan的包了。同时这里的`${CONAN_BIN_DIRS}`也可以添加，添加之后可以调用预编译lib中的一些可执行文件。

还有就是有的库写的find_package都是用xxx_DIR的方式去寻找xxxConfig.cmake的，这种方式和conan的行为不一致，需要修改。

```cmake
  message(STATUS "Loading conan scripts for Clang dependencies...")
  include("${CMAKE_BINARY_DIR}/conanbuildinfo.cmake")
  message(STATUS "Doing conan basic setup")
  conan_basic_setup()
  list(APPEND CMAKE_PROGRAM_PATH ${CONAN_BIN_DIRS})
  message(STATUS "Conan setup done. CMAKE_PROGRAM_PATH: ${CMAKE_PROGRAM_PATH}")
```
`list(APPEND CMAKE_PROGRAM_PATH ${CONAN_BIN_DIRS})`


2.  对于LLVM来说，最好不能使用`BUILD_SHARED_LIBS`，因为LLVM依赖于全局数据，这些数据最终可能会在共享库之间复制可能会导致错误。也就是默认是static的。

3.  conan官方的一些包的recipe里面是把他们的一些cmake文件删除了，比如他的llvm-core只有一堆静态库，这就非常蛋疼了，想依赖这个lib去构建clang是不行的。所以还得自己打包。


4.  默认的conan的build_folder就是同个目录，他都是打包结束后直接从当前文件夹下面选择性的去拷贝到package_folder下面。当然这个过程可以随便自定义，我觉得直接cmake install到package目录就完事了。


5.  报错`ConanException: llvm-core/12.0.0 package_info(): Package require 'libxml2' not used in components requires`
  
    我用官方的llvm-core就有这个问题，我一开始一直以为是我的问题，然后发现他可能打包的时候就出问题了。conan的设计思路是每个package_info里面提供了每个库的详细信息，像llvm这种库，是由多个组件构成的，为了详细起见，他得把每个component的requires写清楚，所以llvm的打包脚本里面就先用cmake生成依赖关系，然后packeage info里面解析依赖关系添加依赖，这个问题的出现就是因为，明明整个库要求了libxml2，但是里面没有一个component去依赖这个库，那不就说明不需要依赖吗，所以直接报错。

    我找了一下发现libxml2是被LLVMWindowsManifest引用的，但是输出依赖信息：`windowsmanifest ['support']` ,并没有添加这个依赖。目前猜想要么是patch没有打上，要么是依赖关系没有生成正确。

    还有就是他的component.json不知道是哪里弄出来的，cmake和conan的文档里面都没有写生成这个文件的地方。

6.  如果直接include整个导出的llvm-core，会报错找不到一些动态链接库，然后我发现llvm官方打包的二进制里面是没有的。应该是conan生成package info的时候没有删除这些不需要的依赖。

7.  conan可以生成Conan find package，然后在cmake中调用find package可以找到对应的包

7.  用打包出来的llvm-core去链接，一直报错找不到一个函数的定义（那个函数里面有个string），各种尝试才发现是conan自己生成的`conan_basic_setup`里面默认会从系统的profile中读取定义`compiler.libcxx=libstdc++`，然后设置`-D_GLIBCXX_USE_CXX11_ABI=0`。但是问题在于我也是用相同的编译器选项去编译llvm的，为什么生成的llvmlib却需要`GLIBCXX_USE_CXX11_ABI=1`呢？

    先用命令查看一下目前的编译器abi版本，发现默认是cxx11的
    ```
    gcc -v 2>&1 | sed -n 's/.*\(--with-default-libstdcxx-abi=new\).*/\1/p'
    --with-default-libstdcxx-abi=new
    ```
    在cmake中可以用如下命令查看添加了说明编译定义。
    ```
    get_directory_property( DirDefs COMPILE_DEFINITIONS )
    message( "COMPILE_DEFINITIONS = ${DirDefs}" )
    ```
    最后发现是我自己忘记在编译llvm的时候添加上conan basic setup了，导致没有指定。

8.  `libxmls nanohttp.c:(.text+0x507): undefined reference to 'fcntl64'`

    发现conan这东西出发点是好的，但是一定得需要把一个包所有的依赖全部展示清楚才好，上面这个问题就是预编译好的xml2需要的fcntl64包我的系统并没有。以后还是不要搞跨平台了，直接都用docker+linux完事了，没有环境问题。不然再怎么打包都会有奇怪的问题。。
    这个问题估计是因为预编译的libxml2的版本ubuntu20的，但是我是ubuntu18的，所以重新编译安装xml2。


9.  `LLVM ERROR: inconsistency in registered CommandLine options`
    
    我不知道为啥编译出来的clang居然是动态链接的，而llvm是静态编译的，从而导致的问题。但是clang的cmake根本就不接受动态链接的编译配置啊，重新configure一下编译就解决了。。

10. 利用cmake运行Conan，因为很多开发环境是不支持自动执行conan再cmake编译的，这样就不能提供自动补全、直接debug等功能了，所以需要自动化这个流程。但是我们要对开发的库进行conan打包的时候，又需要执行conan的命令，所以conan官方的解决方案是这里`https://github.com/conan-io/cmake-conan`,对cmake需要做以下修改：
    
    ```cmake
    if(CONAN_EXPORTED) # in conan local cache
        # standard conan installation, deps will be defined in conanfile.py
        # and not necessary to call conan again, conan is already running
        include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
        conan_basic_setup()
    else() # in user space
        include(conan.cmake)
        # Make sure to use conanfile.py to define dependencies, to stay consistent
        conan_cmake_configure(REQUIRES fmt/6.1.2 GENERATORS cmake_find_package)
        conan_cmake_autodetect(settings)
        conan_cmake_install(PATH_OR_REFERENCE . BUILD missing REMOTE conan-center SETTINGS ${settings})
    endif()
    ```

11. `CONAN_PKG::xxxx`这个是Conan setup时候的一个选项，如果添加了`TARGETS`的选项，使用Conan添加的lib都用统一的接口，但是这样其实不是很好，如果当前开发环境比较混乱的话（引用的包引用了conan的包，但是你只想直接调用，就会报错找不到`CONAN_PKG::xxxx`），所以要么都设置成相同的依赖，要么cmake里面再多写点。

12. 想要正确的导出conan包，让他和原来的包一样使用，还是比较非常麻烦。比如`halide`，原始的halide中有个target叫`Halide::Generator`，他的属性中包含了GenGen.cpp这个文件。首先如果想在cmake中使用这个target，conan打包就需要把这个component明确导出，但是只要导出一个componet，你所有的requirs都得指定到对应的component。llvm中就是先分析依赖图，然后手动解析，构建出对应的component列表，然后在一个个字符串处理，找到对应的依赖、系统库依赖。上面的事情都做完了，你还是没办法使用conan导出的`Halide::Generator`，因为conan还不支持为component添加文件属性，真的无语了。

# 使用第三方包

现在conan已经更新到了`2.2.1`版本了，使用方式也和之前有所不同了。


## conanfile.txt / conanfile.py

用conanfile.txt可以方便的添加依赖，用conanfile.py的话除了添加依赖，还可以控制更多的内容，同时打包自己的库。

下面展示一个典型的conanfile.txt，这里`generators`里面需要写上需要conan生成的东西,`CMakeDeps, CMakeToolchain`生成出来是为了在cmake中正确find package的。 然后`cmake_layout`也是必不可少的，有了他才能把生成的文件放到我们需要的位置。

```
[requires]
gtest/1.17.0
llvm-openmp/20.1.6

[generators]
CMakeDeps
CMakeToolchain

[layout]
cmake_layout

[options]
```


## profile

描述好了依赖，然后需要安装依赖，这个时候conan提供了profile让我们来控制依赖包的配置信息：

这是我构建的profile.user, 默认c++20：
```txt
[settings]
arch=armv8
build_type=Release
compiler=clang
compiler.cppstd=20
compiler.libcxx=libc++
compiler.version=18
os=Macos

[conf]
tools.cmake.cmaketoolchain:generator=Ninja
tools.cmake.cmake_layout:build_folder=build
tools.cmake.cmake_layout:build_folder_vars=[]
```

然后安装的时候还需使用`--profile:all`, 因为conan他可以用于交叉编译，host/target可以配置不同的profile。但我现在只有host，所以统一使用同一个profile：
```sh
conan install . --build=missing --profile:all=profile.user
```


## CMakeUserPresets.json

install之后会自动生成`build/Release/generators/CMakePresets.json`的json，然后项目目录下的`CMakeUserPresets.json`。正常来说基于`CMakePresets.json`就是可以正常编译了的，但是我这里还需要添加新的选项，那么就需要改`CMakeUserPresets.json`。

我这里分了一个`base`的配置和`debug`的配置，要注意最终使用的`debug`配置中，必须写上`binaryDir,toolchainFile`否则还是会无法编译。 然后还有一个隐藏坑点，conan如果按release模式安装，那么他的依赖库都需要通过release模式依赖，但是现在我需要给自己的项目编译debug版本，去依赖包的时候就会丢失各种信息，所以需要` "CMAKE_FIND_PACKAGE_PREFER_CONFIG": true, "CMAKE_MAP_IMPORTED_CONFIG_DEBUG": "Release"`把我当前的debug按release模式去依赖包：

```json
"configurePresets": [
        {
            "name": "base",
            "hidden": true,
            "generator": "Ninja",
            "installDir": "${sourceDir}/out/install/${presetName}",
            "cacheVariables": {
                "CMAKE_EXPORT_COMPILE_COMMANDS": true
            }
        },
        {
            "name": "debug",
            "inherits": [
                "base",
                "conan-release"
            ],
            "binaryDir": "${sourceDir}/build/Release",
            "toolchainFile": "generators/conan_toolchain.cmake",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Debug",
                "CMAKE_FIND_PACKAGE_PREFER_CONFIG": true,
                "CMAKE_MAP_IMPORTED_CONFIG_DEBUG": "Release"
            }
        }
    ]
```

# 手动下载包


## 1.x 版本


我要安装opencv4.5.1一直网络错误，所以需要手动下载包。 首先在`~/.conan/data/opencv/4.5.1/_/_/export`中找到conanfile.py，然后检查他是如何获得source的。 发现他的source是来自`conandata.yml`中的， 然后找到：
```yml
sources:
  4.5.1:
  - sha256: e27fe5b168918ab60d58d7ace2bd82dd14a4d0bd1d3ae182952c2113f5637513
    url: https://github.com/opencv/opencv/archive/4.5.1.tar.gz
  - sha256: 12c3b1ddd0b8c1a7da5b743590a288df0934e5cef243e036ca290c2e45e425f5
    url: https://github.com/opencv/opencv_contrib/archive/4.5.1.tar.gz
```

手动下载它们。 解压之后copy到`~/.conan/data/opencv/4.5.1/_/_/source/src`不保留root。 然后删除``~/.conan/data/opencv/4.5.1/_/_/source.dirty`表示source已经成功了。接着build就可以直接开始编译了。

