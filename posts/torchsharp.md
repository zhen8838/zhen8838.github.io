---
title: 关于如何在M1上使用TorchSharp
mathjax: true
toc: true
categories:
  - 编程语言
date: 2021-11-18 17:23:57
tags:
- Pytorch
- CSharp
---


TorchSharp只有x64的,太蛋疼了,所以需要重新安装一遍.

<!--more-->

## 1.  下载源码

```sh
git clone https://github.com/dotnet/TorchSharp.git --depth 1
```

## 2.  下载编译libtorch

```sh
git clone -b v1.10.0 --recurse-submodule https://github.com/pytorch/pytorch.git --depth 1
mkdir pytorch-build
cd pytorch-build
cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -DCMAKE_INSTALL_PREFIX:PATH=../libtorch ../pytorch -G "Ninja"
cmake --build . --target install
```

注意这里默认是不开test的生成的.

## 3.  修改TorchSharp的编译命令

这里首先是修改libTrochSharp的cmake, 把llvm的路径改成本机的. 然后因为他默认要copy libtorch中的一些test的dylib到nupkg中,我之前编译的libtorch中没有test的输出, 这里需要注释掉. 最后就是把sdk版本改成net6.

```patch
diff --git a/Directory.Build.props b/Directory.Build.props
index 34f28ab..28b977e 100644
--- a/Directory.Build.props
+++ b/Directory.Build.props
@@ -92,7 +92,7 @@
     <!-- By default only TorchSharp and no libtorch-cpu or libtorch-cuda packages are built.  The CI file controls these via 'BuildLibTorchPackages' -->
     <!-- This then selectively turns these on over several CI jobs since different pacakges are done in different jobs -->
     <IncludeTorchSharpPackage>true</IncludeTorchSharpPackage>
-    <IncludeLibTorchCpuPackages>false</IncludeLibTorchCpuPackages>
+    <IncludeLibTorchCpuPackages>true</IncludeLibTorchCpuPackages>
     <IncludeLibTorchCudaPackages>false</IncludeLibTorchCudaPackages>
   </PropertyGroup>
 
diff --git a/Directory.Build.targets b/Directory.Build.targets
index b79d260..3383de5 100644
--- a/Directory.Build.targets
+++ b/Directory.Build.targets
@@ -68,18 +68,18 @@
   <ItemGroup Condition="'$(NativeTargetArchitecture)' == 'x64'and $([MSBuild]::IsOSPlatform('osx')) and '$(TestUsesLibTorch)' == 'true'  and '$(SkipNative)' != 'true' ">
     <NativeAssemblyReference Include="c10" />
     <NativeAssemblyReference Include="caffe2_detectron_ops" />
-    <NativeAssemblyReference Include="caffe2_module_test_dynamic" />
+    <!-- <NativeAssemblyReference Include="caffe2_module_test_dynamic" /> -->
     <NativeAssemblyReference Include="caffe2_observers" />
-    <NativeAssemblyReference Include="fbjni" />
-    <NativeAssemblyReference Include="iomp5" />
-    <NativeAssemblyReference Include="jitbackend_test" />
-    <NativeAssemblyReference Include="pytorch_jni" />
+    <!-- <NativeAssemblyReference Include="fbjni" /> -->
+    <!-- <NativeAssemblyReference Include="iomp5" /> -->
+    <!-- <NativeAssemblyReference Include="jitbackend_test" /> -->
+    <!-- <NativeAssemblyReference Include="pytorch_jni" /> -->
     <NativeAssemblyReference Include="shm" />
     <NativeAssemblyReference Include="torch" />
     <NativeAssemblyReference Include="torch_cpu" />
     <NativeAssemblyReference Include="torch_global_deps" />
     <NativeAssemblyReference Include="torch_python" />
-    <NativeAssemblyReference Include="torchbind_test" />
+    <!-- <NativeAssemblyReference Include="torchbind_test" /> -->
   </ItemGroup>
 
   <!-- Linux CPU libtorch binary list used for examples and testing -->
diff --git a/global.json b/global.json
index 058794f..15f9eb4 100644
--- a/global.json
+++ b/global.json
@@ -1,6 +1,6 @@
 {
   "sdk": {
-    "version": "5.0.402",
+    "version": "6.0.100",
     "allowPrerelease": true,
     "rollForward": "minor"
   }
diff --git a/src/Native/LibTorchSharp/CMakeLists.txt b/src/Native/LibTorchSharp/CMakeLists.txt
index 46b5c32..b1828cc 100644
--- a/src/Native/LibTorchSharp/CMakeLists.txt
+++ b/src/Native/LibTorchSharp/CMakeLists.txt
@@ -1,8 +1,8 @@
 project(LibTorchSharp)
 
 if(APPLE)
- include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
- link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")
+ include_directories("/Users/lisa/Documents/llvm-project/build/install/include")
+ link_directories("/Users/lisa/Documents/llvm-project/build/install/lib")
 endif()
 find_package(Torch REQUIRED PATHS ${LIBTORCH_PATH})
 
diff --git a/src/Redist/libtorch-cpu/libtorch-cpu.proj b/src/Redist/libtorch-cpu/libtorch-cpu.proj
index 4240fe1..3eeeabe 100644
--- a/src/Redist/libtorch-cpu/libtorch-cpu.proj
+++ b/src/Redist/libtorch-cpu/libtorch-cpu.proj
@@ -41,21 +41,21 @@
     <File Include="libtorch\lib\uv.dll" />
   </ItemGroup>
   <ItemGroup Condition="'$(TargetOS)' == 'mac'">
-    <File Include="libtorch\lib\libbackend_with_compiler.dylib" />
+    <!-- <File Include="libtorch\lib\libbackend_with_compiler.dylib" /> -->
     <File Include="libtorch\lib\libc10.dylib" />
     <File Include="libtorch\lib\libcaffe2_detectron_ops.dylib" />
-    <File Include="libtorch\lib\libcaffe2_module_test_dynamic.dylib" />
+    <!-- <File Include="libtorch\lib\libcaffe2_module_test_dynamic.dylib" /> -->
     <File Include="libtorch\lib\libcaffe2_observers.dylib" />
-    <File Include="libtorch\lib\libfbjni.dylib" />
-    <File Include="libtorch\lib\libiomp5.dylib" />
-    <File Include="libtorch\lib\libjitbackend_test.dylib" />
-    <File Include="libtorch\lib\libpytorch_jni.dylib" />
+    <!-- <File Include="libtorch\lib\libfbjni.dylib" /> -->
+    <!-- <File Include="libtorch\lib\libiomp5.dylib" /> -->
+    <!-- <File Include="libtorch\lib\libjitbackend_test.dylib" /> -->
+    <!-- <File Include="libtorch\lib\libpytorch_jni.dylib" /> -->
     <File Include="libtorch\lib\libshm.dylib" />
     <File Include="libtorch\lib\libtorch.dylib" />
     <File Include="libtorch\lib\libtorch_cpu.dylib" />
     <File Include="libtorch\lib\libtorch_global_deps.dylib" />
     <File Include="libtorch\lib\libtorch_python.dylib" />
-    <File Include="libtorch\lib\libtorchbind_test.dylib" />
+    <!-- <File Include="libtorch\lib\libtorchbind_test.dylib" /> -->
   </ItemGroup>
   <ItemGroup Condition="'$(TargetOS)' == 'linux'">
     <File Include="libtorch\lib\libbackend_with_compiler.so" />
```

## 4. 开始打包

执行`dotnet pack --configuration release`,注意他这里会自动下载`libtorch`的release包,但是修改他的下载的脚本又着实麻烦.所以我们要等到他下载好开始编译`Native`的时候按`ctrl+c`先中断,然后把我们的libtorch替换他解压出来的libtorch
```sh
cd bin/obj/AnyCPU.Release/libtorch-cpu/libtorch-macos-1.10.0cpu/
rm -rf libtorch
cp -r ~/Documents/libtorch .  
```
然后再执行`dotnet pack --configuration release`最后打包出来.

## 5. 配置nuget配置

我们要把编译生成的路径作为nuget的一个源,同时为了避免冲突还得关闭原来的源, 所以在项目路径下添加`NuGet.Config`文件

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" protocolVersion="3" />
    <add key="m1" value="/Users/lisa/Documents/TorchSharp/bin/packages/Release" />
  </packageSources>
  <activePackageSource>
    <add key="m1" value="/Users/lisa/Documents/TorchSharp/bin/packages/Release" />
  </activePackageSource>
  <disabledPackageSources>
    <add key="nuget.org" value="true" />
  </disabledPackageSources>
</configuration>
```

## 6. 编译项目

执行`dotnet build -v:d > log`, 我这里在log 文件中就可以看到如下内容
```sh
正在将文件从“/Users/lisa/.nuget/packages/libtorch-cpu-osx-x64/1.10.0.1/runtimes/osx-x64/native/libcaffe2_observers.dylib”复制到“/Users/lisa/Documents/nncase/bin/Nncase.Tests/net6.0/runtimes/osx-x64/native/libcaffe2_observers.dylib”。
正在将文件从“/Users/lisa/.nuget/packages/libtorch-cpu-osx-x64/1.10.0.1/runtimes/osx-x64/native/libcaffe2_detectron_ops.dylib”复制到“/Users/lisa/Documents/nncase/bin/Nncase.Tests/net6.0/runtimes/osx-x64/native/libcaffe2_detectron_ops.dylib”。
正在将文件从“/Users/lisa/.nuget/packages/libtorch-cpu-osx-x64/1.10.0.1/runtimes/osx-x64/native/libc10.dylib”复制到“/Users/lisa/Documents/nncase/bin/Nncase.Tests/net6.0/runtimes/osx-x64/native/libc10.dylib”。
```

我们检查dylib的格式, 如果是arm64就说明是ok的. 如果还是x86的话, 是因为替换libtorch的时机晚了,他是在`src/Redist/libtorch-cpu/libtorch-cpu.proj`中`CopyFilesFromArchive`执行的.

```sh
❯ file /Users/lisa/Documents/nncase/bin/Nncase.Tests/net6.0/runtimes/osx-x64/native/libcaffe2_observers.dylib
/Users/lisa/Documents/nncase/bin/Nncase.Tests/net6.0/runtimes/osx-x64/native/libcaffe2_observers.dylib: Mach-O 64-bit dynamically linked shared library arm64
```

## 7. 修复路径问题

这里还是有个问题, 那就是`TorchSharp`默认会从`TorchSharp.dll`的路径加载`libtorch.dylib`,但是我生成出来的包,默认是把`libtorch.dylib`放到`net6.0/runtimes/osx-x64/native`路径下,导致默认找不到这个`libtorch`. 手动解决到也简单

```sh
cd bin/Nncase.Tests/net6.0/
mv runtimes/osx-x64/native/* .
```