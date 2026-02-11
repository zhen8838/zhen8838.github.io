---
title: Pythonnet踩坑
mathjax: true
toc: true
categories:
  - 编程语言
date: 2021-11-12 12:53:31
tags:
- Python
- CSharp
- 踩坑经验
---

关于python和dotnet互相调用时遇到的一些问题

<!--more-->

# 1.  安装pythonnet失败

## 1. pkg-config 失败
报错没有`pkg-config`,我这里是mac,所以执行:
```sh
brew install pkg-config
```
然后接着报错:
```sh
❯ pkg-config --libs mono-2
Package mono-2 was not found in the pkg-config search path.
Perhaps you should add the directory containing `mono-2.pc'
to the PKG_CONFIG_PATH environment variable
No package 'mono-2' found
```
需要添加环境变量`PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/lib/pkgconfig:/Library/Frameworks/Mono.framework/Versions/6.12.0/lib/pkgconfig:$PKG_CONFIG_PATH`,添加之后能找到正确路径.

## 2. clang 编译问题

安装时报错如下
```sh
  clang -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /Users/lisa/mambaforge/include -arch arm64 -fPIC -O2 -isystem /Users/lisa/mambaforge/include -arch arm64 -I/Users/lisa/mambaforge/include/python3.9 -c src/monoclr/clrmod.c -o build/temp.macosx-11.0-arm64-3.9/src/monoclr/clrmod.o -D_THREAD_SAFE -I/Library/Frameworks/Mono.framework/Versions/6.12.0/lib/pkgconfig/../../include/mono-2.0
  In file included from src/monoclr/clrmod.c:1:
  In file included from src/monoclr/pynetclr.h:4:
  /Users/lisa/mambaforge/include/python3.9/Python.h:25:10: fatal error: 'stdio.h' file not found
  #include <stdio.h>
           ^~~~~~~~~
  1 error generated.
  error: command '/usr/local/bin/clang' failed with exit code 1
  ----------------------------------------
  ERROR: Failed building wheel for pythonnet
```
发现是我用的clang是自己从llvm编译的,然后标准的clang就有这个问题.我删除之后重新安装了apple-clang就可以编译了.

## 3. import clr 失败

```python
ImportError                               Traceback (most recent call last)
~/Documents/nncase/tests/importer/tflite/basic/test_batch_to_space.py in <module>
----> 16 import clr
      17 clr.AddReference("Nncase.Importer")

ImportError: dlopen(/Users/lisa/mambaforge/lib/python3.9/site-packages/clr.cpython-39-darwin.so, 0x0002): symbol not found in flat namespace '_mono_assembly_get_image'
```

发现他对于除了win32的平台,全部都是用mono,但是我明明就是有dotnetcore的....
然后发现他的思路是除了win32,先全部用mono去编译,编译好了再自己选择用什么runtime
```python
from clr_loader import get_coreclr
rt = get_coreclr("/Users/lisa/Documents/nncase/runtimeconfig.json")
from pythonnet import set_runtime
set_runtime(rt)
import clr
```
然后我加载.netruntime的lib的时候又遇到一个问题,我用的是x64的.net,然后直接在m1上转译的运行没问题,但是一段编译好的arm代码要去加载那个x64的lib肯定是不行的, 所以我还得装.net6.
```sh
'/usr/local/share/dotnet/host/fxr/5.0.11/libhostfxr.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e')), 
```

## 4. 从master分支安装

发现上一个release其实并不支持dotnet core, 现在尝试从master分支安装.直接拉最新的代码之后:
```sh
pip install -e .
```
然后运行:
```sh
❯ python
Python 3.9.2 | packaged by conda-forge | (default, Feb 21 2021, 05:00:30) 
[Clang 11.0.1 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import clr
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/lisa/Documents/pythonnet/clr.py", line 6, in <module>
    load()
  File "/Users/lisa/Documents/pythonnet/pythonnet/__init__.py", line 36, in load
    set_default_runtime()
  File "/Users/lisa/Documents/pythonnet/pythonnet/__init__.py", line 22, in set_default_runtime
    set_runtime(clr_loader.get_mono())
  File "/Users/lisa/mambaforge/lib/python3.9/site-packages/clr_loader/__init__.py", line 21, in get_mono
    impl = Mono(
  File "/Users/lisa/mambaforge/lib/python3.9/site-packages/clr_loader/mono.py", line 25, in __init__
    initialize(
  File "/Users/lisa/mambaforge/lib/python3.9/site-packages/clr_loader/mono.py", line 103, in initialize
    _MONO = load_mono(libmono)
  File "/Users/lisa/mambaforge/lib/python3.9/site-packages/clr_loader/ffi/__init__.py", line 38, in load_mono
    return ffi.dlopen(path, ffi.RTLD_GLOBAL)
  File "/Users/lisa/mambaforge/lib/python3.9/site-packages/cffi/api.py", line 150, in dlopen
    lib, function_cache = _make_ffi_library(self, name, flags)
  File "/Users/lisa/mambaforge/lib/python3.9/site-packages/cffi/api.py", line 832, in _make_ffi_library
    backendlib = _load_backend_lib(backend, libname, flags)
  File "/Users/lisa/mambaforge/lib/python3.9/site-packages/cffi/api.py", line 827, in _load_backend_lib
    raise OSError(msg)
OSError: cannot load library '/Library/Frameworks/Mono.framework/Versions/Current/lib/libmonosgen-2.0.dylib': dlopen(/Library/Frameworks/Mono.framework/Versions/Current/lib/libmonosgen-2.0.dylib, 0x000A): tried: '/Users/lisa/Documents/nncase/out/build/debug/lib/libmonosgen-2.0.dylib' (no such file), '/Users/lisa/Documents/k510-gnne-compiler/out/build/debug/lib/libmonosgen-2.0.dylib' (no such file), '/libmonosgen-2.0.dylib' (no such file), '/Library/Frameworks/Mono.framework/Versions/Current/lib/libmonosgen-2.0.dylib' (fat file, but missing compatible architecture (have 'i386,x86_64', need 'arm64e')), '/usr/local/lib/libmonosgen-2.0.dylib' (no such file), '/usr/lib/libmonosgen-2.0.dylib' (no such file), '/Users/lisa/Documents/nncase/out/build/debug/lib/libmonosgen-2.0.1.dylib' (no such file), '/Users/lisa/Documents/k510-gnne-compiler/out/build/debug/lib/libmonosgen-2.0.1.dylib' (no such file), '/libmonosgen-2.0.1.dylib' (no such file), '/Library/Frameworks/Mono.framework/Versions/6.12.0/lib/libmonosgen-2.0.1.dylib' (fat file, but missing compatible architecture (have 'i386,x86_64', need 'arm64e')), '/usr/local/lib/libmonosgen-2.0.1.dylib' (no such file), '/usr/lib/libmonosgen-2.0.1.dylib' (no such file).  Additionally, ctypes.util.find_library() did not manage to locate a library called '/Library/Frameworks/Mono.framework/Versions/Current/lib/libmonosgen-2.0.dylib'
```

发现是mono的lib也是转译的,需要加在arm64的dotnet core.
用上面的切换运行时的方法就可以成功加载了.

## 5. python 调用 .net

我发现他这里有点问题,如果只添加`Nncase.Importer`会提示找不到`Nncase`的namespace, 需要添加`Nncase.Core`才可以. 并且好像是不支持嵌套的 namespace导入.反正我没法直接导入`Nncase.Importer.TFlite`, 并且python调用.net不太好调试, 所以还是放弃了.

```python
import clr
import sys

clr.AddReference(
    "/Users/lisa/Documents/nncase/src/Nncase.Core/bin/Debug/net6.0/Nncase.Core.dll")
clr.AddReference(
    "/Users/lisa/Documents/nncase/src/Nncase.Importer/bin/Debug/net6.0/Nncase.Importer.dll")

from System import String
from System.Collections import *
from Nncase import Importers
```

## 6. .net 调用 python

这里就是需要提供一个环境变量或者手动指定pythonlib的路径比较麻烦, 其他还好.

```csharp
using Xunit;
using Nncase;
using Nncase.IR;
using System.Numerics.Tensors;
using System.Collections.Generic;
using Python.Runtime;
using System;

namespace Nncase.Tests.Importer
{

    public class TestPython
    {
        [Fact]
        public void TestNumpy()
        {
            Runtime.PythonDLL = "/Users/lisa/mambaforge/lib/libpython3.9.dylib";
            using (Py.GIL())
            {
                dynamic np = Py.Import("numpy");
                Console.WriteLine(np.cos(np.pi * 2));

                dynamic sin = np.sin;
                Console.WriteLine(sin(5));

                double c = (double)(np.cos(5) + sin(5));
                Console.WriteLine(c);

                dynamic a = np.array(new List<float> { 1, 2, 3 });
                Console.WriteLine(a.dtype);

                dynamic b = np.array(new List<float> { 6, 5, 4 }, dtype: np.int32);
                Console.WriteLine(b.dtype);

                Console.WriteLine(a * b);
            }
        }
    }


}
```