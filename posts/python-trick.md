---
title: 一些python中的小坑
mathjax: true
toc: true
categories:
  - 编程语言
date: 2021-08-24 16:03:45
tags:
- 踩坑经验
- Python
---

最近在重度使用 `python` , 记录一下一些进阶使用时的问题.

<!--more-->

# 小心输入参数为可能为生成器

给一个例子如下,我们的 `node` 需要接受一个 tuple 参数
```python
class Node():
  def __init__(self, op: str, args: tuple) -> None:
    self.op = op
    self.args = args

  def __repr__(self) -> str:
    if self.args:
      return '{' + f" {self.op}({','.join(str(arg) for arg in self.args)}) " + '}'
    else:
      return f' {self.op}() '
```

但是我们可能在构造的时候没有注意到实际传入的变量类型是什么,当我们调用 `node` 第一次与第二次时,可以观察到出现了不同的结果:
```python
node = Node('a', (Node('a' + str(i), ()) for i in range(10)))
node
# { a( a0() , a1() , a2() , a3() , a4() , a5() , a6() , a7() , a8() , a9() ) }

node
# { a() }
```

这其实就是 `python3` 里面的一个 `feature` 那就是列表推导默认得到生成器, 结果我一不小心踩进去找了半天 `bug`. 所以我们在构造节点的时候需要这样:
```python
node = Node('a', tuple(Node('a' + str(i), ()) for i in range(10)))
```

# `__new__` 的时候默认参数的问题

比如我们这样定义一个类,但是用默认参数初始化(无参数初始化)就会报错, 比如得在 `overwrite` 的 `__new__` 那边加上默认参数才可以,我属实是没明白.

```python
class Base():
  def __new__(cls, name, attr):
    obj = super().__new__(cls)
    obj.name = name
    obj.attr = attr
    return obj

  def __init__(self, value = 0) -> None:
    self.value = value

  def __repr__(self) -> str:
    return f'{self.name} {self.attr}: {self.value}'

class Var(Base):
  def __new__(cls, value: int): # <- 必须在这里也加上默认参数
    return super().__new__(cls, 'var', 'constant')
```


# `__future__.annotations`与`type hint`反射冲突  

通常情况下`__future__.annotations`是用来解决当前`type hint`的类型还没有被声明就被使用的情况,但是如果像我这样拿`type hint`来做自省,那么就会遇到问题,也就是自省得到的结果从一个`type`被转变为了一个`str`,就做不了更多的事情了.

```python
# from __future__ import annotations
from enum import Enum


class color(Enum):
  r = 0x01
  g = 0x02
  b = 0x03


class A():
  c: color.r

print(A.__annotations__)
```

输出
```sh
# 开启__future__.annotations
{'c': 'color.r'}
# 关闭__future__.annotations
{'c': <color.r: 1>}
```

# `python`与`cpp`交互时需要注意对象的生命周期

我们底层在`cpp`中实现了一个解释器,这个解释器的 `pc` 指向了一个指令`bytes`,但是如果在`python`中以这种形式调用就会出错:
```python
interp =  interpreter()
interp.load(open('xx','rb').read())
interp.run()
```
而且报错的行为还是非常诡异的,不开启`python`的 `debug` 时候,直接报错 `core dump`, 如果开启 `debug` 的时候, 解释器`pc`读到的指令全部都是`0x00`.

实际上问题就是`python`会自动把我们读出来的指令流回收,因为他认为退出了`load`的函数域就没有人在使用了,但是实际上我们的指针还指向那块内存. 同时我估计 `python debug` 的时候分配的内存区域和正常执行时不一样,所以`debug`状态指针不会报越界错误. 目前的解决方案就是在`pybind11`外面再用继承的方式重新实现一些类方法,手动保存好程序数据.


# `namedtuple`默认参数为`list`时

代码如下, 如果是默认参数为`list`的`NamedTuple`,在新创建对象时那两个 `list` 还是指向同一个 `list`, 就会导致意外的问题. 其实主要是`NamedTuple`是通过`metaclass`构造的.我们没法重写`new`方法,这个太蛋疼了.

```python
from typing import NamedTuple

class T(NamedTuple):
  l1 = []
  l2 = []
t1 = T()
t1.l1.append(1)
t1.l2.append(2)

t2 = T()
t2.l1 # [1]
t2.l2 # [2]
```

# product 的实现

```python
fi = [i for i in range(2)]
fj = [j for j in range(3)]
fk = [k for k in range(4)]

sequences = [tuple(pool) for pool in [fi, fj, fk]]
accumulator = [[]]
for sequence in sequences:
  temp = []
  for acc in accumulator:
    for item in sequence:
      temp.append(acc + [item])
  print(temp)
  accumulator = temp
# step 1: [[0], [1]]
# step 2: [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
# step 3: [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3], [0, 2, 0], [0, 2, 1], [0, 2, 2], [0, 2, 3], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 0, 3], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 2, 0], [1, 2, 1], [1, 2, 2], [1, 2, 3]]
```

# split combine 实现

```python
sequences = [tuple(pool) for pool in [fi, fj, fk]]
accumulator = [[]]
for sequence in sequences:
  temp = []
  init = True
  for item in sequence:
    if init:
      for acc in accumulator:
        temp.append([item] + acc)
      init = False
    else:
      temp.append([item] + accumulator[-1])
  print(temp)
  accumulator = temp
# step 1. [[0], [1], [2], [3], [4]]
# step 1. [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 4], [2, 4]]
# step 1. [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 1, 4], [0, 2, 4], [1, 2, 4], [2, 2, 4], [3, 2, 4]]
```


# 打包相关

主要还是使用setup.py, 在里面可以继承他原本的cmd类来执行自己的命令，主要就是他默认的命令是有一些执行顺序的。
```sh
python setup.py 
Standard commands:
  build             build everything needed to install
  build_py          "build" pure Python modules (copy to build directory)
  build_ext         build C/C++ extensions (compile/link to build directory)
  build_clib        build C/C++ libraries used by Python extensions
  build_scripts     "build" scripts (copy and fixup #! line)
  clean             (no description available)
  install           install everything from build directory
  install_lib       install all Python modules (extensions and pure Python)
  install_headers   install C/C++ header files
  install_scripts   install scripts (Python or otherwise)
  install_data      install data files
  sdist             create a source distribution (tarball, zip file, etc.)
  bdist             create a built (binary) distribution
  bdist_dumb        create a "dumb" built distribution
  bdist_rpm         create an RPM distribution
  check             perform some checks on the package

Extra commands:     # 这里的命令是用户在setup.py中自定义的部分
  install_ext       install all Python modules (extensions and pure Python)
  develop           (no description available)
  pytests           (no description available)
  ctests            (no description available)
  bdist_wheel       create a wheel distribution
  alias             define a shortcut to invoke one or more commands
  bdist_egg         create an "egg" distribution
  dist_info         DO NOT CALL DIRECTLY, INTERNAL ONLY: create .dist-info directory
  easy_install      (no description available)
  editable_wheel    DO NOT CALL DIRECTLY, INTERNAL ONLY: create PEP 660 editable wheel
  egg_info          create a distribution's .egg-info directory
  install_egg_info  Install an .egg-info directory for the package
  rotate            delete older distributions, keeping N newest files
  saveopts          save supplied options to setup.cfg or other config file
  setopt            set an option in setup.cfg or another config file
  isort             Run isort on modules registered in setuptools
```

`pip install -e .`对应的其实是setup.py中的`develop`命令，