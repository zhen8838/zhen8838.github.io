---
title: TVM TensorIR
mathjax: true
toc: true
categories:
  - 编译器
date: 2021-12-04 13:20:04
tags:
- TVM
---

关于TVM的Tensor level IR.

<!--more-->

# 1. ffi navigator的bug修复

我这里是python3.9, 不知道为什么tvm的ffi navigator插件有一个类型问题启动不了. 所以需要修改`/Users/lisa/mambaforge/lib/python3.9/site-packages/ffi_navigator/dialect/tvm.py line 97`为如下:
```python
        if path.startswith("" if self._pypath_api_internal is None else self._pypath_api_internal):
```
有了这个看tvm的就舒服多了,不然你从python到c++的实现都非常难找.


# 2.  tvm.script.tir 与 tvm.tir

`tvm.tir`是内在实现. `tvm.script.tir`主要是封装了一层用户友好的python类型接口(不存在实现).可以查看[这篇文章](https://tvm.apache.org/docs/tutorial/tensor_ir_blitz_course.html). `tvm.script`实际上就是`tensor ir`的语法表现形式,我们通过写`tvm.script`语法,然后构建出`IRModule`. 避免了直接从ir构造的别扭,因为如果是relay这种,不需要考虑太多的条件以及循环等,如果是底层ir,用函数的方式写这些就非常蛋疼了.
比如从tir直接构造ir是这样的:
```python
ib = tvm.tir.ir_builder.create()
a = tir.Var("a", "float32")
b = tir.Var("b", "float32")
with ib.if_scope(True):
    ib.emit(tir.Evaluate(tir.ret(a)))
ib.emit(tir.Evaluate(tir.ret(b)))
stmt = ib.get()
func = tir.PrimFunc([a, b], stmt)
func = build_tir_func(func)
out = func(1.0, 2.0)
```
如果用`script.tir`就方便多了:
```python
@T.prim_func
def add(a: T.handle, b: T.handle):
  for i in T.parallel(0, 2):
    for j in T.serial(0, 1):
      for z in T.vectorized(3, 4):
        T.evaluate(0)
```

# 3. tvm.script -> tir的流程

首先我们使用`tvm.script.tir`写一个计算函数,然后被转换为`python`的`ast`,由于不同 `python` 版本之间的 `ast` 不同,所以 `tvm` 单独开发了一个和 `python` 版本无关的 `ast parser` 叫 `synr`. 在`parser`的使用利用`tvm`的`lower transformer`把`ast`进行细化. 要注意,用户层面导入`tvm.script.tir as T`实际上都只有类型而已, 他对于这些类型的实际定义并没有导入进来,而是在`tvm.script.parser`中使用.
```python
@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
  A = T.match_buffer(a, [128, 128])
  B = T.match_buffer(b, [128, 128])
  C = T.match_buffer(c, [128, 128])
  for i, j in T.grid(128, 128):
    with T.block("init"):
      vi, vj = T.axis.remap("SS", [i, j])
      C[vi, vj] = T.float32(0)
    for k in range(128):
      with T.block("update"):
        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]
```
转换为`tvm.tir.function.PrimFunc`就如下:
```python
PrimFunc([a, b, c])  {
  block root() {
    reads([])
    writes([])
    for (i, 0, 128) {
      for (j, 0, 128) {
        block init(iter_var(vi, range(min=0, ext=128)), iter_var(vj, range(min=0, ext=128))) {
          bind(vi, i)
          bind(vj, j)
          reads([])
          writes([C[vi, vj]])
          C[vi, vj] = 0f
        }
        for (k, 0, 128) {
          block update(iter_var(vi, range(min=0, ext=128)), iter_var(vj, range(min=0, ext=128)), iter_var(vk, range(min=0, ext=128))) {
            bind(vi, i)
            bind(vj, j)
            bind(vk, k)
            reads([C[vi, vj], A[vi, vk], B[vj, vk]])
            writes([C[vi, vj]])
            C[vi, vj] = (C[vi, vj] + (A[vi, vk]*B[vj, vk]))
          }
        }
      }
    }
  }
}
```

# 4 ir builder流程

`ir builder`提供了另一种构建`tir`的方法,典型用法如下:
```python
  ib = tvm.tir.ir_builder.create()
  n = te.size_var("n")
  A = ib.pointer("float32", name="A")
  tmod = tvm.tir.truncmod
  with ib.for_range(0, n, name="i") as i:
    with ib.if_scope(tmod(i, 2) == 0):
      A[i] = A[i] + 1
    with ib.else_scope():
      A[0] = A[i] + 2
  body = ib.get()
```
所有通过`ib.xx`构造的`ir`对象都会通过`ib.emit`的方式添加到`ir builer`内部,然后对于一些存在`scope`的比如`for if`等等, 是构造了一个`with scope`对象,然后在退出这个`scope`的时候把中间的所有`emit`生成的对象作为`body`构造成一个`for/if`的`ir`.


# 5.  tvm.te 与 tvm.tir

`te`里面的实际上是老的写法,他里面又写了一套`tensor/data producer`等等的`ir`, `te`的`ir`定义实际上是以`operation`为核心的,然后类似于`tensorflow`的`placeholder`的方式进行构建的,实际上在转换到`IRModule`的时候,还是会把这些东西转化为`tir.Buffer`.所以目前可以不看那块的内容.

# 6. 一些tir的作用

## 6.1 block reads && writes

`block`是`tvm`调度的基本单元,他的调度器通常是获得一个`block`,然后对这个块进行融合/分割/并行等等操作,同时还可以分析多个块
在`parser`的`block`的流程,他的`func.body`是只会有一个赋值的操作`C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]`(忽略了前面的`iter var`定义,应该是这些定义到时候都会被固化到代码中,所以也不会出现在计算流程中的原因),然后在`func.exit_scope`时,他会进入`tvm`的`callback`函数中 `python/tvm/script/tir/scope_handler.py line 255`,构造出带有`bind`以及`reads/writes`的`tir`. (实际上底层还分有`BlockRealize`和`Block`两部分)

```python
      with T.block("update"):
        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]
```

```python
        func.enter_scope(node, self.context, arg_list, node.rhs.func_name.span)
        func.body = self.parse_body(node)
        res = func.exit_scope(node, self.context, arg_list, node.rhs.func_name.span)
```

得到的结果,实际上是把`remap`的定义融合到了`block`这个`ir`中.
```python
for (k, 0, 128) {
  block update(iter_var(vi, range(min=0, ext=128)), iter_var(vj, range(min=0, ext=128)), iter_var(vk, range(min=0, ext=128))) {
    bind(vi, i)
    bind(vj, j)
    bind(vk, k)
    reads([C[vi, vj], A[vi, vk], B[vj, vk]])
    writes([C[vi, vj]])
    C[vi, vj] = (C[vi, vj] + (A[vi, vk]*B[vj, vk]))
  }
}
```

## 6.2 block iter_var

`iter_var`我个人把他看作一个`symbol var`,他的好处就是我们可以任意绑定一个时机的`value`,等到`schedule`做完后再消除他得到真正的索引操作. 
这里要说明一下`iter_var`对于一个`Buffer`的索引操作将会得到是`BufferLoad`的`ir`,他的表现形式就是多维索引`B[vi,vj]`. 在后续这个`BufferLoad`会被`lower`到`Load`,表现形式就是`B.Handle[i * w + j]`. 即我们取`symbol var`绑定的`value`并计算出对于一个指针真正的索引.

🌰 原始`TIR`:
```python
for (i: int32, 0, 128) {
  for (j: int32, 0, 128) {
    block([128, 128], "B") as [vi, vj] {
      bind(vi, i)
      bind(vj, j)
      tir.reads([A[vi, vj]])
      tir.writes([B[vi, vj]])
      B[vi, vj] = (A[vi, vj]*2f32)
  }
}
```
经过`split`之后, 可以发现我们只需要修改`iter var`的绑定即可实现`split`, 不然得递归把所有的`i`改成`((i_0*64) + i_1)`,写`transform`就巨麻烦了.
```python
for (i_0: int32, 0, 2) {
  for (i_1: int32, 0, 64) {
    for (j: int32, 0, 128) {
      block([128, 128], "B") as [vi, vj] {
        bind(vi, ((i_0*64) + i_1))
        bind(vj, j)
        tir.reads([A[vi, vj]])
        tir.writes([B[vi, vj]])
        B[vi, vj] = (A[vi, vj]*2f32)
    }
  }
}
```

## 6.3 BufferLoad lower

1. 利用`ConvertBlocksToOpaque`的`transform`把`iter_var.var`都替换成对应的`value`, 这里我其实没明白,为什么不把`itervar`也设计成`expr`, 理论上应该没啥问题吧.
<!-- 2. 把所有的buffer load -->

# 7. 代码生成

## 7.1 ssa赋值

我自己写了一下c代码生成才发现不能无脑对综合了stmt以及expr的ir进行ssa赋值.怪不得tvm的c代码生成默认不开ssa赋值.

🌰 把下面的代码转换为c代码
```csharp
void RefFunc(int[] A, int n)
{
    for (i in (0, n))
    {
        A[i] = A[i] + 1;
        for (j in (0, 10))
        {
            A[i] = A[i] + j;
        }
    }
}
```
如果使用ssa赋值,同时我这里的visit expression的时候是用结构化比较的,所以内外两个循环中相同的`load A[i]`都变成了`_1`这个`tmep var`了. 然后第二次`load`的时候就会出现没有更新值的问题.
```c
#include <stdint.h>
void func_0(int32_t* A, int32_t n) {
  for (int32_t i = 0; i < n; i++) {
    int32_t _3 = (i * 1);
    int32_t _2 = (0 + _3);
    int32_t _1 = A[_2];
    int32_t _0 = (_1 + 1);
     A[_2] = _0;
    for (int32_t j = 0; j < 10; j++) {
      int32_t _4 = (_1 + j); // 这里就会出现load没有更新值的问题
       A[_2] = _4;
    }
  }
}
```

所以我目前也是按照tvm的方法,把这些计算流程都转换成线性的计算. 这样就保证所有的表达式都会被`emit`,不过也带来了一个计算冗余的问题,这个后续我们可以继续优化.
```c
#include <stdint.h>
void func_0(int32_t* A, int32_t n) {
  for (int32_t i = 0; i < n; i++) {
     A[(0 + (i * 1))] = (A[(0 + (i * 1))] + 1);
    for (int32_t j = 0; j < 10; j++) {
       A[(0 + (i * 1))] = (A[(0 + (i * 1))] + j);
    }
  }
}
```

# 从relay到tir

默认tvm是在codegen中执行这个过程, 不过没法直接dump出对应的tir来看, 不过我们可以通过自定义pass的方法插入print节点.

```python
from tvm import relay
from tvm.relay import testing
import tvm

# Resnet18 workload
resnet18_mod, resnet18_params = relay.testing.resnet.get_workload(num_layers=18)
resnet18_mod: tvm.IRModule

@tvm.tir.transform.prim_func_pass(opt_level=0)
def print_tir(f, mod, ctx):
    print(f)

with tvm.transform.PassContext(
        opt_level=3, config={"tir.add_lower_pass": [(3, print_tir)]}
    ):
        lib = relay.build(resnet18_mod, target='c')

```


# 如何更加优雅的写tiling?

## 如果在TVM中: 

如果是手写tiling的话,最麻烦的一点就是每次都需要手动算tile大小,然后开辟出n个for循环进行写操作.
```python
@T.prim_func
def simple_split(a: T.handle) -> None:
  A = T.match_buffer(a, [16])
  for i in T.serial(0, 16):
    with T.block("block"):
      vi = T.axis.remap("S", [i])
      A[vi] = i + 100


def test_simple_split():
  sch = tir.Schedule(simple_split)
  b = sch.get_block("block")
  lps = sch.get_loops(b)
  sch.split(lps[0], [7,10])
  print(sch.mod.script())

# from tvm.script import tir as T
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(a: T.handle) -> None:
        A = T.match_buffer(a, [16], dtype="float32")
        # body
        # with T.block("root")
        for i_0, i_1 in T.grid(7, 10):
            with T.block("block"):
                vi = T.axis.spatial(16, i_0 * 10 + i_1)
                T.where(i_0 * 10 + i_1 < 16)
                T.reads([])
                T.writes([A[vi]])
                A[vi] = i_0 * 10 + i_1 + 100
```

不过tvm的tir中是简化了for循环,也就是无法自定义stride,因为他面向的对象都是cpu/gpu这些的设备. 但是如果对于一些大颗粒算子的dsa来说,最好还是带有stride的for循环比较合理,否则对于一段程序我们需要这样写:
```python
@T.prim_func
def simple_split(a: T.handle) -> None:
  A = T.match_buffer(a, [16])
  chunk_n = 3
  chunk_c = 5
  for n in T.serial(0, compute_segment(16, chunk_n)):
    for c in T.serial(0, compute_segment(32, chunk_c)):
      with T.block("block"):
        vi, vj = T.axis.remap("SS", [n,c])
        A[vi * chunk_n + vj * chunk_c] = 100
```

如果每次都自己控制chunk,那么如果有6d的tensor,也就是6层循环, 那么变量绝对多到难以控制的程度.

如果可以这样写肯定就舒服多了, 然后关键是就是chunk固定但是length还得每次求, 不过应该是合理一些了:
```python
@T.prim_func
def simple_split(a: T.handle) -> None:
  A = T.match_buffer(a, [16])
  chunk_n = 3
  chunk_c = 5
  for n in T.serial(0, 16, chunk_n):
    for c in T.serial(0, 32, chunk_c):
      with T.block("block"):
        vi, vj = T.axis.remap("SS", [n,c])
        with T.let(length_n, min(chunk_n, 16 - vi)):
          with T.let(length_c, min(chunk_c, 32 - vj)):
            A[vi + vj] = 100
```

但是还是有一点非常麻烦,那就是求tir中定义一个变量就需要声明他的作用域,那么对于真的多层的循环复杂逻辑肯定还是很麻烦的.

## 如果在CSharp中:

我的想法是在csharp中基于Linq实现两套写法, 那些shape之类的可能还是没法用expr进行lazy的运算,因为一旦那样就很难用linq语法, 写起来就复杂很多.

### 1. 适配老架构的segment的方式

之前因为是cpp的语法,所以要实现一套基于Enumerable的dsl还是比较麻烦,所以for循环之类的刻板代码比较多, 目前我也先支持这种写法. 通过linq拆分出segment之后构造segment 4d然后进行计算. csharp的linq可以再嵌套linq所以不用担心复杂的逻辑无法处理, 最后返回出expr即可.

```csharp
T.PrimFunc("TileLoadStore").Body(
  (from item in glb.items
    let mmu = item.Value
    select I.MmuConf((MMU_CONF_WIDTH)mmu.width, mmu.id, mmu.start_bank, mmu.start_depth, mmu.depth)).ToSequential(),
  (from glb_input_batch in SegmentByChunk(0, glb.last_out_shape[0], input_shape[0])
    from glb_input_channel in SegmentByChunk(0, glb.last_out_shape[1], input_shape[1])
      from glb_input_row in SegmentByChunk(0, glb.last_out_shape[2], input_shape[2])
        from glb_input_column in SegmentByChunk(0, glb.last_out_shape[3], input_shape[3])
          let ofmap = new tensor4d_segment( glb_input_batch.OutputByStride(strides[0]),
                                            glb_input_channel.OutputByStride(strides[1]),
                                            glb_input_row.OutputByStride(strides[2]),
                                            glb_input_column.OutputByStride(strides[3]))
          let ifmap = new tensor4d_segment(glb_input_batch, glb_input_channel, glb_input_row, glb_input_column)
          let c_pp_split_size = (uint)Math.Ceiling(1.0 * glb_input_channel.Length / glb.n_ping_pong_split)
          let in_chan_split = SegmentByChunk((int)glb_input_channel.Start, (int)c_pp_split_size, (int)glb_input_channel.End)
          from inst in in_chan_split.Select(c_pp_split =>
          {
              // load ifmap
              // 再次对c进行切分. 然后更新ifmap中c的segment.
              tensor4d_segment ifmap_pp = new(ifmap[0], c_pp_split, ifmap[2], ifmap[3]);
              // 然后再把ifmap_pp的start全部减去一个base,因为这个segment起始地址是切分后的.
              tensor4d_segment ifmap_pp_glb = glb_tensor_index_shift(ifmap_pp, ifmap);

              bool clear_qarg_ccr = false;
              if (input_type.IsQuantType())
              {
                  // action_updater.update_load_load_qarg(i_pp, ifmap_pp, ifmap_pp_glb, load_type);
                  clear_qarg_ccr = true;
              }

              CcrSet ifmap_pp_ccrset = new(0, 0, 0);
              tensor4d_segment ifmap_pp_ld_glb = glb_tensor_index_shift(ifmap_pp, ifmap_pp);
              // action_updater.update_load_if(ifmap_pp_ccrset, ifmap_pp, ifmap_pp_glb, ifmap_pp_ld_glb, load_type, dt_bfloat16, false, i_pp, clear_qarg_ccr);

              segment oc_pp_split = c_pp_split.OutputByStride(strides[1]);
              tensor4d_segment ofmap_pp = new(ofmap[0], oc_pp_split, ofmap[2], ofmap[3]);
              tensor4d_segment ofmap_pp_glb = glb_tensor_index_shift(ofmap_pp, ofmap);

              if (output_type.IsQuantType())
              {
                  // action_updater.update_load_store_qarg(i_pp, ofmap_pp, ofmap_pp_glb, store_type);
              }

              tensor4d_segment ofmap_pp_st = new(ofmap_pp.Segments);
              for (int i = 0; i < 4; i++) { ofmap_pp_st[i] = ofmap_pp_st[i] with { Start = ofmap_pp_st[i].Start * (uint)strides[i] }; }
              tensor4d_segment ofmap_pp_st_glb = glb_tensor_index_shift(ofmap_pp_st, ifmap_pp);
              // action_updater.update_store_t(item_name::ifmap, ofmap_pp, ofmap_pp_glb, ofmap_pp_st_glb, store_type, of_buf_num, i_pp, i_pp);
              return new Var("1", AnyType.Default);
          })
    select inst).ToSequential()
);
```


### 2. 输入glb_tensor,可以通过索引的方式进行tiling, 而后构造指令.

这个glb_tensor应该是一个可以多层级的数据结构,比如当前的sub_tensor可以求关于上一层tensor的地址偏移,然后也可以求关于父节点的内存偏移. 然后基于之前segment的逻辑,就可以把写出一个优雅的tensor处理逻辑.

```csharp
from in_seg in compute_segment(N,chunk_n)
  from ic_seg in compute_segment(C,chunk_c)
    from ih_seg in compute_segment(H,chunk_h)
      from iw_seg in compute_segment(W,chunk_w)
        let sub_input = input[in_seg, ic_seg, ih_seg, iw_seg];
        from cpp_seg in compute_segment(ic_seg,pp_chunk)
          let ping_input = sub_input[..,cpp_seg,..,..]
          // ! can't direct add Expr in here.
          select I.Load(ping_input.addr,ping_input.stride,....)
```



