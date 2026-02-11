---
title: Cute概念速通
mathjax: true
toc: true
categories:
  - 推理框架
date: 2026-02-03 22:29:32
tags:
- Layout
---

这篇文章将快速的介绍Cute中的一些基本概念、 layout algorithm、swizzle等，具体代码位于[cute概念速通](https://github.com/zhen8838/handson-polyhedral/blob/main/16_cute_concepts.ipynb)。

<!--more-->


```python
import pycute as cute
import numpy as np
import itertools
import ml_dtypes
np.set_printoptions(linewidth=100)

```

# Layout Component

先介绍Layout本身， 是由Shape,Stride组成的一个结构体， 其中Shape/Stride都是一个可嵌套的向量。

他们组成的Layout是一个从index到offset的函数，它的数学表示为：
```
Layout(coord) = offset
```

实际计算过程如下：
```
Coord ∘ Shape => Index
Index ∘ Stride => Offset
```






```python
layout = cute.Layout((2, 3), (3, 6))
coord = (1, 2)
print(f'offset: {layout(coord)}')
```

    offset: 15


## Mode

layout可以通过mode取其中的每个元素


```python
print(f'mode 0: {layout[0]}')
print(f'mode 1: {layout[1]}')

```

    mode 0: 2:3
    mode 1: 3:6


## Domain & Codomain

domain是layout的索引空间，codomain是layout的访问空间。


```python
print(f'domain: {layout.size()}')
print(f'codomain: {layout.cosize()}')
```

    domain: 6
    codomain: 16


# Complement

先解释一下为什么需要补集，观察上面，可以发现Layout的domain != codomain，这就意味着在内存中某些位置是不能被访问到的。 而补集就是这些不能被访问到的位置所构成的layout。


Complement是许多layout algorithm中常用的一个操作。 他的主要原理是通过计算codomain，再分解当前的Shape/Stride得到无法访问到的位置，剩余部分所构成的原layout就是补集。



```python
comp = cute.complement(layout)
comp
```




    Layout(3,1)




```python
full = cute.make_layout(layout, comp) # layout ⋆ comp(layout) 得到当前全集 
print(full)

assert full.size() == full.cosize() # size 和 cosize 相等，说明是全集
```

    ((2, 3), 3):((3, 6), 1)



```python
# 补集的逻辑很简单，就是从最大N开始不停的除shape*stride
comp = cute.complement(layout) 
print(comp) 
comp = cute.complement(layout, 54) # 也可以设置更大codom来构建补集
print(comp) # 也就是说如果补的len实际上得先看内部有个多少空洞， 然后最后再添加一个能重复到max idx的。

print(cute.size(layout), cute.cosize(layout))
print(cute.size(comp), cute.cosize(comp))
```

    3:1
    (3, 3):(1, 18)
    6 16
    9 39


# Composition

C = A ∘ B 表示用B的codomain映射到A的domain上, 这里需要注意是B的codomain需要和A的domain匹配，并且C的domain是C的domain，而codomain是A的domain。

```
C(coord) = A(B(coord))
```


composition后的取offsetC等价于如下的公式。
```
B(coord) => offsetB
index2crd(offsetB, A.shape) => coordA
A(coordA) => offsetC
```

这里还有一个细节需要注意, 实际index2crd的过程是从左向右进行拆分的，这其实和cute中默认column major的布局对应。这也就是为什么cute中通常把tile的内层放到最左边，而剩余部分放到右边。这样composite出来的结果索引和cute中默认的column major布局是一致的。



```python
x = cute.Layout((8), (4))
y = cute.Layout((4), (1))

print('x ∘ y:', cute.composition(x, y)) # 此时输出的dom变成了y的dom
print('y ∘ x:', cute.composition(y, x)) # 此时输出的dom变成了x的dom
```

    x ∘ y: 4:4
    y ∘ x: 8:4


# Divide

他的公式如下
```
C = A ∘ (B ⋆ comp(B, size(A))). 
```
扩展B的codomain到A的domain，然后进行composite。

实际上divide通常是用于实现tiling， 设想B作为一个固定的inner factor， 将他补到A的domain， 那么(B * restB)匹配A的domain，此时restB就相当于outer factor。 最后再进行一次composite， 就得到了C，此时C的domain由(inner factor,outer  factor)组成, 也就是呈现了tiled A。



```python
a = cute.Layout((128, 32), (32, 1))
print('original:', a[0]) # 对于128，采用不同的factor去divide，会得到不同的结果
print('divide by 8:', cute.logical_divide(a[0], cute.Layout(8)))
print('divide by 4:', cute.logical_divide(a[0], cute.Layout(4)))
```

    original: 128:32
    divide by 8: (8, 16):(32, 256)
    divide by 4: (4, 32):(32, 128)


## zipped_divide/tiled_divide

不同divide的区别在于返回的layout的group方式不同:


```python
print('zipped_divide:', cute.zipped_divide(a, (8, 4)))
print('tiled_divide:', cute.tiled_divide(a, (8, 4)))
```

    zipped_divide: ((8, 4), (16, 8)):((32, 1), (256, 4))
    tiled_divide: ((8, 4), 16, 8):((32, 1), 256, 4)


# Product

公式如下：
```
C = A ⋆ (comp(A, size(A) * cosize(B)) ∘ B)
```

如果是divide是拆分layout A，那么product就是重复layout A。保持A本身不变，然后整体是需要把A重复B次，所以需要先补到(size A) * (cosize B)， 再配合一个composite让B映射每一个A上。最后再拼接原始的A，得到C具备 (size A) * (size B) 的domain。




```python
ta = cute.Layout((2, 2), (4, 1))
tb = cute.Layout(6, 1)
max_idx = ta.size() * tb.cosize()
comp_ta = cute.complement(ta, max_idx)
print('max_idx:', max_idx, 'complement:', comp_ta)
print("composite", cute.composition(comp_ta, tb))
tc = cute.make_layout(ta, cute.composition(comp_ta, tb))
print('product:', tc)
```

    max_idx: 24 complement: (2, 3):(2, 8)
    composite (2, 3):(2, 8)
    product: ((2, 2), (2, 3)):((4, 1), (2, 8))


## Zipped Product & Tiled Product

同样也是把inner part放到左边，并且也是在group方式上存在区别。


```python
print('zipped:', cute.zipped_product(a, (8, 4)))
print('tiled:', cute.tiled_product(a, (8, 4)))
```

    zipped: ((128, 32), (8, 4)):((32, 1), (1, 32))
    tiled: ((128, 32), 8, 4):((32, 1), 1, 32)


## Block Rroduct

如果是logical product，那么最后的domain就是(A, B)，按照cute默认的迭代方式进行访问，就是先访问A的所有元素，然后按B的方式访问扩展后的A元素。

但是有没有可能，通过调整domain，在默认的迭代方式下实现不同的访问模式呢？ 比如我们有一个小的2维矩阵tiledA(m,n)，它匹配硬件计算的粒度，但是在内存中他是以A(3m, 4n)的形式存储的。 那么我们希望在默认的迭代方式下， 先将A的n维度访问完，再访问m维度，这个需求就需要blocked product来实现。

其实blocked product的本质就是调整logical product的domain顺序， 让inner part的维度在左边， outer part的维度在右边。 这样在默认的迭代方式下，就实现了先访问inner part，再访问outer part的效果。而实际上在内存上的元素顺序并没有任何改变。


```python
def hier_zip(layoutA, layoutB):
  assert len(layoutA) == len(layoutB)
  return cute.make_layout(itertools.chain((cute.make_layout(layoutA[i], layoutB[i]) for i in range(0, len(layoutA)))))

def blocked_product(block, tiler):
  res = cute.logical_product(block, tiler)
  return hier_zip(res[0], res[1])

a = cute.Layout((2, 5), (5, 1))
b = cute.Layout((3, 4), (1, 3))
logical_producted = cute.logical_product(a, b)
blocked_producted = blocked_product(a, b)  # 实际上就是把logical product拆分且zip
print("logical_producted", logical_producted)
print("blocked_producted", blocked_producted)
```

    logical_producted ((2, 5), (3, 4)):((5, 1), (10, 30))
    blocked_producted ((2, 3), (5, 4)):((5, 10), (1, 30))


我写了一个print_offsets的函数，打印按默认的访问顺序下，layout所映射的offset的情况， 用于展示blocked product的效果。 首先是看logical product的情况，这里crd默认也是按col major生成的，所以layout a会按列呈现在offset矩阵中：


```python
def print_offsets(layout: cute.Layout):
  rk = len(layout)
  shape = [layout[i].size() for i in range(rk)]
  arr = np.zeros(shape, dtype=np.int32)
  for crd in itertools.product(*[range(shape[i]) for i in range(rk)]):
    arr[crd] = layout(crd)
  print(arr)

print_offsets(logical_producted)
```

    [[  0  10  20  30  40  50  60  70  80  90 100 110]
     [  5  15  25  35  45  55  65  75  85  95 105 115]
     [  1  11  21  31  41  51  61  71  81  91 101 111]
     [  6  16  26  36  46  56  66  76  86  96 106 116]
     [  2  12  22  32  42  52  62  72  82  92 102 112]
     [  7  17  27  37  47  57  67  77  87  97 107 117]
     [  3  13  23  33  43  53  63  73  83  93 103 113]
     [  8  18  28  38  48  58  68  78  88  98 108 118]
     [  4  14  24  34  44  54  64  74  84  94 104 114]
     [  9  19  29  39  49  59  69  79  89  99 109 119]]


但是如果采用blocked product，则是按`[m * 3, n * 4]`这样的shape来采样，当我们访问A的n维度时，是优先访问完一个tile的n维度，然后跳到下一个tile的n维度进行访问：


```python
print_offsets(blocked_producted)
```

    [[  0   1   2   3   4  30  31  32  33  34  60  61  62  63  64  90  91  92  93  94]
     [  5   6   7   8   9  35  36  37  38  39  65  66  67  68  69  95  96  97  98  99]
     [ 10  11  12  13  14  40  41  42  43  44  70  71  72  73  74 100 101 102 103 104]
     [ 15  16  17  18  19  45  46  47  48  49  75  76  77  78  79 105 106 107 108 109]
     [ 20  21  22  23  24  50  51  52  53  54  80  81  82  83  84 110 111 112 113 114]
     [ 25  26  27  28  29  55  56  57  58  59  85  86  87  88  89 115 116 117 118 119]]


本质上两个layout表达的offset是相同的，区别在于使用`cute默认的迭代顺序`下他们的行为，这一点其实是十分重要的，这会令logical/blocked layout进行composite的时候得到完全不同的结果。

当然cute layout的自由度还是很大的，比如对于logical product的结果，我们也可以自行按block coord来采样，再采样tile内部，也是可以得到相同的offset序列的：


```python
tile = logical_producted[0]
block = logical_producted[1]
for block_crd in itertools.product(*[range(mode.size()) for mode in block]):
  block_offset = block(block_crd)
  array = np.zeros(tile.shape, dtype=np.int32)
  for tile_crd in itertools.product(*[range(mode.size()) for mode in tile]):
    offset = tile(tile_crd)
    array[tile_crd] = offset + block_offset
  print(f"block {block_crd}:\n", array)
```

    block (0, 0):
     [[0 1 2 3 4]
     [5 6 7 8 9]]
    block (0, 1):
     [[30 31 32 33 34]
     [35 36 37 38 39]]
    block (0, 2):
     [[60 61 62 63 64]
     [65 66 67 68 69]]
    block (0, 3):
     [[90 91 92 93 94]
     [95 96 97 98 99]]
    block (1, 0):
     [[10 11 12 13 14]
     [15 16 17 18 19]]
    block (1, 1):
     [[40 41 42 43 44]
     [45 46 47 48 49]]
    block (1, 2):
     [[70 71 72 73 74]
     [75 76 77 78 79]]
    block (1, 3):
     [[100 101 102 103 104]
     [105 106 107 108 109]]
    block (2, 0):
     [[20 21 22 23 24]
     [25 26 27 28 29]]
    block (2, 1):
     [[50 51 52 53 54]
     [55 56 57 58 59]]
    block (2, 2):
     [[80 81 82 83 84]
     [85 86 87 88 89]]
    block (2, 3):
     [[110 111 112 113 114]
     [115 116 117 118 119]]


## Raked product

Raked product和blocked product类似，也是调整logical product的domain顺序，不过是把outer part的维度放到左边， inner part的维度放到右边。 这样在默认的迭代方式下，就实现了先访问outer part，再访问inner part的效果。而实际上在内存上的元素顺序并没有任何改变。


```python
def raked_product(block, tiler):
  res = cute.logical_product(block, tiler)
  return hier_zip(res[1], res[0])

a = cute.Layout((2, 5), (5, 1))
b = cute.Layout((3, 4), (1, 3))
logical_producted = cute.logical_product(a, b)
raked_producted = raked_product(a, b) # 实际上就是把logical layout进行了拆分zip, 不过顺序反过来了。
print('logical_producted', logical_producted)
print('raked_producted', raked_producted)
```

    logical_producted ((2, 5), (3, 4)):((5, 1), (10, 30))
    raked_producted ((3, 2), (4, 5)):((10, 5), (30, 1))


它同样也是先遍历完A的某一个维度，只是遍历维度内部的顺序从inner优先变成了outer优先。


```python
print_offsets(raked_producted)
```

    [[  0  30  60  90   1  31  61  91   2  32  62  92   3  33  63  93   4  34  64  94]
     [ 10  40  70 100  11  41  71 101  12  42  72 102  13  43  73 103  14  44  74 104]
     [ 20  50  80 110  21  51  81 111  22  52  82 112  23  53  83 113  24  54  84 114]
     [  5  35  65  95   6  36  66  96   7  37  67  97   8  38  68  98   9  39  69  99]
     [ 15  45  75 105  16  46  76 106  17  47  77 107  18  48  78 108  19  49  79 109]
     [ 25  55  85 115  26  56  86 116  27  57  87 117  28  58  88 118  29  59  89 119]]


# Inverse

## Right Inverse

它的数学表示为：

Layout_Rinv(Layout(index)) = index

原始的layout是一个从index到offset的函数，所以right inverse就是得到一个layout invese可以从offset 到 index。

```
crd2idx(coord)
      |
      v
    index
      |
      v
  F_L(index) -> offset
      ∧             |
      |             v
   index <- F_Linv(offset)
```




```python
layout = cute.Layout((32, 64), (64, 1))
x = cute.crd2idx((3, 4), layout.shape)
print('layout:', layout, "index:", x, "offset:", layout(3, 4))  # 3 * 64 + 4
```

    layout: (32, 64):(64, 1) index: 131 offset: 196



```python
layout_inv = cute.right_inverse(layout)
x = layout_inv(cute.idx2crd(196, layout_inv.shape))
print('layout_inv', layout_inv, "index:", 196, "offset:", x)
```

    layout_inv (64, 32):(32, 1) index: 196 offset: 131


所以其实它比较大的用处是, 这样可以快速找到原来的一个coord所对应的index

F_Linv(F_L(coord)) = index


```python
crd = (3, 4)
index = layout_inv(layout(*crd))
assert layout(index) == layout(*crd)
print(f"we find the coord {crd}'s index is:", index)
```

    we find the coord (3, 4)'s index is: 131


# Thread Value layout

我的理解是这样，thread value layout(tv layout)本质上是提供了一个在thread level的sharding tensor在各个thread上访问local data的方式。 在cute的一个教程中，是这样描述tv layout的使用的：
```
    (16,256)   :  (2048,1)
     ~~~~~~        ~~~~~~
        |             |        Tiled/Composed with TV Layout
        |             |    
        |             |    o   ((32,4),(8,4)):((128,4),(16,1))
        V             V         
~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~ 
((32,4), (8,4))  :  ((4,8192),(1,2048))
    |      |
    |      `--------> per thread fragment
    |
Thread Block
  Shape

Sliced to Thread local sub-tensor (a (4,8) tile):  tidfrgA[(tidx, None)]
```

将一个`(16,256)`的tensor，按32线程，4warp的方式sharding到每个thread上， 每个thread访问一个`(4,8)`的local tile。 此时tv layout的作用就是提供一个方式，让每个thread可以访问到自己local tile的数据。

注意到这里`32线程，4warp的方式`是按照`(32,4)`的shape来描述的，这是因为cute的index to coord是按照从左向右拆分的，所以需要把thread维度放到左边， warp维度放到右边。 这样在默认的迭代方式下，`tidfrgA[(tidx, None)]`的`tidx`才能被正确映射到thread，warp上。

在`cutlass/python/CuTeDSL/cutlass/cute/core.py#L5093`中有一个make tv layout实现，我们来拆解一下：



```python
# 1. 首先cute的逻辑不是对原始tensor做拆分，而是对thread和value的layout做组合来构建tv layout
thr_layout = cute.Layout((4, 32), (32, 1))  # 32 threads, 4warps
val_layout = cute.Layout((4, 8), (8, 1))  # 每个thread访问4行， 每行8个元素

# 2. 此时thr layout作为block，val layout作为tiler，由于 raked product的结果是tiler优先
# 得到((val_m, thr_m), (val_n, thr_n)) -> [1~M*N]的映射
layout_mn = raked_product(thr_layout, val_layout)

# 3. 然后构建tv domain，即(thr_size, val_size)，这里thr_size=128, val_size=32
# 得到 (threads, values) -> [1~M*N]的映射
thr_size, val_size = cute.size(thr_layout), cute.size(val_layout)
tv_domain = cute.Layout((thr_size, val_size))

# 4. layout_mn的右逆是得到了 [1~M*N] -> ((val_m, thr_m), (val_n, thr_n))，
# 进行composite： (threads, values) -> [1~M*N] -> [1~M*N] -> ((val_m, thr_m), (val_n, thr_n))
# 得到 (threads, values) -> ((val_m, thr_m), (val_n, thr_n))的映射
layout_tv = cute.composition(cute.right_inverse(layout_mn), tv_domain)
tiler_mn = tuple(cute.product(s) for s in layout_mn.shape)

print('tiler_mn:', tiler_mn)
print('layout_tv:', layout_tv)
```

    tiler_mn: (16, 256)
    layout_tv: ((32, 4), (8, 4)):((128, 4), (16, 1))


至此得到了layout tv，但实际上它只映射到一个logical的(M,N)上。使用一个实际上的tiled tensor与它进行compose后，可以用layout tv去采样tiled tensor内的元素。 回想之前设计的value layout是`((4, 8), (8, 1))`，那么去取一个row major矩阵就可以是8个元素连续读，而取一个col major的矩阵就是4个元素连续读：


```python
tiledA = cute.Layout((16, 256), (512, 1)) # row major tiled tensor
print("row major   tiledA: ", cute.composition(tiledA, layout_tv)) # ... (8, 4) : ... (1,512)
tiledA = cute.Layout((16, 256), (1, 512)) # colum marjor tiled tensor
print("colum major tiledA: ", cute.composition(tiledA, layout_tv)) # ... (8, 4) : ... (512,1)
```

    row major   tiledA:  ((32, 4), (8, 4)):((8, 2048), (1, 512))
    colum major tiledA:  ((32, 4), (8, 4)):((4096, 4), (512, 1))


# Swizzle

Swizzle的本质上是将offset进行重映射到新的offset上，cute这里选择将offset的值拆分为行和列部分的组合，通过移动行和列所对应的bits实现行列交错， 从而将逻辑上同一列的地址映射到物理的不同列上。

```
0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                 ^-^       ^-^      B 是要移动的位个数
                              ^--^  M 保持不变的位个数
                   ^---------^      S 是移动的距离
                                      (pos shifts YYY to the right, neg shifts YYY to the left)
e.g. Given
0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
the result is
0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY
```

简单来说， Swizzle这里三个参数可理解为M是保持不变的位个数，对应连续读取的元素个数。S是移动的距离，看做逻辑上的列数。B是要移动的位个数，看作逻辑上的行数。 我下面设计了一个visualize_bank_distribution函数，用于展示swizzle前后，线程访问内存bank的分布情况。


```python
def visualize_bank_distribution(access_indices_per_thread, swizzle=None, title="", cols=32):
    # 1) 先收集所有物理地址
    phys_addrs = []
    for tid, indices in access_indices_per_thread.items():
        for ptr in indices:
            phys_ptr = swizzle(ptr) if swizzle else ptr
            phys_addrs.append(phys_ptr)

    if not phys_addrs:
        print("No accesses.")
        return

    # 2) 以访问的最大地址决定可视化的“memory”规模
    max_addr = max(phys_addrs)
    total_elems = max_addr + 1
    rows = (total_elems + cols - 1) // cols

    # 3) 用 cute.Layout 表示 bank 的二维布局 (row, col) -> linear addr
    bank_layout = cute.Layout((rows, cols), (cols, 1))

    # 4) 建立二维表记录每个地址的线程集合
    table = [[set() for _ in range(cols)] for _ in range(rows)]
    for tid, indices in access_indices_per_thread.items():
        for ptr in indices:
            phys_ptr = swizzle(ptr) if swizzle else ptr
            r, c = cute.idx2crd(phys_ptr, bank_layout.shape, bank_layout.stride)
            table[r][c].add(tid)

    # 5) 打印整个 memory 的 bank 视图
    print(f"\n[{title}] | " + (f"Swizzle: {swizzle}" if swizzle else "No Swizzle"))
    header = "      " + "".join(f"{b:02d} " for b in range(cols))
    print(header)
    print("      " + "-" * (cols * 3))

    for r in range(rows):
        row_cells = []
        for c in range(cols):
            if table[r][c]:
                row_cells.append("/".join(f"{t:02d}" for t in sorted(table[r][c])))
            else:
                row_cells.append("  ")
        print(f"R{r:02d} | " + " ".join(f"{x:>2}" for x in row_cells))

    bank_threads = [set() for _ in range(cols)]
    for r in range(rows):
        for c in range(cols):
            bank_threads[c].update(table[r][c])
    max_conflicts = max(len(s) for s in bank_threads)
    print(f"最大冲突深度: {max_conflicts}")
```

假设我们有一个32x64的f32矩阵，采用row-major布局。 32个线程，每个线程访问一列元素，此时这个32个线程刚好访问同一个bank的不同地址，产生严重的bank conflict：


```python
def my_access_pattern(layout, threads, v_size):
    access_map = {}
    for tid in range(threads):
        access_map[tid] = [layout(tid, v) for v in range(v_size)]
    return access_map

num_threads = 32
vec_size = 1
a_layout = cute.Layout((32, 64), (64, 1))

thread_accesses = my_access_pattern(a_layout, num_threads, vec_size)

visualize_bank_distribution(thread_accesses, swizzle=None, title="Row-Major Vector Read")
```

    
    [Row-Major Vector Read] | No Swizzle
          00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
          ------------------------------------------------------------------------------------------------
    R00 | 00                                                                                             
    R01 |                                                                                                
    R02 | 01                                                                                             
    R03 |                                                                                                
    R04 | 02                                                                                             
    R05 |                                                                                                
    R06 | 03                                                                                             
    R07 |                                                                                                
    R08 | 04                                                                                             
    R09 |                                                                                                
    R10 | 05                                                                                             
    R11 |                                                                                                
    R12 | 06                                                                                             
    R13 |                                                                                                
    R14 | 07                                                                                             
    R15 |                                                                                                
    R16 | 08                                                                                             
    R17 |                                                                                                
    R18 | 09                                                                                             
    R19 |                                                                                                
    R20 | 10                                                                                             
    R21 |                                                                                                
    R22 | 11                                                                                             
    R23 |                                                                                                
    R24 | 12                                                                                             
    R25 |                                                                                                
    R26 | 13                                                                                             
    R27 |                                                                                                
    R28 | 14                                                                                             
    R29 |                                                                                                
    R30 | 15                                                                                             
    R31 |                                                                                                
    R32 | 16                                                                                             
    R33 |                                                                                                
    R34 | 17                                                                                             
    R35 |                                                                                                
    R36 | 18                                                                                             
    R37 |                                                                                                
    R38 | 19                                                                                             
    R39 |                                                                                                
    R40 | 20                                                                                             
    R41 |                                                                                                
    R42 | 21                                                                                             
    R43 |                                                                                                
    R44 | 22                                                                                             
    R45 |                                                                                                
    R46 | 23                                                                                             
    R47 |                                                                                                
    R48 | 24                                                                                             
    R49 |                                                                                                
    R50 | 25                                                                                             
    R51 |                                                                                                
    R52 | 26                                                                                             
    R53 |                                                                                                
    R54 | 27                                                                                             
    R55 |                                                                                                
    R56 | 28                                                                                             
    R57 |                                                                                                
    R58 | 29                                                                                             
    R59 |                                                                                                
    R60 | 30                                                                                             
    R61 |                                                                                                
    R62 | 31                                                                                             
    最大冲突深度: 32


根据之前对swizzle的理解，它的参数实际上在描述一个逻辑的访问矩阵，要将行和列进行交错实现conflict free。此时一个元素占一个bank，所以M=0 (2^0=1)。然后是访问32行时，每一行都交错开，B=5 (2^5=32)。 最后S=6 (2^6=64)，表示每行有64个元素。 


```python
visualize_bank_distribution(thread_accesses, swizzle=cute.Swizzle(5, 0, 6), title="With Swizzle Optimization")
```

    
    [With Swizzle Optimization] | Swizzle: SW_5_0_6
          00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
          ------------------------------------------------------------------------------------------------
    R00 | 00                                                                                             
    R01 |                                                                                                
    R02 |    01                                                                                          
    R03 |                                                                                                
    R04 |       02                                                                                       
    R05 |                                                                                                
    R06 |          03                                                                                    
    R07 |                                                                                                
    R08 |             04                                                                                 
    R09 |                                                                                                
    R10 |                05                                                                              
    R11 |                                                                                                
    R12 |                   06                                                                           
    R13 |                                                                                                
    R14 |                      07                                                                        
    R15 |                                                                                                
    R16 |                         08                                                                     
    R17 |                                                                                                
    R18 |                            09                                                                  
    R19 |                                                                                                
    R20 |                               10                                                               
    R21 |                                                                                                
    R22 |                                  11                                                            
    R23 |                                                                                                
    R24 |                                     12                                                         
    R25 |                                                                                                
    R26 |                                        13                                                      
    R27 |                                                                                                
    R28 |                                           14                                                   
    R29 |                                                                                                
    R30 |                                              15                                                
    R31 |                                                                                                
    R32 |                                                 16                                             
    R33 |                                                                                                
    R34 |                                                    17                                          
    R35 |                                                                                                
    R36 |                                                       18                                       
    R37 |                                                                                                
    R38 |                                                          19                                    
    R39 |                                                                                                
    R40 |                                                             20                                 
    R41 |                                                                                                
    R42 |                                                                21                              
    R43 |                                                                                                
    R44 |                                                                   22                           
    R45 |                                                                                                
    R46 |                                                                      23                        
    R47 |                                                                                                
    R48 |                                                                         24                     
    R49 |                                                                                                
    R50 |                                                                            25                  
    R51 |                                                                                                
    R52 |                                                                               26               
    R53 |                                                                                                
    R54 |                                                                                  27            
    R55 |                                                                                                
    R56 |                                                                                     28         
    R57 |                                                                                                
    R58 |                                                                                        29      
    R59 |                                                                                                
    R60 |                                                                                           30   
    R61 |                                                                                                
    R62 |                                                                                              31
    最大冲突深度: 1



上面的swizzle是直接异或，对于理解起来不怎么直观，所以我准备了一个更加符合直觉的图示。 实际上我们可以把swizzle还原到一个cute layout，他表示的是一个逻辑上的box，通过将原始tensor offset映射到这个box上，再通过box的coord得到逻辑行和列号（irow，icol），对icol进行异或得到xicol, 保证了**不同行的相同列被映射到不同的列**上。 再使用`(irow,xicol)`得到的index去访问shared memory bank， 同样通过idx2crd得到`(prow,pcol)`，即可发现所有的pcol的值都不相同， 实现了bank conflict free。
```
┌─────────────────────┐          ┌──────────────────────┐          ┌──────────────────────┐
│   TENSOR LAYOUT     │          │   SWIZZLE LAYOUT     │          │  BANK LAYOUT         │
│   stride=64         │          │                      │          │  (32 banks)          │
│   32x64 row-major   │          │  shape: (1,64,32)    │          │                      │
│                     │          │  stride:(1,1,64)     │          │  Bank distribution   │
│ tid=0→addr 0        │--------->│                      │--------->│                      │
│ tid=1→addr 64       │  idx2crd │  (base,icol,irow)    │ idx2crd  │  tid=0→Bank 0        │
│ tid=2→addr 128      │          │  icol%width ^ irow   │          │  tid=1→Bank 1        │
│ tid=3→addr 192      │          │                      │          │  tid=2→Bank 2        │
│ tid=4→addr 256      │          │  (base,xicol,irow)   │          │  tid=3→Bank 3        │
│ tid=5→addr 320      │          │         │            │          │  tid=4→Bank 4        │
│ tid=6→addr 384      │          │         v            │          │  tid=5→Bank 5        │
│ tid=7→addr 448      │          │     phy_offset       │          │  tid=6→Bank 6        │
│                     │          └──────────────────────┘          │  tid=7→Bank 7        │
└─────────────────────┘                                            │                      │
                                                                   │ 已映射到不同 bank      │
                                                                   │ (冲突深度 = 1)        │
                                                                   │ 不存在 bank conflict  │
                                                                   └──────────────────────┘
```


```python

def decompose_swizzle_mapping(access_indices_per_thread, swizzle: cute.Swizzle):
  base = 2 ** swizzle.base
  width = 2 ** swizzle.shift
  height = 2 ** swizzle.bits
  swizzle_layout = cute.Layout((base, width, height), (1, base, base * width))

  max_offset = max([offset for _, indices in access_indices_per_thread.items() for offset in indices])
  bank_layout = cute.Layout((32, (max_offset + 32) // 32), (1, 32))

  for tid, indices in access_indices_per_thread.items():
      accesses = []
      for offset in indices:
        (_, icol, irow) = cute.idx2crd(offset, swizzle_layout.shape)
        micol = icol % width # NOTE abs(shift) >= bits
        xicol = micol ^ irow
        
        phy_offset = swizzle_layout(0, xicol, irow)
        (pcol, prow) = cute.idx2crd(phy_offset, bank_layout.shape)
        accesses.append(f'(R{prow:02d}, {pcol:02d})')
      print(f'T{tid:02d} accesses: {str.join(", ", accesses)}')


decompose_swizzle_mapping(thread_accesses, cute.Swizzle(5, 0, 6))
```

    T00 accesses: (R00, 00)
    T01 accesses: (R02, 01)
    T02 accesses: (R04, 02)
    T03 accesses: (R06, 03)
    T04 accesses: (R08, 04)
    T05 accesses: (R10, 05)
    T06 accesses: (R12, 06)
    T07 accesses: (R14, 07)
    T08 accesses: (R16, 08)
    T09 accesses: (R18, 09)
    T10 accesses: (R20, 10)
    T11 accesses: (R22, 11)
    T12 accesses: (R24, 12)
    T13 accesses: (R26, 13)
    T14 accesses: (R28, 14)
    T15 accesses: (R30, 15)
    T16 accesses: (R32, 16)
    T17 accesses: (R34, 17)
    T18 accesses: (R36, 18)
    T19 accesses: (R38, 19)
    T20 accesses: (R40, 20)
    T21 accesses: (R42, 21)
    T22 accesses: (R44, 22)
    T23 accesses: (R46, 23)
    T24 accesses: (R48, 24)
    T25 accesses: (R50, 25)
    T26 accesses: (R52, 26)
    T27 accesses: (R54, 27)
    T28 accesses: (R56, 28)
    T29 accesses: (R58, 29)
    T30 accesses: (R60, 30)
    T31 accesses: (R62, 31)


下面再看一个例子，32x48的f32矩阵，依旧row major。 当进行load matrix，每一行load 4个元素(16Byte)，此时8个线程就能组成一个transaction，同样也会访问同一个bank的不同地址，产生严重的bank conflict：


```python
num_threads = 8
vec_size = 4
a_layout = cute.Layout((32, 64), (64, 1))

thread_accesses = my_access_pattern(a_layout, num_threads, vec_size)

visualize_bank_distribution(thread_accesses, swizzle=None, title="32x64 Row-Major vec4 (No Swizzle)")
```

    
    [32x64 Row-Major vec4 (No Swizzle)] | No Swizzle
          00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
          ------------------------------------------------------------------------------------------------
    R00 | 00 00 00 00                                                                                    
    R01 |                                                                                                
    R02 | 01 01 01 01                                                                                    
    R03 |                                                                                                
    R04 | 02 02 02 02                                                                                    
    R05 |                                                                                                
    R06 | 03 03 03 03                                                                                    
    R07 |                                                                                                
    R08 | 04 04 04 04                                                                                    
    R09 |                                                                                                
    R10 | 05 05 05 05                                                                                    
    R11 |                                                                                                
    R12 | 06 06 06 06                                                                                    
    R13 |                                                                                                
    R14 | 07 07 07 07                                                                                    
    最大冲突深度: 8


现在开始计算Swizzle的参数。首先4个元素保持连续，因此M=2 (2^2=4)。需要让每一行的开头位于不同位置，刚好当前tensor stride为64，可以令此时4个元素组成的16个单元作为一行(宽64)，因此S=4 (2^4=16)。最后需要访问8行，因此B=3 (2^3=8)：


```python
visualize_bank_distribution(thread_accesses, swizzle=cute.Swizzle(3, 2, 4),
                           title="32x64 Row-Major vec4")
```

    
    [32x64 Row-Major vec4] | Swizzle: SW_3_2_4
          00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
          ------------------------------------------------------------------------------------------------
    R00 | 00 00 00 00                                                                                    
    R01 |                                                                                                
    R02 |             01 01 01 01                                                                        
    R03 |                                                                                                
    R04 |                         02 02 02 02                                                            
    R05 |                                                                                                
    R06 |                                     03 03 03 03                                                
    R07 |                                                                                                
    R08 |                                                 04 04 04 04                                    
    R09 |                                                                                                
    R10 |                                                             05 05 05 05                        
    R11 |                                                                                                
    R12 |                                                                         06 06 06 06            
    R13 |                                                                                                
    R14 |                                                                                     07 07 07 07
    最大冲突深度: 1


我们修改上一个例子的配置为32x48的f32矩阵，依旧row major。同样也会访问同一个bank的不同地址，产生4路冲突的bank conflict：


```python
num_threads = 8
vec_size = 4
a_layout = cute.Layout((32, 48), (48, 1))

def my_access_pattern(layout, threads, v_size):
    access_map = {}
    for tid in range(threads):
        access_map[tid] = [layout(tid, v) for v in range(v_size)]
    return access_map

thread_accesses = my_access_pattern(a_layout, num_threads, vec_size)

visualize_bank_distribution(thread_accesses, swizzle=None, title="32x48 Row-Major vec4")
```

    
    [32x48 Row-Major vec4] | No Swizzle
          00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
          ------------------------------------------------------------------------------------------------
    R00 | 00 00 00 00                                                                                    
    R01 |                                                 01 01 01 01                                    
    R02 |                                                                                                
    R03 | 02 02 02 02                                                                                    
    R04 |                                                 03 03 03 03                                    
    R05 |                                                                                                
    R06 | 04 04 04 04                                                                                    
    R07 |                                                 05 05 05 05                                    
    R08 |                                                                                                
    R09 | 06 06 06 06                                                                                    
    R10 |                                                 07 07 07 07                                    
    最大冲突深度: 4


依旧4个元素保持连续，因此M=2 (2^2=4)。但是此时访问的stride为48，并且4个元素为一个单元，再叠加cute Swizzle 2的幂次限制，我们只能选择`4*8=32`或者`4*16=64`作为行宽度。 由于32/64都并不是48的倍数，所以不能通过匹配S的大小刚好令每一行的开头按顺序错开。 这里其实会存在余数，也就是余数所造成icol的周期循环`icol: [(i * 48 % 64) // 4 for i in range(8)] = [0, 12, 8, 4, 0, 12, 8, 4]`，但是最终的`xicol`实际上还需要取决于`irow`的，因此我目前也没有理清楚对于swizzle的影响公式，需要后续进一步研究：


```python
visualize_bank_distribution(thread_accesses, swizzle=cute.Swizzle(3, 2, 4), title="32x48 Row-Major vec4")
```

    
    [32x48 Row-Major vec4] | Swizzle: SW_3_2_4
          00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
          ------------------------------------------------------------------------------------------------
    R00 | 00 00 00 00                                                                                    
    R01 |                                                 01 01 01 01                                    
    R02 |                                                                                                
    R03 |             02 02 02 02                                                                        
    R04 |                                                                         03 03 03 03            
    R05 |                                                                                                
    R06 |                                     04 04 04 04                                                
    R07 |                                                                                     05 05 05 05
    R08 |                                                                                                
    R09 |                                                 06 06 06 06                                    
    R10 |             07 07 07 07                                                                        
    最大冲突深度: 2


所以我们选择更小的行宽度，32为一行。同样也有周期循环 `icol: [(i * 48 % 32) // 4 for i in range(8)] = [0, 4, 0, 4, 0, 4, 0, 4]`， 但是这样最终的xicol反而并不会有重叠：


```python
visualize_bank_distribution(thread_accesses, swizzle=cute.Swizzle(2, 2, 3), title="32x48 Row-Major vec4")
```

    
    [32x48 Row-Major vec4] | Swizzle: SW_2_2_3
          00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
          ------------------------------------------------------------------------------------------------
    R00 | 00 00 00 00                                                                                    
    R01 |                                                             01 01 01 01                        
    R02 |                                                                                                
    R03 |                                     02 02 02 02                                                
    R04 |                                                 03 03 03 03                                    
    R05 |                                                                                                
    R06 |                         04 04 04 04                                                            
    R07 |                                                                                     05 05 05 05
    R08 |                                                                                                
    R09 |             06 06 06 06                                                                        
    R10 |                                                                         07 07 07 07            
    最大冲突深度: 1


下面这个例子好像就没办法做到完全bank conflict free：


```python
num_threads = 8
vec_size = 4
a_layout = cute.Layout((32, 40), (40, 1))
thread_accesses = my_access_pattern(a_layout, num_threads, vec_size)

visualize_bank_distribution(thread_accesses, swizzle=cute.Swizzle(2, 2, 3), title="Swizzle")
```

    
    [Swizzle] | Swizzle: SW_2_2_3
          00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
          ------------------------------------------------------------------------------------------------
    R00 | 00 00 00 00                                                                                    
    R01 |                                     01 01 01 01                                                
    R02 |                                                                         02 02 02 02            
    R03 |                                                             03 03 03 03                        
    R04 |                                                                                                
    R05 |             04 04 04 04                                                                        
    R06 | 05 05 05 05                                                                                    
    R07 |                                                                                     06 06 06 06
    R08 |                                                                         07 07 07 07            
    最大冲突深度: 2

