---
title: Model Driven Optimization
mathjax: true
toc: true
categories:
- 编译器
date: 2024-04-30 11:08:53
tags:
- 后端优化
- 性能建模
---

关于`Model-Driven Optimization For Tensor Computations`论文的阅读笔记.

<!--more-->

# 1. 单内存层级建模

```cpp
for (i2 = 0; i2 < Ni; i2+=Ti1)
  for (j2 = 0; j2 < Nj; j2+=Tj1)
    for (k2 = 0; k2 < Nk; k2+=Tk1)
      for (i1 = 0; i1 < Ti1; i1++)
        for (j1 = 0; j1 < Tj1; j1++)
          for (k1 = 0; k1 < Tk1; k1++)
            C[i1+i2][j1+j2] += A[i1+i2][k1+k2] * B[k1+k2][j1+j2];
```

以这样的矩阵乘作为例子, 进行单层内存层级的tiling建模. 首先循环变量表示为$i_{1},i_{2},\ldots,i_{l+1}$, 其中$l$为tiling层级(我理解这个应该就是循环层级), $l==0$表示的是statement, $l+1$表示的就是最外层循环,$l==1$表示最内层循环. tilesize变量表示为$T_{i_{1}},T_{i_{2}},\ldots,T{i_{l+1}}$.

对于一个固定的循环order, 我们可以用上面定义的变量来建模数据移动. 令$DF(A,i)$表示的是数组$A$从循环$i$开始的被访问过的**不同**元素的数量(Data Footprint).令$DM(A,i)$表示在循环$i$处数组$A$从主存到cache的数据移动(Data Movement). 下面这段代码展示令如何计算这两个不同的项目, 其实他们的区别也就是在于存在数据复用的时候, 内存足迹是不改变的, 但是内存移动量在内存足迹大于cache size的时候是需要重复load的. 根据这个计算方式, 也就是表示如果有一个维度足够小, 可以让数据完整的放在cache中, 那么他的就不需要重复load.

```csharp
foreach (loop i in Loops.Reverse())  {
  if (i == 0) { // statement level
    foreach(tensor A in Tensors) {
      DM(A, i) = DF(A, i) = 1;
    }
  }
  else {
    foreach(tensor A in Tensors) {
      if (A.Indices.Contains(i)) {
        DM(A, i) = DM(A, i − 1) * range(i);
        DF(A, i) = DF(A, i − 1) * range(i);
      } else {
        DF(A, i) = DF(A, i − 1);
        if (Sum(Tensors, A => DF(A, i − 1)) < CacheCapacity) {
          DM(A, i) = DM(A, i − 1);
        } else {
          DM(A, i) = DM(A, i − 1) * range(i);
        }
      }
    }
  }
}
```

# 2. 多内存层级建模

目前的现代处理架构通常有多个内存层级, 更高的内存层级会有更大的存储以及更小的带宽. 假设每一个循环只能tile到其中一个内存层级, 对于一个两级内存层级的tiling示例代码如下:
```cpp
for (i3 = 0; i3 < Ni; i3+=Ti2)
  for (j3 = 0; j3 < Nj; j3+=Tj2)
    for (k3 = 0; k3 < Nk; k3+=Tk2)
      for (i2 = 0; i2 < Ti2; i2+=Ti1)
        for (j2 = 0; j2 < Tj2; j2+=Tj1)
          for (k2 = 0; k2 < Tk2; k2+=Tk1)
            for (i1 = 0; i1 < Ti1; i1++)
              for (j1 = 0; j1 < Tj1; j1++)
                for (k1 = 0; k1 < Tk1; k1++)
                  C[i1+i2+i3][j1+j2+j3] += A[i1+i2+i3][k1+k2+k3] * B[k1+k2+k3][j1+j2+j3];
```

为了适配多内存层级, 需要修改$DM$的定义为$DM(A,i,l)$表示数组$A$在循环$i$上从内存层级$l$到$l+1$的数据移动, 但原始论文只说了更新建模的代码从`Sum(Tensors, A => DF(A, i − 1)) < CacheCapacity`到`Sum(Tensors, A => DF(A, i − 1)) < CacheCapacity(l)`, 并没有给出完整的实现. 
这里实际上每一个出现`DM`的地方都需要添加$l$参数, 那么可能需要在最外层再添加关于内存层级的循环, 但是从直觉上来说内存层级的选择这里是多分支的, 不是简单的双循环就可以构造出来的.

# 3. 估计执行时间

这里假设访存是主要的瓶颈, 那么通过统计每个内存层级的数据移动量以及对应的带宽大小来估计总时间. 令$L$表示内存层级编号, $C_1,\ldots,C_{L}$表示各个内层层级, 而$C_0$表示计算单元,$C_{L+1}$表示主存. 令$BW_1,\ldots,BW_{L+1}$表示各个内存层级的带宽. 令$C\_DM_(l)$表示从内存层级$l$到$l-1$的数据移动量,$C\_Time(\mathcal{P}, l)$在特定的循环排列下从内存层级$l$到$l-1$的数据移动时间. 那么给定一个固定的循环排列$\mathcal{P}$,对应的时间为:
$$
\begin{aligned}
  C\_Time(\mathcal{P}, l) &= C\_DM(l) / BW_l \\
  TotTime(\mathcal{P}) &= \max_{l = 1}^{L+1} \left( C\_Time(\mathcal{P}, l) \right)
\end{aligned}
$$

# 4. tilesize与循环排列选择

首先对于一个固定循环排列下的tile size选择就变成了一个约束求解问题:
$$
\begin{aligned}
  \arg \min_{tile-sizes}(TotTime(\mathcal{P})) = \arg \min_{tile-sizes}\left(\max_{l = 1}^{L+1} \left( C\_Time(\mathcal{P}, l) \right)\right)
\end{aligned}
$$

为了减少搜索空间, 限制对于内存层级$l$的所有内存移动量$C\_DM(l)$必须小于等于内存层级的容量, 令$group\_outer(l)$表示属于内存层级$l$的最外层循环, 构建容量约束如下:
$$
\begin{aligned}
  \forall l \in [1,L],\ \sum_{A\in Tensors} DM(A, group\_outer(l)) \leq CacheCapacity(l)
\end{aligned}
$$

但这样还是没有考虑tile size对于不同层级的访存速度影响, 需要构建一个多级约束. 令$T$表示所有的tile变量集合, $T_l$表示会影响内存层级$l$的总数据移动量$C\_DM(l)$的tile变量子集. 也就是多个循环可以对应同一个内存层级, 那么这些循环对应的tile变量组成的子集, 这些变量每个都会影响当前内存层级的数据移动量. 

假设$j$是最大瓶颈的内存层级, 也就是他的时间大于其他内存层级时间:
$$
\begin{aligned}
  \forall i \in [1, L+1], \ (C\_DM(j)/C\_BW(j)) \geq (C\_DM(i)/C\_BW (i)
\end{aligned}
$$

当固定了$j$层级的tile size后, 下一个瓶颈的内存层级可以使用如下公式找到:
$$
\begin{aligned}
  \argmin_{T-T_j} (\max_{l\in[1,L+1] - T_j} C\_Time(\mathcal{P},l))
\end{aligned}
$$

但是论文中并没有详细描述怎么进行多级优化, 这里的约束应该都还是变量, 除非是求解器可以通过某种方式添加优化指导.

接下来就是考虑不同的循环排列, 对于两级tiling, 就有9个循环那么排列方式就有9!种. 但其实只需要考虑每个tile出来的循环内部进行排序,也就是$3!\times3!\times3! = 216$种即可. 最终的解为:
$$
\begin{aligned}
final\_solution =  \argmin_{\mathcal{P}\in \mathcal{R}} (\argmin_{tile\_sizes}(TotTime(\mathcal{P})))
\end{aligned}
$$

# 5. micro kernel

对于包含SIMD的指令集的处理器来说, 需要考虑设计一个最大硬件利用率的micro kernel, 这里需要对micro kernel也进行建模. 首先是`MaxIssue`表示一个时钟周期最大可以发射的指令数, `WordPerVec`表示指令集宽度, `Latency`表示一个指令周期所需要的时钟周期数. 假设在一个指令周期中每个时钟周期都可以发射`MaxIssue`个指令数, 并且他们之间是没有数据依赖的, 那么`MaxIssue * Latency`则是可以保证流水线打满的最小指令数, `MaxIssue * Latency * WordPerVec`则表示最小寄存器容量. 比如在`BLIS`库中,通常使用外积的方式来设计micro kernel, 但是这个kernel它需要一块布局优化后的数据块才可以开始计算, 这就要求进行**packing**.

packing的一个好处是可以减少Conflict Miss,现代处理器中cache的存放策略通常是[set-associative](https://en.wikipedia.org/wiki/Cache_placement_policies#Set-associative_cache), 整个缓存被划分为多个set,而这些set内部进一步被细分为lines/ways. 一个映射函数决定内存地址到set的映射. 在每个set内,一个给定的内存地址可能出现在任意一个cache line上. 当在多个不同的内存地址由于缓存映射规则而被迫存储在同一个cache line时, 由于每个缓存行只能存储一个数据块, 当一个新的数据块需要被存储到这个缓存行时, 原有的数据块必须被替换掉, 在理想情况下, 被替换的应该是最近最少使用(LRU)的数据块, 然而在某些情况下, 由于映射规则的限制, 一个不是LRU的数据块也可能被替换, 这就是发生了Conflict Miss.

通过选择合适tile大小,并依赖于packing设计,可以将即数据元素的排列顺序与它们将被访问的顺序相同,从而避免Conflict Miss. 虽然packed buffer在内存中是连续的存储. 但是实际上他们分散在cache中各个地方, 由于大部分的cache是无法编程的, 加载新的数据会将其他tensor从cache中驱除, 为了避免这种情况, 每个tensor拥有的cache line的数量要被小心的控制, 对于矩阵乘的例子这里计算分配给各个tensor的cache line数量:
$$
\begin{aligned}
lineA = [DF(A,l) / (NumOfSets(l)*lineSize(l))] \\
lineB = [DF(B,l) / (NumOfSets(l)*lineSize(l))] \\
lineC = [DF(C,l) / (NumOfSets(l)*lineSize(l))] \\
s.t Line_A(l)+Line_B(l)+Line_C(l) \leq Associativity(l) \\
\end{aligned}
$$
假设只访问A/B矩阵, 考虑在l层级的j循环, A矩阵并没有被j索引, 但是在k循环中多次访问A行, 此时A会停留在cache中, 而B矩阵的行j循环才会被访问, 因此B矩阵在cache中是流式传输的.

packing 会增加额外的数据移动, 为了减少packing带来的额外时间. 比如要复用pack过的数据, 但是由于cache容量的限制, 大部分情况是没办法存储整个的buffer的. 假设A时需要pack的tneosr, 设$IS$表示整个迭代空间, 令$IS_A$表示参与访问A的循环子集, 设packing在最后一层cache $ll$, 那么packing的cost为包含从主存加载以及存储数据到$ll$层上:

$$
\begin{aligned}
PackCost^{A, buf\in Mem}_{mem \rightarrow l3} = \prod_{idx \in IS_A} idx
\end{aligned}
$$

假设l3级别的tiling循环为$i_1^{L3},i_2^{L3},\ldots,i_l^{L3}$, 假设只有$i_2^{L3}, i_l^{L3}$是A的reuse index,那么意味着$i_2^{L3}, i_l^{L3} \notin IS_A$. 假设packed A在level 3进行构造, 那么代码可能类似这样:
$$
\begin{aligned}
for\ loop&\  i_1^{L3}\\
for\ &loop\  i_2^{L3} \\
& \ldots\\
&for\ loop\  i_l^{L3}\\
& \text{Packing buffer resides here;}
\end{aligned}
$$

注意这里packed A将会在$i_2^{L3}$的循环中被填充满, 即使$i_2^{L3}$是他的reuse index, 这意味着对于给定tenors的内部cache packing的总数据移动是tensor size和packing cache level以上的所有reuse loops的乘积. 公式化的描述如下:

$$
\begin{gathered}
R D X^{L 3}=\left\{i_g^{L 3} \mid i_g \notin I S_A \wedge\left(\exists i_h \in I S_A\right)\left[i_g^{L 3}>i_h^{L 3}\right]\right\} \\
\text { PackCost } t_{m e m \rightarrow L b}^{A, b u f \in L 3}=\prod_{i d x \in I S_A} i d x * \prod_{r d x \in R D X^{L 3}} \operatorname{NIter}(r d x)
\end{gathered}
$$

首先令$i^{L3}_p> i^{L3}_q$表示在L3内存层级中$i^{L3}_p$循环在$i^{L3}_q$之上. 再令$\operatorname{NIter}(i^{L3}_p)$表示循环的迭代次数, 令$Tile(i^{L3}_p )$表示索引$p$的tile大小, $N_p$表示索引$p$的全局大小(我理解就是这个tensor在$p$维度的大小). 然后$RDX^{L3}$表示在与A无关的循环迭代中, 存在的迭代$i_g^{L 3}$级别高于$i_h^{L 3}$, 将这些迭代的总次数累积起来与参与A的循环$idx$进行乘积. 

对于任意内存层级的pack cost可以通过如下公式计算:

$$
\begin{aligned}
& R D X^{L_c}=\left\{i_g^{L_c} \mid i_g \notin I S_A \wedge\left(\exists i_h \in I S_A\right)\left[i_g^{L_c}>i_h^{L_c}\right]\right\} \\
& \text { PackCost } \operatorname{mem}_{m \rightarrow L_c}^{A, b u f \in L_c}=\prod_{i d x \in I S_A} i d x * \prod_{r d x \in R D X^{L_c}} \text { NIter }(r d x) \\
& =\prod_{i d x \in I S_A} i d x * \prod_{i_p \in R D X^{L_c}}\left(N_p / \text { Tile }\left(i_p^{L_c}\right)\right) \\
&
\end{aligned}
$$


# 6. 疑问

1. 我在思考他这里的DM和DF和显式的指定sub tensor放置在哪个循环下是否有共同性.
2. 他这里的`group_outer(l)`的函数是一个离散函数, 但是求解器只能支持连续的变量, 现在问题转化为如何通过连续的变量构造出分段函数? 这个方法实际上和对于多层memory的循环加在哪个位置息息相关.
3. 在约束编程中通过什么方式控制几个变量属于一个区间?
4. 怎么样进行多阶段求解?
5. 他这里感觉没有考虑在l1上cache packed buffer, 实际上可以开很大一块buffer, 然后在l1的循环中每次load并pack一小块, 然后在l1中缓存起来, 这样切换m/n的时候可以尽量复用.

