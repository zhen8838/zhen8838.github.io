---
title: "TileFlow: A Framework for Modeling Fusion Dataflow via Tree-based Analysis"
mathjax: true
toc: true
categories:
  - 编译器
date: 2023-12-29 17:41:46
tags:
- 多面体模型
- 性能建模
---

学习`TileFlow`这篇论文中是如何进行多个内存层级的`tiling`.

<!--more-->

# 1. 输入格式描述

输入需要如下几个文件, 每个文件描述了不同的内容.
```sh
tileflow arch/arch.yaml prob/prob.yaml map/map.yaml macro.yaml 
```

## 1.1 arch.yaml

描述了整个芯片的架构层级.
```yaml
architecture: 
  version: 0.2 

  subtree:
  - name: System
    
    local: 
    - name: MainMemory
      class: DRAM 
      attributes:
        block-size: 16384
        depth: 1
        word-bits: 16
        read_bandwidth: 4.3
        write_bandwidth: 2.9
      
    subtree: 
    - name: Buffer 
    
      local:  
      - name: Cache 
        class: SRAM
        attributes:
          word-bits: 16
          block_size: 16384
          depth: 3
          read_bandwidth: 52
          write_bandwidth: 20 # 16 


      subtree:
      - name: PE

        local: 
        - name: RegFile[0..255] 
          class: regfile
          attributes:
            meshX: 16
            meshY: 16
            depth: 1
            block_size: 3
            word-bits: 16
            read_bandwidth: 3.2
            write_bandwidth: 3.2

        - name: mac[0..255] 
          class: intmac 
          attributes: 
            word-bits: 16
            meshX: 16
            meshY: 16
```

## 1.2 prob.yaml

这里其实类似`halide`, 需要列出所有共享的迭代变量, 然后下面每个算子都是用这些维度来构建. 具体可以[参考这里](https://timeloop.csail.mit.edu/timeloop/input-formats/problem).

```yaml
problem:
  io:
    ins: A, B, D
    outs: E
  dimensions: [M,N,K,L]
  instance:
    M: M 
    N: N
    L: L
    K: K

  ops:
  - name: GEMM1
    dimensions: [M,L,K] 
    data-spaces:
    - name: C 
      projection:
        - [[M]] 
        - [[L]] 
      read-write: True 
    - name: A 
      projection:
        - [[M]]
        - [[K]]
    - name: B
      projection:
        - [[K]]
        - [[L]]
    ins: A, B
    out: C
    
  - name: GEMM2 
    dimensions: [M,L,N]
    data-spaces:
    - name: E 
      projection: 
        - [[M]]
        - [[N]]
      read-write: True 
    - name: C
      projection: 
        - [[M]]
        - [[L]]
    - name: D 
      projection: 
        - [[L]]
        - [[N]]
    ins: C, D
    out: E 
```

对应的计算如下所示:

```python
M = 512
N = 64
K = 64 
L = 512
C[M,L] = A[M,K] @ B[K,L]
E[M,N] = C[M,L] @ D[L,N]
```

## 1.3 map.yaml

map是一个比较重要的配置, 这里重点说明一下. `type`分为temporal表示顺序执行和spatial表示并行执行. `factors: M = MO N = NO K= KO`表示它将M/N/K维度分别分为MO/NO/KO块.`permutation: NMK`表示循环从内到外分别为`NMK`. 这里的`factors: M=MM K=KM N=NI`表示再次切分这里的三个维度. `multicast: true`表示多播. `split: 1`表示映射到硬件xy.

原始文档[参考这里](https://timeloop.csail.mit.edu/timeloop/input-formats/mapping).

```yaml
mapping:
  node-type: Tile 
  type: temporal 
  factors: M=MO 
  target: MainMemory 

  subtree: 
  - node-type: Scope 
    type: Sequential 

    subtree:
    - node-type: Tile 
      factors: K=KO L=LO
      type: temporal 
      bypass: [C]
      target: MainMemory
      profile: False
      tag: op1
      

      subtree: 
      - node-type: Tile 
        type: temporal 
        factors: K=KM L=LI M=MM
        permutation: LMK
        target: Cache 
        tag: op1 
        
        subtree:
        - node-type: Tile
          type: Spatial 
          factors: M=SX K=SY
          split: 1
          permutation: MK
          target: Cache
          tag: op1 
          
          subtree: 
          - node-type: Tile 
            type: temporal  
            factors: M=1 L=1 K=1
            permutation: MLK 
            target: RegFile
            tag: op1 
            
            subtree:
            - node-type: Op
              name: GEMM1 

      # A common spatial tile
    - node-type: Tile
      type: temporal 
      factors: L=LO N=NO
      target: MainMemory
      profile: False
      bypass: [C]
      tag: op2

      subtree: 
      - node-type: Tile 
        type: temporal 
        factors: M=MM L=LM N=NI
        permutation: NML 
        target: Cache
        tag: op2
        
        subtree:
        - node-type: Tile
          type: Spatial 
          factors: M=SX L=SY
          split: 1
          permutation: ML
          target: Cache
          tag: op2

          subtree:
          - node-type: Tile 
            type: temporal 
            factors: M=1 L=1 N=1 
            permutation: MLN 
            target: RegFile
            tag: op2
            
            subtree: 
            - node-type: Op
              name: GEMM2
```


之前tile flow, 可以得到初始化时的一些关键信息:

```python
-----------------Mapping---------------
root: 0xa23020
read: A B E D update: E 
for M in [0:MO), MainMemory
  read: A B E D update: E fill: A B E D write-back: E 
  Scope: Sequential{
    read: A B fill: A B 
    for K in [0:KO), MainMemory
      for L in [0:LO), MainMemory
        read: C A B update: C fill: A B 
        for K in [0:KM), Cache
          for M in [0:MM), Cache
            for L in [0:LI), Cache
              read: C A B update: C fill: C A B write-back: C 
              for K in [0:16) (Spatial-Y), Cache
                for M in [0:16) (Spatial-X), Cache
                  read: C A B update: C fill: C A B write-back: C 
                  for K in [0:1), RegFile
                    for L in [0:1), RegFile
                      for M in [0:1), RegFile
                        read: C A B update: C fill: C A B write-back: C 
                        Op: GEMM1(A,B,)->C

    read: E D update: E fill: E D write-back: E 
    for L in [0:LO), MainMemory
      for N in [0:NO), MainMemory
        read: C E D update: E fill: E D write-back: E 
        for L in [0:LM), Cache
          for M in [0:MM), Cache
            for N in [0:NI), Cache
              read: C E D update: E fill: C E D write-back: E 
              for L in [0:16) (Spatial-Y), Cache
                for M in [0:16) (Spatial-X), Cache
                  read: C E D update: E fill: C E D write-back: E 
                  for N in [0:1), RegFile
                    for L in [0:1), RegFile
                      for M in [0:1), RegFile
                        read: C E D update: E fill: C E D write-back: E 
                        Op: GEMM2(C,D,)->E

  }
---------------------------------------
constraints:
        KO*KM==4        #  loopcount constraint for tiling of dim K
        LO*LI==512      #  loopcount constraint for tiling of dim L
        MO*MM==32       #  loopcount constraint for tiling of dim M
        LO*LM==32       #  loopcount constraint for tiling of dim L
        MO*MM==32       #  loopcount constraint for tiling of dim M
        NO*NI==64       #  loopcount constraint for tiling of dim N
        (1*1*1*1+1*1*1*1+1*1*1*1)<=3    # Memory constraint at Tile::op1::RegFile::Temporal
        (Max(MM*16*1*1,MM*16*1*1)*Max(LO*LI*1*1,LO*LM*16*1*1)+MM*16*1*1*KM*16*1*1+KM*16*1*1*LI*1*1)<=131072     # Memory constraint at Tile::op1::Cache::Temporal
        (1*1*1*1+1*1*1*1+1*1*1*1)<=3    # Memory constraint at Tile::op2::RegFile::Temporal
        (Max(MM*16*1*1,MM*16*1*1)*Max(LO*LI*1*1,LO*LM*16*1*1)+MM*16*1*1*NI*1*1+LM*16*1*1*NI*1*1)<=131072        # Memory constraint at Tile::op2::Cache::Temporal
        (MO*Max(MM*16*1*1,MM*16*1*1)*Max(KO*KM*16*1*1,1)+Max(KO*KM*16*1*1,1)*Max(LO*LI*1*1,LO*LM*16*1*1)+MO*Max(MM*16*1*1,MM*16*1*1)*Max(1,NO*NI*1*1)+Max(LO*LI*1*1,LO*LM*16*1*1)*Max(1,NO*NI*1*1))<=524288   # Memory constraint at Tile::MainMemory::Temporal
        <16, 16> <= <16, 16>      # Resource constraint at Tile::op1::Cache::Spatial
        <16, 16> <= <16, 16>      # Resource constraint at Tile::op2::Cache::Spatial
        Max(<1, 1>,<1, 1>) <= <1, 1>      # Resource constraint at Scope::Sequential
==============Checker END================
```

其实我对于tileflow最好奇的一点是 temporal buffer是在哪个内存层级申请的, 通过上面这个constraints很好的解释了这一点. 他应该是在每个tile node上都会开buffer, 对于op1在cache上的这个node上设定了c为pypass, 因此他的大小计算为`C[MM*16,L]`, 而其他两个buffer就是根据factor来计算的:`A[MM*16,KM*16]`以及`B[KM*16,LI]`.

## 1.4 执行结果

这里应该是只搜索了buffer size.
```python
***Optimal Mapping:
-----------------Nest Analysis----------------
Tile::MainMemory::Temporal,
strides,low,high:MNKL[4]: 128 64 64 512 ,[4]: 0 0 0 0 ,[4]: 511 63 63 511 
read: A B E D update: E 
for M in [0:MO(4)), MainMemory
   read: A B E D update: E fill: A B E D write-back: E 
   Scope: Sequential
   {
      Tile::op1::MainMemory::Temporal,
      strides,low,high:MNKL[4]: 128 1 32 512 ,[4]: 0 0 0 0 ,[4]: 127 0 63 511 
      read: A B fill: A B 
      for K in [0:KO(2)), MainMemory
        for L in [0:LO(1)), MainMemory
            Tile::op1::Cache::Temporal,
            strides,low,high:MNKL[4]: 128 1 16 512 ,[4]: 0 0 0 0 ,[4]: 127 0 31 511 
            read: C A B update: C fill: A B 
            for K in [0:KM(2)), Cache
              for M in [0:MM(8)), Cache
                for L in [0:LI(512)), Cache
                     Tile::op1::Cache::Spatial,
                     strides,low,high:MNKL[4]: 16 1 1 1 ,[4]: 0 0 0 0 ,[4]: 15 0 15 0 
                     read: C A B update: C fill: C A B write-back: C 
                     for K in [0:16) (Spatial-Y), Cache
                       for M in [0:16) (Spatial-X), Cache
                           Tile::op1::RegFile::Temporal,
                           strides,low,high:MNKL[4]: 1 1 1 1 ,[4]: 0 0 0 0 ,[4]: 0 0 0 0 
                           read: C A B update: C fill: C A B write-back: C 
                           for K in [0:1), RegFile
                             for L in [0:1), RegFile
                               for M in [0:1), RegFile
                                    read: C A B update: C fill: C A B write-back: C 
                                    Op: GEMM1(A,B,)->C

                                    repFactor:0
                                    accesses:0
                                    expanison:16,16
      Tile::op2::MainMemory::Temporal,
      strides,low,high:MNKL[4]: 128 64 1 512 ,[4]: 0 0 0 0 ,[4]: 127 63 0 511 
      read: E D update: E fill: E D write-back: E 
      for L in [0:LO(1)), MainMemory
        for N in [0:NO(1)), MainMemory
            Tile::op2::Cache::Temporal,
            strides,low,high:MNKL[4]: 128 64 1 16 ,[4]: 0 0 0 0 ,[4]: 127 63 0 511 
            read: C E D update: E fill: E D write-back: E 
            for L in [0:LM(32)), Cache
              for M in [0:MM(8)), Cache
                for N in [0:NI(64)), Cache
                     Tile::op2::Cache::Spatial,
                     strides,low,high:MNKL[4]: 16 1 1 1 ,[4]: 0 0 0 0 ,[4]: 15 0 0 15 
                     read: C E D update: E fill: C E D write-back: E 
                     for L in [0:16) (Spatial-Y), Cache
                       for M in [0:16) (Spatial-X), Cache
                           Tile::op2::RegFile::Temporal,
                           strides,low,high:MNKL[4]: 1 1 1 1 ,[4]: 0 0 0 0 ,[4]: 0 0 0 0 
                           read: C E D update: E fill: C E D write-back: E 
                           for N in [0:1), RegFile
                             for L in [0:1), RegFile
                               for M in [0:1), RegFile
                                    read: C E D update: E fill: C E D write-back: E 
                                    Op: GEMM2(C,D,)->E

                                    repFactor:0
                                    accesses:0
                                    expanison:16,16
   }
Cycle: 140288, Energy: 4.18771e+08
--------------END Nest Analysis---------------
```