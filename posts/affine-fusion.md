---
title: Affine Fusion Pass浅析
mathjax: true
toc: true
categories:
  - 编译器
date: 2024-01-11 16:00:08
tags:
- mlir
- 多面体模型
---

学习`mlir`中`Affine Fusion Pass`, 主要关注依赖分析部分.


<!--more-->

# 1. 准备工作

首先我们的待测试的`ir`为:

```mlir
module {
  func.func @main(%arg0: memref<8x128x384xf32>, %arg1: memref<8x384x512xf32>, %arg2: memref<8x128x512xf32>, %arg3: memref<8x512x64xf32>, %arg4: memref<8x128x64xf32>) {
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 128 {
        affine.for %arg7 = 0 to 512 {
          affine.for %arg8 = 0 to 384 {
            %0 = affine.load %arg0[%arg5, %arg6, %arg8] : memref<8x128x384xf32>
            %1 = affine.load %arg1[%arg5, %arg8, %arg7] : memref<8x384x512xf32>
            %2 = affine.load %arg2[%arg5, %arg6, %arg7] : memref<8x128x512xf32>
            %3 = arith.mulf %0, %1 : f32
            %4 = arith.addf %2, %3 : f32
            affine.store %4, %arg2[%arg5, %arg6, %arg7] : memref<8x128x512xf32>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 128 {
        affine.for %arg7 = 0 to 64 {
          affine.for %arg8 = 0 to 512 {
            %0 = affine.load %arg2[%arg5, %arg6, %arg8] : memref<8x128x512xf32>
            %1 = affine.load %arg3[%arg5, %arg8, %arg7] : memref<8x512x64xf32>
            %2 = affine.load %arg4[%arg5, %arg6, %arg7] : memref<8x128x64xf32>
            %3 = arith.mulf %0, %1 : f32
            %4 = arith.addf %2, %3 : f32
            affine.store %4, %arg4[%arg5, %arg6, %arg7] : memref<8x128x64xf32>
          }
        }
      }
    }
    return
  }
}
```

# 2. `performFusionsIntoDest(unsigned dstId, unsigned maxSrcUserCount)`

1. 进入`affine fusion pass`之后, 通过`dstId`在`MemRefDependenceGraph`中找到`producer`的`affine for`节点作为`src`节点. 在我们的例子中, 显然是融合上下两个循环块. 

2. 通过`gatherProducerConsumerMemrefs(srcId, dstId, mdg, producerConsumerMemrefs)`收集`src`节点与`dst`节点中的存在生产消费链接的`store/load`.

3. 通过`dstLoopDepthTest = getInnermostCommonLoopDepth(dstMemrefOps)`获取`dst`节点中的内存操作的循环层级, 我们的例子中的循环深度为4.

4. 遍历目标循环的深度`[1, dstLoopDepthTest]`通过`FusionResult result = affine::canFuseLoops(...) }`测试能否将`src loop`放到`dest loop`中

# 3. `affine::canFuseLoops(srcAffineForOp, dstAffineForOp, i, &depthSliceUnions[i - 1], strategy)`

验证是否可以`fusion`是一个复杂的过程. 经过一些琐碎的边界条件处理后, 开始执行判断过程.

1. `numCommonLoops = affine::getNumCommonSurroundingLoops(*srcForOp, *dstForOp);`检查两个`op`外围是否存在共同的循环, 目前的例子中并没有.

2. `switch (fusionStrategy.getStrategy()) `根据不同的策略放入不同的关键路径`op`, 这里`opsA`表示producer, `opsB`表示consumer的.

3. `sliceComputationResult = affine::computeSliceUnion(strategyOpsA, opsB, dstLoopDepth, numCommonLoops, isSrcForOpBeforeDstForOp, srcSlice)`


## 3.1 `computeSliceUnion`

计算`opsA`和`OpsB`在指定循环层级位置计算得到的`slice bounds`是否满足他们之间的依赖. 首先对于producer来说只有写出是重要的, 因此这里的opsA为`affine.store %4, %arg2[%arg5, %arg6, %arg7] : memref<8x128x512xf32>`. 对于consumer来说, 读写同样重要, 因此此时的opsB为
```cpp
%0 = affine.load %arg2[%arg5, %arg6, %arg8] : memref<8x128x512xf32>
%1 = affine.load %arg3[%arg5, %arg8, %arg7] : memref<8x512x64xf32>
%2 = affine.load %arg4[%arg5, %arg6, %arg7] : memref<8x128x64xf32>
affine.store %4, %arg4[%arg5, %arg6, %arg7] : memref<8x128x64xf32>
```

因为我们需要测试前一个执行节点的内存对后面所有的执行的内存的依赖关系, 所以这里是一个全排列组合的测试:

```cpp
  for (auto *i : opsA) {
    MemRefAccess srcAccess(i);
    for (auto *j : opsB) {
      MemRefAccess dstAccess(j);
      if (srcAccess.memref != dstAccess.memref)
        continue;
    }
  }
```

如果他们读写的是同一块`memref`, 那么也就是存在着依赖, 那么就可能存在着潜在的依赖, 需要进行进一步的依赖测试:

```cpp
bool readReadAccesses = isa<AffineReadOpInterface>(srcAccess.opInst) &&
                        isa<AffineReadOpInterface>(dstAccess.opInst);
FlatAffineValueConstraints dependenceConstraints;
// Check dependence between 'srcAccess' and 'dstAccess'.
DependenceResult result = checkMemrefAccessDependence( /* 如果操作的是同一个buffer, 那么需要检查依赖 */
    srcAccess, dstAccess, /*loopDepth=*/numCommonLoops + 1,
    &dependenceConstraints, /*dependenceComponents=*/nullptr,
    /*allowRAR=*/readReadAccesses)
```


## 3.2 `checkMemrefAccessDependence`

此时我们的`src/dst`分别为:
```sh
Checking for dependence at depth: 1 between:
mlir-asm-printer: Verifying operation: func.func
affine.store %4, %arg2[%arg5, %arg6, %arg7] : memref<8x128x512xf32>
mlir-asm-printer: Verifying operation: func.func
%0 = affine.load %arg2[%arg5, %arg6, %arg8] : memref<8x128x512xf32>
```

接下来从`access`中获得对应的`access relation`:

```cpp
// Create access relation from each MemRefAccess.
FlatAffineRelation srcRel, dstRel;
if (failed(srcAccess.getAccessRelation(srcRel)))
  return DependenceResult::Failure;
if (failed(dstAccess.getAccessRelation(dstRel)))
  return DependenceResult::Failure;
```

首先展示srcRel和dstRel的FlatAffineRelation:
```python
srcRel:
Domain: 0, Range: 7, Symbols: 0, Locals: 0
11 constraints
(Value  Value   Value   Value   None    None    None    const)
 1      0       0       0       -1      0       0       0       = 0
 0      1       0       0       0       -1      0       0       = 0
 0      0       1       0       0       0       -1      0       = 0
 1      0       0       0       0       0       0       0       >= 0
 -1     0       0       0       0       0       0       7       >= 0
 0      1       0       0       0       0       0       0       >= 0
 0      -1      0       0       0       0       0       127     >= 0
 0      0       1       0       0       0       0       0       >= 0
 0      0       -1      0       0       0       0       511     >= 0
 0      0       0       1       0       0       0       0       >= 0
 0      0       0       -1      0       0       0       383     >= 0
dstRel:
Domain: 0, Range: 7, Symbols: 0, Locals: 0
11 constraints
(Value  Value   Value   Value   None    None    None    const)
 1      0       0       0       -1      0       0       0       = 0
 0      1       0       0       0       -1      0       0       = 0
 0      0       0       1       0       0       -1      0       = 0
 1      0       0       0       0       0       0       0       >= 0
 -1     0       0       0       0       0       0       7       >= 0
 0      1       0       0       0       0       0       0       >= 0
 0      -1      0       0       0       0       0       127     >= 0
 0      0       1       0       0       0       0       0       >= 0
 0      0       -1      0       0       0       0       63      >= 0
 0      0       0       1       0       0       0       0       >= 0
 0      0       0       -1      0       0       0       511     >= 0
```

这里需要先讲解一下mlir中的PresburgerSpace的变量类型`enum class VarKind { Symbol, Local, Domain, Range, SetDim = Range };`:

1. Symbol, 表示一个固定但是展示未知的值.
2. Local, 表示的是存在量化变量(existentially quantified variables), 我理解就是farkas引理中的lambda系数, 可以通过约束求解来消除. 考虑这个一个space为`(x,exists q)`, 约束为`1 <= x <= 7, x = 2q`, 此时x为维度变量,q为存在量化变量, 即`(x) : (exists q : q <= x <= 7, x = 2q)`. 此时带入一些值进去, 可以得到满足约束的结果集`{(2,1),(4,2),(6,3)}`
3. Dimension变量被进一步分为Domain 和 Range变量.

在mlir中多面体是和ir深度结合的,比如这里的FlatAffineValueConstraints中是包含了PresburgerSpace以及AffineValue的, 上面输出依赖多面体中的Value列实际上就是一个affine ir的ssa value, 这个例子中其实就是四个迭代变量`%arg5,%arg6,%arg7,%arg8`. 并且access relation中的`numDomainDims`和`numRangeDims`与presburger space中的`numDomainVars`和`numRangeVars`并不是一致的. 上面两个约束他们的domainDims和RangeDims分别都是4和3, 但是这些dim对应的变量类型都是`SetDim = Range`, 所以上面两个relation的Ranges变量个数为`4+3=7`

将两个relation写为isl的形式如下:
```python
srcRel = [i0,i1,i2,i3] -> [l0,l1,l2] : 
 i0 == l0 and
 i1 == l1 and 
 i2 == l2 and
 0 <= i0 < 8 and
 0 <= i1 < 128 and
 0 <= i2 < 512 and
 0 <= i3 < 384
dstRel = [j0,j1,j2,j3] -> [l0,l1,l2] : 
 j0 == l0 and
 j1 == l1 and
 j3 == l2 and
 0 <= j0 < 8 and
 0 <= j1 < 128 and
 0 <= j2 < 64 and
 0 <= j3 < 512
```


这里获得对应的他们对应的domain:

```cpp
FlatAffineValueConstraints srcDomain = srcRel.getDomainSet();
FlatAffineValueConstraints dstDomain = dstRel.getDomainSet();
```

此时srcDomain和dstDomain的约束多面体分别如下:

```python
srcDomain:
Domain: 0, Range: 4, Symbols: 0, Locals: 3
11 constraints
(Value  Value   Value   Value   Local   Local   Local   const)
 1      0       0       0       -1      0       0       0       = 0
 0      1       0       0       0       -1      0       0       = 0
 0      0       1       0       0       0       -1      0       = 0
 1      0       0       0       0       0       0       0       >= 0
 -1     0       0       0       0       0       0       7       >= 0
 0      1       0       0       0       0       0       0       >= 0
 0      -1      0       0       0       0       0       127     >= 0
 0      0       1       0       0       0       0       0       >= 0
 0      0       -1      0       0       0       0       511     >= 0
 0      0       0       1       0       0       0       0       >= 0
 0      0       0       -1      0       0       0       383     >= 0

dstDomain:
Domain: 0, Range: 4, Symbols: 0, Locals: 3
11 constraints
(Value  Value   Value   Value   Local   Local   Local   const)
 1      0       0       0       -1      0       0       0       = 0
 0      1       0       0       0       -1      0       0       = 0
 0      0       0       1       0       0       -1      0       = 0
 1      0       0       0       0       0       0       0       >= 0
 -1     0       0       0       0       0       0       7       >= 0
 0      1       0       0       0       0       0       0       >= 0
 0      -1      0       0       0       0       0       127     >= 0
 0      0       1       0       0       0       0       0       >= 0
 0      0       -1      0       0       0       0       63      >= 0
 0      0       0       1       0       0       0       0       >= 0
 0      0       0       -1      0       0       0       511     >= 0
```

实际domain的约束多面体和access relation的多面体并无大的区别, 将一些变量的类型进行了转换, 同时作为一个set他是不存在domain dims和range dims的.

然后组合两个`relation`, 这里的compose实际上等价`srcRel.apply_range(dstRel)`

```cpp
  dstRel.inverse();
  dstRel.compose(srcRel); // src.domain -> [src.range == dst.domain] -> dst.range
```

`compose`后此时`dstRel`为:
```python
Domain: 0, Range: 8, Symbols: 0, Locals: 0
19 constraints
(Value  Value   Value   Value   Value   Value   Value   Value   const)
 -1     0       0       0       1       0       0       0       0       = 0
 0      -1      0       0       0       1       0       0       0       = 0
 0      0       -1      0       0       0       0       1       0       = 0
 1      0       0       0       0       0       0       0       0       >= 0
 -1     0       0       0       0       0       0       0       7       >= 0
 0      1       0       0       0       0       0       0       0       >= 0
 0      -1      0       0       0       0       0       0       127     >= 0
 0      0       1       0       0       0       0       0       0       >= 0
 0      0       -1      0       0       0       0       0       511     >= 0
 0      0       0       1       0       0       0       0       0       >= 0
 0      0       0       -1      0       0       0       0       383     >= 0
 0      0       0       0       1       0       0       0       0       >= 0
 0      0       0       0       -1      0       0       0       7       >= 0
 0      0       0       0       0       1       0       0       0       >= 0
 0      0       0       0       0       -1      0       0       127     >= 0
 0      0       0       0       0       0       1       0       0       >= 0
 0      0       0       0       0       0       -1      0       63      >= 0
 0      0       0       0       0       0       0       1       0       >= 0
 0      0       0       0       0       0       0       -1      511     >= 0
```

这里的Range为8是因为只存在上下两个循环迭代变量的range变量, 此时的domain dims和range dims均为4, 用isl形式表示应该是:
```python
{ [i0, i1, i2, i3] -> [j0 = i0, j1 = i1, j2, j3 = i2] : 0 <= i0 < 8 and 0 <= i1 < 128 and 0 <= i2 < 512 and 0 <= i3 < 384 and 0 <= j0 < 8 and 0 <= j1 < 128 and 0 <= j2 < 64 and 0 <= j3 < 512 }
```

得到新的`dstRel`后, 添加顺序约束, 也就是当他们的外侧还存在有共享循环时, 需要添加顺序约束, 目前这个例子中没有共享循环, 所以也不做什么. 
```cpp
// Add 'src' happens before 'dst' ordering constraints.
addOrderingConstraints(srcDomain, dstDomain, loopDepth, &dstRel);
```

最终就是检查约束`dstRel.isEmpty()`, 这里`isEmpty`检查的是否存在整数解, 也就是在当前order下上面的map约束是否能满足. 

## 3.3 `getComputationSliceState`

上面这个case检测到存在依赖, 接下来计算依赖的slice大小:

```cpp
mlir::affine::getComputationSliceState(
    Operation *depSourceOp, Operation *depSinkOp,
    FlatAffineValueConstraints *dependenceConstraints, unsigned loopDepth,
    bool isBackwardSlice, ComputationSliceState *sliceState)
```

首先这个case传入的参数`depSourceOp`为前一个块的store, `depSinkOp`为后一个块的load,`dependenceConstraints`为上一步计算得到的dst->src的map, `loopDepth`为需要合并到的循环深度, 当前为1. `isBackwardSlice`为true, 因为source op是在sink op前执行的.

因为我们要计算的是插入到`loopDepth`时的slice大小, 那么第一步则是要删除所有高于`loopDepth`的维度. 因为是反向依赖, 所以dst loop的var在后面, 因此pos为`src loop nums + loopDepth = 5`, 然后num为`dst loop nums - loopDepth = 3` .

```cpp
  // Project out dimensions other than those up to 'loopDepth'.
  unsigned pos = isBackwardSlice ? numSrcLoopIVs + loopDepth : loopDepth;
  unsigned num =
      isBackwardSlice ? numDstLoopIVs - loopDepth : numSrcLoopIVs - loopDepth;
  dependenceConstraints->projectOut(pos, num);
```

消除不需要的变量后, `dependenceConstraints`为如下:

```python
Domain: 0, Range: 5, Symbols: 0, Locals: 0
11 constraints
(Value  Value   Value   Value   Value   const)
 -1     0       0       0       1       0       = 0
 1      0       0       0       0       0       >= 0
 -1     0       0       0       0       7       >= 0
 0      1       0       0       0       0       >= 0
 0      -1      0       0       0       127     >= 0
 0      0       1       0       0       0       >= 0
 0      0       -1      0       0       511     >= 0
 0      0       0       1       0       0       >= 0
 0      0       0       -1      0       383     >= 0
 0      0       0       0       1       0       >= 0
 0      0       0       0       -1      7       >= 0
 ```

 等价于:
```python
{ [i0, i1, i2, i3] -> [j0 = i0] : 0 <= i0 <= 7 and 0 <= i1 <= 127 and 0 <= i2 <= 511 and 0 <= i3 <= 383 }
```

获得循环迭代的SSAValue, 这里因为是backward,因此src变量的起点为0, 总个数在这个例子中为4.
```cpp
  // Add slice loop IV values to 'sliceState'.
unsigned offset = isBackwardSlice ? 0 : loopDepth;
unsigned numSliceLoopIVs = isBackwardSlice ? numSrcLoopIVs : numDstLoopIVs;
dependenceConstraints->getValues(offset, offset + numSliceLoopIVs,
                                  &sliceState->ivs);

// Set up lower/upper bound affine maps for the slice.
sliceState->lbs.resize(numSliceLoopIVs, AffineMap());
sliceState->ubs.resize(numSliceLoopIVs, AffineMap());

// Get bounds for slice IVs in terms of other IVs, symbols, and constants.
dependenceConstraints->getSliceBounds(offset, numSliceLoopIVs,
                                      depSourceOp->getContext(),
                                      &sliceState->lbs, &sliceState->ubs);
```

更新后`sliceState->ivs`中存在了`i0,i1,i2,i3`四个变量. 同时为slice state的lower bounds 和 upper bounds分配好四个affine map, 并通过ir的连接关系得到这些affine map.
`getSliceBounds`是将从offset开始的前num个维度变量上下界作为剩余变量的map, 也就是说要基于上一步的依赖约束得到基于dst为domain所对应src domain的上下界, 由于上一步中project掉了三个dst的循环变量, 因此bounds map的domain维度为1, 同时因为`i0=j0`, 因此得到的lower bounds为`[(d0) -> (d0), (d0) -> (0), (d0) -> (0), (d0) -> (0)]`, upper bounds为`[(d0) -> (d0 + 1), (d0) -> (128), (d0) -> (512), (d0) -> (384)]`.  


接下来获取dst循环的iter var value, 因为这里project out之后所以`numDimsAndSymbols`, 然后又跳过了`offset + numSliceLoopIVs`, 因此这里`sliceBoundOperands`只保留了一个`j0`. 然后将这个vector再分配给`lbOperands, ubOperands`. 最好这里的insertPoint就是dst loop在loop depth的位置.

```cpp
  SmallVector<Value, 4> sliceBoundOperands;
  unsigned numDimsAndSymbols = dependenceConstraints->getNumDimAndSymbolVars();
  for (unsigned i = 0; i < numDimsAndSymbols; ++i) {
    if (i < offset || i >= offset + numSliceLoopIVs) {
      sliceBoundOperands.push_back(dependenceConstraints->getValue(i));
    }
  }

  // Give each bound its own copy of 'sliceBoundOperands' for subsequent
  // canonicalization.
  sliceState->lbOperands.resize(numSliceLoopIVs, sliceBoundOperands);
  sliceState->ubOperands.resize(numSliceLoopIVs, sliceBoundOperands);

  // Set destination loop nest insertion point to block start at 'dstLoopDepth'.
  sliceState->insertPoint =
      isBackwardSlice ? dstLoopIVs[loopDepth - 1].getBody()->begin()
                      : std::prev(srcLoopIVs[loopDepth - 1].getBody()->end());

```

此时如果不考虑复杂的情况, sliceState就算是更新完毕了. 这里直接回到了canFuseLoops之后.

# 4. isFusionProfitable

前面是遍历了dst循环的四个insert point, 检测在这些循环内能否插入source循环.在我们的这个例子中, 显然四个层级都可以插入source 循环. 那么就需要找到最合适的循环插入位置.

首先获得srcloop的循环变量, 然后拿到是两个循环分析结果.
```cpp
  // Compute cost of sliced and unsliced src loop nest.
  SmallVector<AffineForOp, 4> srcLoopIVs;
  getAffineForIVs(*srcOpInst, &srcLoopIVs);

  // Walk src loop nest and collect stats.
  LoopNestStats srcLoopNestStats;
  if (!getLoopNestStats(srcLoopIVs[0], &srcLoopNestStats))
    return false;

  // Compute cost of dst loop nest.
  LoopNestStats dstLoopNestStats;
  if (!getLoopNestStats(dstForOp, &dstLoopNestStats))
    return false;
```

然后计算原始src loop的cost, 以及dst loop的cost
```cpp
// Compute op instance count for the src loop nest without iteration slicing.
  uint64_t srcLoopNestCost = getComputeCost(srcLoopIVs[0], srcLoopNestStats);

  // Compute src loop nest write region size.
  MemRefRegion srcWriteRegion(srcStoreOpInst->getLoc());
  if (failed(srcWriteRegion.compute(srcStoreOpInst, /*loopDepth=*/0))) {
    LLVM_DEBUG(llvm::dbgs()
               << "Unable to compute MemRefRegion for source operation\n");
    return false;
  }

  std::optional<int64_t> maybeSrcWriteRegionSizeBytes =
      srcWriteRegion.getRegionSize();
  if (!maybeSrcWriteRegionSizeBytes.has_value())
    return false;
  int64_t srcWriteRegionSizeBytes = *maybeSrcWriteRegionSizeBytes;

  // Compute op instance count for the src loop nest.
  uint64_t dstLoopNestCost = getComputeCost(dstForOp, dstLoopNestStats);
```


然后开始固定每一种循环合并方式并计算fusion之后的cost
```cpp
  for (unsigned i = maxLegalFusionDepth; i >= 1; --i) {
    const ComputationSliceState &slice = depthSliceUnions[i - 1];
    // Skip slice union if it wasn't computed for this depth.
    if (slice.isEmpty())
      continue;

    int64_t fusedLoopNestComputeCost;
    if (!getFusionComputeCost(srcLoopIVs[0], srcLoopNestStats, dstForOp,
                              dstLoopNestStats, slice,
                              &fusedLoopNestComputeCost)) {
      LLVM_DEBUG(llvm::dbgs() << "Unable to compute fusion compute cost\n");
      continue;
    }
    .
    .
    .
  }
```

# 4.1 getFusionComputeCost


进入基于当前的slice state计算循环次数, 这里`slice.ivs`包含了src的四个循环迭代变量, 将他们对应的for op作为key, 循环次数作为value.
```cpp
  bool mlir::affine::buildSliceTripCountMap(
    const ComputationSliceState &slice,
    llvm::SmallDenseMap<Operation *, uint64_t, 8> *tripCountMap) {
  unsigned numSrcLoopIVs = slice.ivs.size();
  // Populate map from AffineForOp -> trip count
  for (unsigned i = 0; i < numSrcLoopIVs; ++i) {
    AffineForOp forOp = getForInductionVarOwner(slice.ivs[i]);
    auto *op = forOp.getOperation();
    AffineMap lbMap = slice.lbs[i];
    AffineMap ubMap = slice.ubs[i];
    std::optional<uint64_t> tripCount = getConstDifference(lbMap, ubMap);
    // Slice bounds are created with a constant ub - lb difference.
    if (!tripCount.has_value())
      return false;
    (*tripCountMap)[op] = *tripCount;
  }
  return true;
}

```

当前的loop depth为4, 对应ComputationSliceState的lbs和ubs分别为:
```python
lbs[0] : (d0, d1, d2, d3) -> (d0)
ubs[0] : (d0, d1, d2, d3) -> (d0 + 1)
lbs[1] : (d0, d1, d2, d3) -> (d1)
ubs[1] : (d0, d1, d2, d3) -> (d1 + 1)
lbs[2] : (d0, d1, d2, d3) -> (d3)
ubs[2] : (d0, d1, d2, d3) -> (d3 + 1)
lbs[3] : (d0, d1, d2, d3) -> (0)
ubs[3] : (d0, d1, d2, d3) -> (384)
```
最终得到tripCountMap中的trip count则分别为`1,1,1,384`.

使用tripCountMap累乘得到所依赖的slice的循环次数, 当前为`384`.
```cpp
  int64_t sliceIterationCount = getSliceIterationCount(sliceTripCountMap);
```

最后, 计算当前slice的ComputeCost, 计算好后将cost添加到dst的insert point, 再重新计算dst的compute cost.

```cpp
  // Compute op instance count for the src loop nest with iteration slicing.
  int64_t sliceComputeCost = getComputeCostHelper(
      srcForOp, srcStats, &sliceTripCountMap, &computeCostMap);

  // Compute cost of fusion for this depth.
  computeCostMap[insertPointParent] = sliceComputeCost;

  *computeCost =
      getComputeCostHelper(dstForOp, dstStats,
                           /*tripCountOverrideMap=*/nullptr, &computeCostMap);
```

getComputeCostHelper是一个递归+回溯的过程, 首先逐渐递归到最内层循环, 获得当前计算的statement的cost. 假设上一步中`computeCostMap[insertPointParent] = 2364`, 那么在最内存循环原本的opCount为`6`, 检测到`computeCostMap`存在值, 那么累加opCount得到`2372`, 接下来在逐步累乘tirp count.

```cpp
static int64_t getComputeCostHelper(
    Operation *forOp, LoopNestStats &stats,
    llvm::SmallDenseMap<Operation *, uint64_t, 8> *tripCountOverrideMap,
    DenseMap<Operation *, int64_t> *computeCostMap) {
  // 'opCount' is the total number operations in one iteration of 'forOp' body,
  // minus terminator op which is a no-op.
  int64_t opCount = stats.opCountMap[forOp] - 1;
  if (stats.loopMap.count(forOp) > 0) {
    for (auto childForOp : stats.loopMap[forOp]) {
      opCount += getComputeCostHelper(childForOp, stats, tripCountOverrideMap,
                                      computeCostMap);
    }
  }
  // Add in additional op instances from slice (if specified in map).
  if (computeCostMap != nullptr) {
    auto it = computeCostMap->find(forOp);
    if (it != computeCostMap->end()) {
      opCount += it->second;
    }
  }
  // Override trip count (if specified in map).
  int64_t tripCount = stats.tripCountMap[forOp];
  if (tripCountOverrideMap != nullptr) {
    auto it = tripCountOverrideMap->find(forOp);
    if (it != tripCountOverrideMap->end()) {
      tripCount = it->second;
    }
  }
  // Returns the total number of dynamic instances of operations in loop body.
  return tripCount * opCount;
}
```

# 4.2 Update bestDstLoopDepth

计算到fuse loop的cost之后, 计算得到fusion后增加的计算比例系数:
```cpp
    double additionalComputeFraction =
        fusedLoopNestComputeCost /
            (static_cast<double>(srcLoopNestCost) + dstLoopNestCost) -
        1;
```

最终打印在每一个层的fuse之后的结果:

```cpp
  evaluating fusion profitability at depth : 4
   additional compute fraction: 5400.00%
   storage reduction factor: 1.00x
   fused nest cost: 77510737920
   src write region size: 2097152
   slice write region size: 2097152

  evaluating fusion profitability at depth : 3
   additional compute fraction: 5400.00%
   storage reduction factor: 1.00x
   fused nest cost: 77510737920
   src write region size: 2097152
   slice write region size: 2097152

  evaluating fusion profitability at depth : 2
   additional compute fraction: 0.00%
   storage reduction factor: 1.00x
   fused nest cost: 1409286144
   src write region size: 2097152
   slice write region size: 2097152

  evaluating fusion profitability at depth : 1
   additional compute fraction: 0.00%
   storage reduction factor: 1.00x
   fused nest cost: 1409286144
   src write region size: 2097152
   slice write region size: 2097152
```

最终他默认选择在更内层fusion:

```cpp
LoopFusion fusion stats:
  best loop depth: 2
  src loop nest compute cost: 1207959552
  dst loop nest compute cost: 201326592
  fused loop nest compute cost: 1409286144
   src mem: 9961472
   dst mem: 3407872
   fused mem: 5505024
   slice mem: 2097152
 fusion is most profitable at depth 2 with 0% redundant computation and a 58.823529% storage reduction.
Fused src loop 0 into dst loop 1 at depth 2:

affine.for %arg5 = 0 to 8 {
  affine.for %arg6 = 0 to 128 {
    affine.for %arg7 = 0 to 512 {
      affine.for %arg8 = 0 to 384 {
        %0 = affine.load %arg0[%arg5, %arg6, %arg8] : memref<8x128x384xf32>
        %1 = affine.load %arg1[%arg5, %arg8, %arg7] : memref<8x384x512xf32>
        %2 = affine.load %arg2[%arg5, %arg6, %arg7] : memref<8x128x512xf32>
        %3 = arith.mulf %0, %1 : f32
        %4 = arith.addf %2, %3 : f32
        affine.store %4, %arg2[%arg5, %arg6, %arg7] : memref<8x128x512xf32>
      }
    }
    affine.for %arg7 = 0 to 64 {
      affine.for %arg8 = 0 to 512 {
        %0 = affine.load %arg2[%arg5, %arg6, %arg8] : memref<8x128x512xf32>
        %1 = affine.load %arg3[%arg5, %arg8, %arg7] : memref<8x512x64xf32>
        %2 = affine.load %arg4[%arg5, %arg6, %arg7] : memref<8x128x64xf32>
        %3 = arith.mulf %0, %1 : f32
        %4 = arith.addf %2, %3 : f32
        affine.store %4, %arg4[%arg5, %arg6, %arg7] : memref<8x128x64xf32>
      }
    }
  }
}
```

affine loop fusion虽然最终的结果并不一定是好的, 因为他这里没有考虑内存层级以及数据复用等因素, 不过把这些内容作为约束多面体最佳使用实践来学习帮助非常大.