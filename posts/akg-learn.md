---
title: AKG 学习
mathjax: true
toc: true
categories:
  - 编译器
date: 2022-10-17 00:47:57
tags:
- 多面体模型
- 后端优化
---

学习AKG的算子编译流程, 主要关于后端.

<!--more-->

# Example 1

这个例子在m1上无法执行, 因为在codegen中直接加了x86的指令.
```sh
❯ pytest -s tests/st/ops/test_batch_matmul.py -m "platform_x86_cpu"
```
然后调用`tests/common/test_run/batch_matmul_run.py`中的`batch_matmul_run`执行核心测试逻辑.

1. 通过te构建batch matmul的op, 以及得到输入输出的Expr.
2. 调用c++部分的`third_party/incubator-tvm/include/tvm/schedule.h:create_schedule`返回得到当前op的schedule
3. 回到python中将schedule与op var等交给c++部分进行function build `src/codegen/build_module.cc:BuildToFunc`

## BuildToFunc

1. 执行提前注册好的llvm的lower`src/codegen/lower_llvm.cc:LLVMLowerImpl`对schedule的结果进行lower, 同时每个stage的逻辑都在这个文件中被注册:
  
  ```cpp
  REG_STAGE_LOWER("llvm", StageType::Begin, "BEGIN", LLVMLowerBegin);
  REG_STAGE_LOWER("llvm", StageType::Tuning, "TUNING", LLVMLowerStageTuning);
  REG_STAGE_LOWER("llvm", StageType::Poly, "POLY", LLVMLowerPoly);
  REG_STAGE_LOWER("llvm", StageType::BeforeFlattern, "BEFORE_FLATTERN", LLVMLowerBeforeFlattern);
  REG_STAGE_LOWER("llvm", StageType::Flattern, "FLATTERN", LLVMLowerFlattern);
  REG_STAGE_LOWER("llvm", StageType::BeforeLowerFunc, "BeforeLowerFunc", LLVMBeforeLowerFunc);
  REG_STAGE_LOWER("llvm", StageType::End, "END", LLVMLowerDone);
  ```

2. 在lower中通过StageManager获取所有需要lower的stage: [Begin,Tuning,Poly,BeforeFlattern,Flattern,BeforeLowerFunc,End],然后依次执行.

### Begin Stage

对应`src/codegen/lower_llvm.cc:LLVMLowerBegin`. 在`LLVMLowerBegin`中需要执行一些基础的pass/analysis来为后续的lower pass提供信息.
```puml
LowerInitWithSchedule -> GetBinds
note right
  将所有原始schedule中的tensor构造出对应的buffer并存放到`data->arg_list_0`和`data->binds_0`中.
      注意`LowerData`中有着如下代码, 让人无法了解具体的含义.

        ```cpp
        Array<NodeRef> args;
        Array<NodeRef> arg_list_0;
        Map<Tensor, Buffer> binds;
        Map<Tensor, Buffer> binds_0;  
        ```
end note
LowerInitWithSchedule -> AutoInline
LowerInitWithSchedule -> AutoFuse
note right
  the <U+0025>autonumber<U+0025> works everywhere.
  Here, its value is ** %autonumber% **
end note
LowerInitWithSchedule -> TensorAccessRewrite : 
```

1. LowerInitWithSchedule
  - GetBinds 

  - AutoInline 应该是inline一些op, 具体实现没有细看, 里面有许多需要特判的case.
  - AutoFuse 应该是将多层的op自动fuse起来, 后续具体的fuse逻辑比较复杂.
    ```cpp
    bool NeedToFuse() {
      if (HasExternOp()) { // 如果包含ExternOp那么不fuse
        return false;
      }
      // ReduceCheck 检查内部的reduce最大有没有超过shared memory的大小. (同时目前的代码是有matmul就不fuse,可能有什么限制.) 
      if (ReduceCheck() || !split_config_.empty()) { // 当然如果强行指定了split config那么也能fuse
        return true;
      }
      return false;
    }
    ```

  - 调用tvm自带的`InferBound`和`ScheduleOps`进行操作
  - TensorAccessRewrite 这个应该是消除显式的`tensor load/store`.
2. ReplaceSeparator 将所有的名字中的`.`替换成`_`, 可能是为了方便调用`isl`.
3. RewriteMultiValueFunc
4. RenameRealize
5. ElementwiseFlatten
6. FuseAxisExternOp
7. AddAttrForLayoutOp
8. RewriteTensorIndex
  


# Tips

卷积的test无法在cpu上执行,不知道为什么在构造Conv的te compute的时候限制了.

