---
title: Halide 进阶
mathjax: true
toc: true
categories:
  - 编译器
date: 2022-03-19 01:29:29
tags:
- Halide
- DSL
---

主要分析halide内部机制.

<!--more-->

# High Level Ir

## 1. Func的构造

首先我们分析如下代码:
```cpp
float *buffer_raw = new float[size];
Buffer<float> buffer_host(buffer_raw, shape, name);
Func buffer_device(name + "_device"); // 开始定义的是一个空的function
buffer_device(c, i, j) = buffer_host(c, i, j);
return buffer_device;
```

### Rhs

使用`buffer(a,b,c)`时,将会调用`buffer_accessor`构造出一个`call expr`进行返回.

```cpp
template<typename... Args>
    Expr operator()(const std::vector<Expr> &args) const {
        return buffer_accessor(Buffer<>(*this), args); // Call::make(buf, int_args);
    }
```

返回的`call`的`call_type`是一个`image`, 这里halide的call其实不如tvm relay中的直观.
```cpp
Call *node = new Call;
node->type = type; // 返回类型 , 因为当前图像是f32,因此这里也是f32.
node->name = name; // 名字
node->args = args; // 参数
node->call_type = call_type; 
                  // Image,              A load from an input image
                  //  Extern,            A call to an external C-ABI function, possibly with side-effects
                  //  ExternCPlusPlus,   A call to an external C-ABI function, possibly with side-effects
                  //  PureExtern,        A call to a guaranteed-side-effect-free external function
                  //  Halide,            A call to a Func
                  //  Intrinsic,         A possibly-side-effecty compiler intrinsic, which has special handling during codegen
                  //  PureIntrinsic      A side-effect-free version of the above.
node->func = std::move(func); // 如果call type是halide func,那么需要存储函数指针
node->value_index = value_index; // 如果call的函数有很多个值, 那么需要保存call的那个index.
node->image = std::move(image); // 如果call type是image,那么需要存储image指针
node->param = std::move(param); // 如果call 是image param,那么需要存储param的指针
```

### Lhs


首先取使用`buffer_device(a,b,c)`时,将会调用`operator()`构造出一个`FuncRef`进行返回. 其实我觉得可以把它理解成`FuncWrapper`, 就是用来暂存一些dsl需要的信息的结构体.

```cpp
FuncRef Func::operator()(vector<Expr> args) const {
    int placeholder_pos, count;
    std::tie(placeholder_pos, count) = add_implicit_vars(args);
    return FuncRef(func, args, placeholder_pos, count);
}
```

然后调用`FuncRef`的`operator=`去加载右边返回的`call`. 加载完成之后构造一个stage返回, 这里c++好的一点就是重载`operator=`是可以返回值的.
```cpp
Stage FuncRef::operator=(const Tuple &e) {
    func.define(expanded_args_str, e.as_vector());
    return Stage(func, func.definition(), 0);
}
```

### func 转换为expr

func其实只是暂时定义好了数据流中的输入args以及输出ouputs,是一种游离在expr外的结构. 而当一个func要赋值给另一个func时, 才把这个funcion嵌入到数据流中,因此构造一个`call expr`返回, 再作为

```cpp
Stage FuncRef::operator=(const FuncRef &e) {
    if (e.size() == 1) {
        return (*this) = Expr(e);
    } else {
        return (*this) = Tuple(e);
    }
}

FuncRef::operator Expr() const {
    user_assert(func.has_pure_definition() || func.has_extern_definition())
        << "Can't call Func \"" << func.name() << "\" because it has not yet been defined.\n";

    user_assert(func.outputs() == 1)
        << "Can't convert a reference Func \"" << func.name()
        << "\" to an Expr, because " << func.name() << " returns a Tuple.\n";

    return Call::make(func, args);
}
```

## 2. Func Lower 

下面看如何从Func转换到stmt.
```cpp
buffer_device.compile_to_lowered_stmt(buffer_device.name() + ".julia", {});
```

### PipeLine

首先会把Funcion转换为PipeLine
```cpp
Pipeline::Pipeline(const Func &output)
    : contents(new PipelineContents) {
    output.function().freeze();
    contents->outputs.push_back(output.function());
}
```

### Module
然后把pipeline转换的module, 通过调用lower.

```cpp
Module lower(const vector<Function> &output_funcs,
             const string &pipeline_name,
             const Target &t,
             const vector<Argument> &args,
             const LinkageType linkage_type,
             const vector<Stmt> &requirements,
             bool trace_pipeline,
             const vector<IRMutator *> &custom_passes) {
    Module result_module{extract_namespaces(pipeline_name), t};
    run_with_large_stack([&]() {
        lower_impl(output_funcs, pipeline_name, t, args, linkage_type, requirements, trace_pipeline, custom_passes, result_module);
    });
    return result_module;
}
```

#### lower_impl

首先构造一个全局的for循环
```cpp
string root_var = LoopLevel::root().lock().to_string();
Stmt s = For::make(root_var, 0, 1, ForType::Serial, DeviceAPI::Host, Evaluate::make(0));
```
得到了:
```cpp
before injector:
for (.__root, 0, 1) {
 0
}
```

先是`schedule_functions`, 然后通过`InjectFunctionRealization`插入一系列的函数.
```cpp
InjectFunctionRealization injector(funcs, is_output_list, target, env);
s = injector.mutate(s);
internal_assert(injector.found_store_level() && injector.found_compute_level());
```

接着就是一堆的传统编译器pass对lowerd ir进行分析和优化..

#### partition_loops

比如我们写了一个卷积之后,halide可以把一个循环体划分为序言、稳态和尾声。通过寻找使用`clamped ramps`或`likely`的字段来找到稳定状态, 然后进行分离再对每个循环内部进行优化.



## 3. 关于一些IR设计

### 1. produce consume

halide是提供了一个ir,区别两种类型, 其中`is_producer=true`表示对于`buffer`可读可写,否则就是对于buffer只读. 
```cpp
struct ProducerConsumer : public StmtNode<ProducerConsumer> {
    std::string name;
    bool is_producer;
    Stmt body;
}
```

然后构造的时候,如果有一个消费者对,那就构造一个block存储. 这个block也是专为producer和consumer设计的, 既可以保存配对信息,还能保证block中内容都必须要顺序执行, 不会出现执行问题.

```cpp
struct Block : public StmtNode<Block> {
    Stmt first, rest;
}

if (is_no_op(consumer)) {
    // For the very first output to be scheduled, the consumer
    // Stmt can be a no-op. No point in preserving it.
    return producer;
} else {
    return Block::make(producer, consumer);
}
```