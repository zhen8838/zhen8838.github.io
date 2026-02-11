---
title: mlc-llm 浅析
mathjax: true
toc: true
categories:
  - 编译器
date: 2023-11-01 11:52:09
tags:
- LLM
- tvm
---

学习tvm是如何解决LLM推理问题.

<!--more-->


# 1. Model Arch Generator

LLM有一个特点就是其动态与自回归的特性, 传统CNN的模型的计算通路都保存在模型中, 对于DL Compiler来说只需要将固定shape下的模型进行编译优化即可, 而LLM的计算通路并没有体现在模型中, 万幸的是没有多少厂商会大改LLM的模型结构, 所以DL Compiler的前端去手动去处理也问题不大.

使用`mlc.build`对模型进行编译, 进入`build_model_from_args`函数:


```python
def build_model_from_args(args: argparse.Namespace):
    # 各种配置处理

    # 选择模型进行解析
    model_generators = {
        "llama": llama,
        "mistral": llama,
        "stablelm_epoch": stablelm_3b,
        "gpt_neox": gpt_neox,
        "gpt_bigcode": gpt_bigcode,
        "minigpt": minigpt,
        "gptj": gptj,
        "rwkv": rwkv,
        "rwkv_world": rwkv,
        "chatglm": chatglm,
    }

    # 
```


目前tvm是基于relax的分支支持LLM的, 构建模型的过程主要就是使用relax的主要特性按原始模型结构重新构造了一遍tvm的ir module:

首先是构造`BlockBuilder`的scope, 然后在其中构造整个模型运行的每个阶段.

```python
def get_model(args, hf_config):
    # 处理配置...
    param_manager = ParamManager()
    bb = relax.BlockBuilder()

    if sep_embed:
        create_embed_func(bb, param_manager, config, args.quantization)
    # 省略batching的构造...
    create_prefill_func_for_single_seq(bb, param_manager, config, args.quantization, sep_embed)
    create_decoding_func_for_single_seq(bb, param_manager, config, args.quantization)
    create_kv_cache_func(bb, config)
    create_softmax_func_for_single_seq(bb, config)

    create_metadata_func(
        bb,
        model_name=model_name,
        max_window_size=config.max_sequence_length,
        stop_tokens=[2],
        add_prefix_space=False,
    )
    # 设定动态dim的上下界
    mod = bb.get()
    for gv in mod.functions:
        func = mod[gv]
        if isinstance(func, relax.Function):
            mod[gv] = func.with_attr( "tir_var_upper_bound", { "n": config.max_sequence_length, "m": config.max_sequence_length, }, )

    if args.build_model_only:
        return mod, param_manager, None, config

    return setup_params(mod, param_manager, dtype, config, args)
```

在relax中支持同时包含构造relay的数据流以及tir, 所以下面会使用`nn.emit`以及`nn.emit_te`, 同时还可以使用一些手动优化的vm函数`relax.extern("vm.builtin.paged_attention_kv_cache_append")`以及直接编写的`prim_func`. 

```python
class Linear(nn.Module):
    # ...
    def forward(self, input: relax.Expr) -> relax.Var:
        return nn.emit(relax.op.linear(input, self.weight, self.bias))

def apply_rotary_pos_emb(q, k, position_embedding_base, offset: int = 0):
    def f_rotary_embedding(tensor, offset):
        def rotary_compute(*idx):
            pos = (offset + idx[-3]).astype("float32")
            return rotary_modulate_by_freq(
                tensor,
                idx,
                pos,
                position_embedding_base,
            )

        return tvm.te.compute(tensor.shape, rotary_compute, name="rotary")

    q_embed = nn.emit_te(f_rotary_embedding, q, offset, primfunc_name_hint="rotary_embedding")
    k_embed = nn.emit_te(f_rotary_embedding, k, offset, primfunc_name_hint="rotary_embedding")
    return q_embed, k_embed

class LlamaPagedAttention(LlamaAttentionBase):
    # ...
    def attention_fwd(
        self,
        query_states: relax.Expr,
        key_states: relax.Expr,
        value_states: relax.Expr,
        past_key_values: relax.Expr,
        batch_size: tir.PrimExpr,
        q_len: tir.PrimExpr,
        **kwargs,
    ) -> Tuple[relax.Expr, relax.Expr]:
        assert "layer_id" in kwargs and isinstance(kwargs["layer_id"], int)
        layer_id = kwargs["layer_id"]

        f_kv_cache_append = relax.extern("vm.builtin.paged_attention_kv_cache_append")
        past_key_values = nn.emit(
            relax.call_pure_packed(
                f_kv_cache_append,
                past_key_values,
                self.kv_cache_transpose_append,
                key_states,
                value_states,
                relax.PrimValue(layer_id),
                sinfo_args=relax.ObjectStructInfo(),
            )
        )
        # ...
        return attn_output, past_key_values

def emit_paged_kv_cache_op(bb: relax.BlockBuilder, dtype: str) -> None:
    from tvm.script import tir as T

    # fmt: off
    @T.prim_func
    def kv_cache_transpose_append(
        var_pages: T.handle,
        var_k_data: T.handle,
        var_v_data: T.handle,
        var_page_table_indptr: T.handle,
        var_page_table_values: T.handle,
        var_last_page_offset: T.handle,
        var_append_length_indptr: T.handle,
        var_pos2seqidx: T.handle,
        layer_id: T.int32,
    ):
        # 省略buffer构造...
        for global_pos, h, f in T.grid(ntoken, nhead, nfeat):
            with T.block("k_transpose_append"):
                vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                seq_idx = pos2seqidx[vgpos]
                seqlen: T.int32 = (page_table_indptr[seq_idx + 1] - page_table_indptr[seq_idx] - 1) * page_size + last_page_offset[seq_idx]
                pages[
                    page_table_values[page_table_indptr[seq_idx] + T.floordiv(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size)],
                    layer_id,
                    0,
                    vh,
                    T.floormod(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size),
                    vf,
                ] = k_data[vgpos, vh, vf]
            with T.block("v_transpose_append"):
                vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                seq_idx = pos2seqidx[vgpos]
                seqlen: T.int32 = (page_table_indptr[seq_idx + 1] - page_table_indptr[seq_idx] - 1) * page_size + last_page_offset[seq_idx]
                pages[
                    page_table_values[page_table_indptr[seq_idx] + T.floordiv(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size)],
                    layer_id,
                    1,
                    vh,
                    T.floormod(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size),
                    vf,
                ] = v_data[vgpos, vh, vf]
    # fmt: on

    bb.add_func(kv_cache_transpose_append, "kv_cache_transpose_append")
    # Todo: integrating attention TIR func/kernel.
    bb.add_func(relax.extern("attention_func"), "attention")
```


在源代码中检索了一下, 发现是在vm中是直接实现了kv cache, 同时将kv cache的接口进行了封装, 让relax可以进行调用.

```cpp
class AttentionKVCacheObj : public Object {
 public:
  /*!
   * \brief Underlying support data.
   */
  NDArray data;

  /*!
   * \brief number of slots already filled.
   */
  int64_t fill_count{0};

  /*!
   * \brief View all current cached values as one array.
   * \param shape The cached values.
   */
  NDArray View(const ShapeTuple& shape) {
    // ..
  }

  /** Clear the cache */
  void Clear() { /* ... */ }

  /** pop n entries */
  void PopN(size_t n) {
    // ...
  }

  void Update(NDArray value) {
    // ...
  }

  /*!
   * \brief Append value to the cache.
   * \param value The value to be appended.
   */
  void Append(NDArray value) {
    // ...
  }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.vm.AttentionKVCache";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttentionKVCacheObj, Object);
};

// register
TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_create")
TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_create_multiple")
TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_update")
TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_append")
TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_view")
TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_array_popn")
TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_array_clear")
```


其实tvm这种直接在module中构造操作的方式也是很方便的, 如果是传统的编译流程对于每个层还需要写pattern去切子图, 并且一些kv cache相关的优化可能还需要通过一些选项去在某些位置强行添加.

[mod_after_get_model.py](https://gist.github.com/zhen8838/ad8bbe9286e6fa13798fbdb70f1de4ac)

# 2. Module Transform

如果开启了量化还需要更新全部的参数, 然后对构造好的`IR.Module`进行优化, 这里也是一些比较有针对性的优化pass:
```python
def mod_transform_before_build(
    mod: tvm.IRModule,
    param_manager: param_manager.ParamManager,
    args: argparse.Namespace,
    config: Dict,
) -> tvm.IRModule:
  # 
  mod = param_manager.transform_dequantize()(mod)
  mod = relax.transform.BundleModelParams()(mod)
  use_ft_quant = args.quantization.name in ["q4f16_ft", "q8f16_ft"]
  mod = mlc_llm.transform.FuseDecodeTranspose(skip_gemm=not use_ft_quant)(mod)

  if max_seq_len:
    num_key_value_heads = config.get_num_key_value_heads()
    mod = fuse_split_rotary_embedding(
            config.num_attention_heads // args.num_shards,
            num_key_value_heads // args.num_shards,
            config.hidden_size // args.num_shards,
            config.position_embedding_base,
        )(mod)
  if args.target_kind == "cuda":
    # ...
  mod = mlc_llm.transform.FuseTransposeMatmul()(mod)
  mod = relax.pipeline.get_pipeline()(mod)  # pylint: disable=no-value-for-parameter
  mod = mlc_llm.transform.FuseDecodeMatmulEwise()(mod)
  mod = mlc_llm.transform.FuseDecodeTake()(mod)
  mod = relax.transform.DeadCodeElimination(model_names)(mod)
  mod = mlc_llm.transform.CleanUpTIRAttrs()(mod)
  mod_deploy = mod
  return mod_deploy

```

修改后的Module如下, 相比原本的Module多了许多Fused的算子.

[mod_depoly.py](https://gist.github.com/zhen8838/1cecf3800d37fbaf5bb9d1c39e0700f2)

# 3. Module Build

build的过程就是调用原本tvm中的编译下降进行处理, 这里我的target为`m1-metal`:

```python
def build(mod_deploy: tvm.IRModule, args: argparse.Namespace) -> None:
    # dump ...
    if target_kind != "cpu":
        dispatch_target = (
            args.target
            if args.target_kind != "webgpu"
            else tvm.target.Target("apple/m1-gpu-restricted")
        )
        with dispatch_target:
            mod_deploy = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                dl.gpu.Matmul(),
                dl.gpu.GEMV(),
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            )(mod_deploy)
            mod_deploy = (
                mlc_llm.transform.LiftTIRGlobalBufferAlloc()(  # pylint: disable=not-callable
                    mod_deploy
                )
            )
            mod_deploy = tvm.tir.transform.ForceNarrowIndexToInt32()(mod_deploy)

    # 省略使用cuda...
    args.lib_path = os.path.join(args.artifact_path, output_filename)
    ex.export_library(args.lib_path, **args.export_kwargs)
    print(f"Finish exporting to {args.lib_path}")
```

relax中的原生支持动态shape, 所以在`decode`过程中是通过`dataflow`的形式来执行:
```python
@R.function
    def decode(input_ids1: R.Tensor((1, 1), dtype="int32"), all_seq_len: R.Shape(["n"]), kv_cache:...):
        cls = Module
        with R.dataflow():
          # ...
          lv1897 = R.call_tir(cls.transpose5, (lv1894,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
          lv1898 = R.call_tir(cls.transpose5, (lv1895,), out_sinfo=R.Tensor((1, 32, n, 80), dtype="float16"))
          lv722 = R.call_tir(cls.fused_NT_matmul7_divide2_maximum1_minimum1_cast9, (lv1896, lv1897, lv1871), out_sinfo=R.Tensor((1, 32, 1, n), dtype="float32"))
          # ...
```

例如`decode`中的`transpose5`函数在`before build`阶段, tir中是以动态的方式进行构造的:

```python
    @T.prim_func(private=True)
    def transpose5(var_A: T.handle, var_T_transpose: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(80)), "float16")
        T_transpose = T.match_buffer(var_T_transpose, (T.int64(1), T.int64(32), n, T.int64(80)), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, T.int64(80)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]
```

在`after build`阶段, 经过编译下降之后的block中的`iterVar`被映射到了`thread`和`block`两个层级. 我估计在tvm中对于动态申请的内存默认都是连续的, 所以这里`match buffer`也没有特别的`stride`.

```python
    @T.prim_func(private=True)
    def transpose5(var_A: T.handle, var_T_transpose: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int32()
        A = T.match_buffer(var_A, (1, n, 32, 80), "float16")
        T_transpose = T.match_buffer(var_T_transpose, (1, 32, n, 80), "float16")
        # with T.block("root"):
        for ax0_ax1_ax2_fused_0 in T.thread_binding((n * 2560 + 1023) // 1024, thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(1024, thread="threadIdx.x"):
                with T.block("T_transpose"):
                    v0 = T.axis.spatial(32, (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1) // (80 * n))
                    v1 = T.axis.spatial(n, (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1) % (80 * n) // 80)
                    v2 = T.axis.spatial(80, (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1) % 80)
                    T.where(ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1 < n * 2560)
                    T.reads(A[0, v1, v0, v2])
                    T.writes(T_transpose[0, v0, v1, v2])
                    T_transpose[0, v0, v1, v2] = A[0, v1, v0, v2]
```

编译后的模型如下:

[mod_build_stage.py](https://gist.github.com/zhen8838/9a17fa51763ee24baea00f57c2e4e73f)

# 4. Chat

chat 其实经过之前的编译过程后会非常的精简, 只需要获取对应编译后模型的`packed func`然后反复调用即可.

```cpp
class ChatModule {
 public:
  /*!
   * \brief Constructor
   * \param device the device to run the chat on.
   */
  explicit ChatModule(const DLDevice& device) {
    this->chat_mod_ = mlc::llm::CreateChatModule(device);
    this->prefill_ = this->chat_mod_->GetFunction("prefill");
    this->decode_ = this->chat_mod_->GetFunction("decode");
    this->stopped_ = this->chat_mod_->GetFunction("stopped");
    this->get_message_ = this->chat_mod_->GetFunction("get_message");
    this->reload_ = this->chat_mod_->GetFunction("reload");
    this->get_role0_ = this->chat_mod_->GetFunction("get_role0");
    this->get_role1_ = this->chat_mod_->GetFunction("get_role1");
    this->runtime_stats_text_ = this->chat_mod_->GetFunction("runtime_stats_text");
    this->verbose_runtime_stats_text_ = this->chat_mod_->GetFunction("verbose_runtime_stats_text");
    this->reset_chat_ = this->chat_mod_->GetFunction("reset_chat");
    this->process_system_prompts_ = this->chat_mod_->GetFunction("process_system_prompts");
    this->lib_path_ = "";
    this->executable_ = tvm::runtime::Module(nullptr);
    ICHECK(prefill_ != nullptr);
    ICHECK(decode_ != nullptr);
    ICHECK(stopped_ != nullptr);
    ICHECK(get_message_ != nullptr);
    ICHECK(reload_ != nullptr);
    ICHECK(get_role0_ != nullptr);
    ICHECK(get_role1_ != nullptr);
    ICHECK(runtime_stats_text_ != nullptr);
    ICHECK(verbose_runtime_stats_text_ != nullptr);
    ICHECK(reset_chat_ != nullptr);
  }
```