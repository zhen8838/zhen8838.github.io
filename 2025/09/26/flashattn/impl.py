import unittest
import jax
import jax.numpy as jnp
from jax import random, grad
import numpy as np
from jax import lax
from functools import partial
import time

MAX_EXP = 50.0


@partial(jax.jit, static_argnames=['block_size_q', 'block_size_kv'])
def flash_attention_forward(query, key, value, key_padding_mask, query_padding_mask, causal, block_size_q, block_size_kv):
  """
  统一命名规范的 FlashAttention 前向传播实现
  """
  # 获取输入形状
  batch_heads, query_seq_len, head_dim = query.shape
  _, key_seq_len, value_dim = value.shape

  # 默认掩码处理
  if key_padding_mask is None:
    key_padding_mask = jnp.ones((batch_heads, key_seq_len), dtype=bool)
  if query_padding_mask is None:
    query_padding_mask = jnp.ones((batch_heads, query_seq_len), dtype=bool)

  scale = 1.0 / jnp.sqrt(head_dim)

  # 块大小定义
  query_block_size = min(block_size_q, query_seq_len)
  kv_block_size = min(block_size_kv, key_seq_len)

  assert query_seq_len % query_block_size == 0, " query_seq_len must be divisible by query_block_size"
  assert key_seq_len % kv_block_size == 0, " key_seq_len must be divisible by kv_block_size"
  # 计算块数量
  num_query_blocks = (query_seq_len + query_block_size - 1) // query_block_size
  num_kv_blocks = (key_seq_len + kv_block_size - 1) // kv_block_size

  # 初始化输出
  output = jnp.zeros_like(query)  # [batch_heads, query_seq_len, value_dim]
  max_values = jnp.full((batch_heads, query_seq_len), -jnp.inf, dtype=jnp.float32)
  sum_exp_values = jnp.zeros((batch_heads, query_seq_len), dtype=jnp.float32)
  # 初始化全局有效性标志
  has_valid_global = jnp.zeros((batch_heads, query_seq_len), dtype=bool)
  # ---------- 内层：KV 块循环 ----------

  def kv_block_loop(kv_block_idx, carry):
    """处理单个 KV 块"""
    block_output, block_sum_exp, block_max, block_query, query_start_idx, has_valid = carry
    kv_start_idx = kv_block_idx * kv_block_size

    # 切片 KV 块
    key_block = lax.dynamic_slice_in_dim(key, kv_start_idx, kv_block_size, axis=1)
    value_block = lax.dynamic_slice_in_dim(value, kv_start_idx, kv_block_size, axis=1)

    # 切片掩码
    key_mask_block = lax.dynamic_slice_in_dim(key_padding_mask, kv_start_idx, kv_block_size, axis=1)
    query_mask_block = lax.dynamic_slice_in_dim(
        query_padding_mask, query_start_idx, query_block_size, axis=1)

    # 组合掩码
    def causal_mask_case():
      """创建因果掩码"""
      query_indices = jnp.arange(query_block_size) + query_start_idx
      key_indices = jnp.arange(kv_block_size) + kv_start_idx

      causal_mask = (query_indices[:, None] >= key_indices[None, :])
      causal_mask = causal_mask[None, :, :]  # 增加批次维度
      full_mask = causal_mask & query_mask_block[:, :, None] & key_mask_block[:, None, :]
      return full_mask

    def padding_only_mask_case():
      """仅填充掩码"""
      return query_mask_block[:, :, None] & key_mask_block[:, None, :]

    attention_mask = lax.cond(
        causal,
        causal_mask_case,
        padding_only_mask_case
    )

    # 计算注意力分数
    attention_scores = jnp.einsum('bqd,bkd->bqk', block_query, key_block) * scale
    attention_scores = jnp.clip(attention_scores, -MAX_EXP, MAX_EXP)
    attention_scores = jnp.where(attention_mask, attention_scores, -MAX_EXP)

    # 计算当前块的有效性
    current_valid = jnp.any(attention_mask, axis=-1)
    # 更新全局有效性
    has_valid = has_valid | current_valid

    # 计算注意力权重， 对的，这里block_max 本身是[head, query_block_size] , 然后和 s做max，相当于以及拿到了所有m上的max值。
    new_block_max = jnp.maximum(block_max[..., None], jnp.max(
        attention_scores, axis=-1, keepdims=True)) # max_i
    attention_weights = jnp.exp(attention_scores - new_block_max) # exp(x_i - max_i)
    # 累积计算
    exp_factor = jnp.exp(block_max[..., None] - new_block_max) # exp(max_i-1 - max_i)
    new_sum_exp = exp_factor * block_sum_exp[..., None] + \
        jnp.sum(attention_weights, axis=-1, keepdims=True)
    new_block_output = exp_factor * block_output + \
        jnp.einsum('bqk,bkd->bqd', attention_weights, value_block) # 这里其实没有考虑 sum_i-1 / sum_i 这一步, 而是放到最后去除总的sum

    # 更新状态
    new_block_max = jnp.squeeze(new_block_max, axis=-1)
    new_block_sum_exp = jnp.squeeze(new_sum_exp, axis=-1)

    return (new_block_output, new_block_sum_exp, new_block_max, block_query, query_start_idx, has_valid)

  # ---------- 外层：Q 块循环 ----------
  def query_block_loop(query_block_idx, state):
    """处理单个 Q 块"""
    output, sum_exp, max_vals, has_valid_global = state
    query_start_idx = query_block_idx * query_block_size

    # 切片 Q 块
    query_block = lax.dynamic_slice_in_dim(query, query_start_idx, query_block_size, axis=1)

    # 初始化当前块状态
    block_output = jnp.zeros((batch_heads, query_block_size, value_dim), dtype=query.dtype)
    block_sum_exp = jnp.zeros((batch_heads, query_block_size), dtype=jnp.float32)
    block_max = jnp.full((batch_heads, query_block_size), -jnp.inf, dtype=jnp.float32)
    # 初始化当前块有效性标志
    has_valid_block = jnp.zeros((batch_heads, query_block_size), dtype=bool)
    # 处理所有 KV 块
    carry_init = (block_output, block_sum_exp, block_max,
                  query_block, query_start_idx, has_valid_block)
    block_output, block_sum_exp, block_max, _, _, has_valid_block = lax.fori_loop(
        0, num_kv_blocks, kv_block_loop, carry_init)

    # 归一化输出， 这一步才除sum_i 得到最终的softmax的输出。
    block_output = block_output / jnp.maximum(block_sum_exp[..., None], 1e-6)

    # 更新全局状态
    output = lax.dynamic_update_slice_in_dim(output, block_output, query_start_idx, axis=1)
    sum_exp = lax.dynamic_update_slice_in_dim(sum_exp, block_sum_exp, query_start_idx, axis=1)
    max_vals = lax.dynamic_update_slice_in_dim(max_vals, block_max, query_start_idx, axis=1)
    has_valid_global = lax.dynamic_update_slice_in_dim(
        has_valid_global, has_valid_block, query_start_idx, axis=1)
    return (output, sum_exp, max_vals, has_valid_global)

  # 执行所有 Q 块循环
  init_state = (output, sum_exp_values, max_values, has_valid_global)
  output, sum_exp_values, max_values, has_valid_global = lax.fori_loop(
      0, num_query_blocks, query_block_loop, init_state)

  # 计算 log-sum-exp
  log_sum_exp = jnp.where(jnp.isinf(max_values),
                          max_values,
                          max_values + jnp.log(sum_exp_values + 1e-6))

  # 返回结果和残差
  residuals = (query, key, value, log_sum_exp, output, causal,
               key_padding_mask, query_padding_mask, has_valid_global)
  return output, residuals


@partial(jax.jit, static_argnames=['block_size_q', 'block_size_kv'])
def flash_attention_backward(block_size_q, block_size_kv, residuals, grad_input):
  """
  统一命名规范的 FlashAttention 后向传播实现
  """
  # 解包残差
  query, key, value, log_sum_exp, output, causal, key_padding_mask, query_padding_mask, has_valid_global = residuals

  # 获取输入形状
  batch_heads, query_seq_len, head_dim = query.shape
  _, key_seq_len, value_dim = value.shape

  # 默认掩码处理
  if key_padding_mask is None:
    key_padding_mask = jnp.ones((batch_heads, key_seq_len), dtype=bool)
  if query_padding_mask is None:
    query_padding_mask = jnp.ones((batch_heads, query_seq_len), dtype=bool)

  scale = 1.0 / jnp.sqrt(head_dim)

  # Delta 计算: 输出与输出梯度的点积
  output_dot_grad = jnp.sum(output * grad_input, axis=-1)  # [batch_heads, query_seq_len]

  # 块大小定义
  query_block_size = min(block_size_q, query_seq_len)
  kv_block_size = min(block_size_kv, key_seq_len)

  # 计算块数量
  num_query_blocks = (query_seq_len + query_block_size - 1) // query_block_size
  num_kv_blocks = (key_seq_len + kv_block_size - 1) // kv_block_size

  # 初始化梯度
  grad_query = jnp.zeros_like(query)  # [batch_heads, query_seq_len, head_dim]
  grad_key = jnp.zeros_like(key)      # [batch_heads, key_seq_len, head_dim]
  grad_value = jnp.zeros_like(value)  # [batch_heads, key_seq_len, value_dim]

  # 外层循环：Q块循环
  def query_block_loop(query_block_idx, carry):
    """处理单个 Q 块的后向传播"""
    grad_query, grad_key, grad_value = carry
    query_start_idx = query_block_idx * query_block_size

    # 加载当前 Q 块相关的数据
    query_block = lax.dynamic_slice_in_dim(query, query_start_idx, query_block_size, axis=1)
    # query_mask_block = lax.dynamic_slice_in_dim(query_padding_mask, query_start_idx, query_block_size, axis=1)
    # query_block = jnp.where(query_mask_block[:, :, None], query_block, 0.0)

    grad_input_block = lax.dynamic_slice_in_dim(
        grad_input, query_start_idx, query_block_size, axis=1)
    log_sum_exp_block = lax.dynamic_slice_in_dim(
        log_sum_exp, query_start_idx, query_block_size, axis=1)
    output_dot_grad_block = lax.dynamic_slice_in_dim(
        output_dot_grad, query_start_idx, query_block_size, axis=1)
    has_valid_block = lax.dynamic_slice_in_dim(
        has_valid_global, query_start_idx, query_block_size, axis=1)

    # 添加额外维度以便计算
    log_sum_exp_block = log_sum_exp_block[..., None]
    output_dot_grad_block = output_dot_grad_block[..., None]

    # 初始化当前 Q 块的梯度
    grad_query_block = jnp.zeros_like(query_block)

    # 内层循环：KV块循环
    def kv_block_loop(kv_block_idx, inner_carry):
      """处理单个 KV 块的后向传播"""
      grad_query_block, grad_key, grad_value = inner_carry
      kv_start_idx = kv_block_idx * kv_block_size

      # 创建掩码
      key_mask_block = lax.dynamic_slice_in_dim(
          key_padding_mask, kv_start_idx, kv_block_size, axis=1)
      query_mask_block = lax.dynamic_slice_in_dim(
          query_padding_mask, query_start_idx, query_block_size, axis=1)
      # 加载当前 KV 块数据
      key_block = lax.dynamic_slice_in_dim(key, kv_start_idx, kv_block_size, axis=1)
      # key_block = jnp.where(key_mask_block[:, :, None], key_block, 0.0)
      value_block = lax.dynamic_slice_in_dim(value, kv_start_idx, kv_block_size, axis=1)
      # value_block = jnp.where(key_mask_block[:, :, None], value_block, 0.0)

      def causal_mask_case():
        """创建因果掩码"""
        # 创建局部索引 (固定大小)
        query_indices = jnp.arange(query_block_size) + query_start_idx
        key_indices = jnp.arange(kv_block_size) + kv_start_idx

        causal_mask = (query_indices[:, None] >= key_indices[None, :])
        causal_mask = causal_mask[None, :, :]  # [1, query_block_size, kv_block_size]
        full_mask = causal_mask & query_mask_block[:, :, None] & key_mask_block[:, None, :]
        return full_mask

      def padding_only_mask_case():
        """仅填充掩码"""
        return query_mask_block[:, :, None] & key_mask_block[:, None, :]

      attention_mask = lax.cond(
          causal,
          causal_mask_case,
          padding_only_mask_case
      )

      # 计算注意力分数

      attention_scores = jnp.einsum('bqd,bkd->bqk', query_block, key_block) * scale
      attention_scores = jnp.clip(attention_scores, -MAX_EXP, MAX_EXP)
      attention_scores = jnp.where(attention_mask, attention_scores, -MAX_EXP)

      # 计算注意力权重
      attention_weights = jnp.exp(attention_scores - log_sum_exp_block)
      # 计算V梯度
      value_grad = jnp.einsum('bqk,bqd->bkd', attention_weights, grad_input_block)
      value_grad_block = lax.dynamic_slice_in_dim(grad_value, kv_start_idx, kv_block_size, axis=1)
      updated_value_grad = value_grad_block + value_grad
      grad_value = lax.dynamic_update_slice_in_dim(
          grad_value, updated_value_grad, kv_start_idx, axis=1)

      # 计算梯度 P → Q/K
      # value_block = jnp.where(key_mask_block[:, :, None], value_block, 0.0)
      dP_ij = jnp.einsum('bqd,bkd->bqk', grad_input_block, value_block)
      # dP_ij = jnp.where(attention_mask, dP_ij, 0.0)

      # 计算注意力权重梯度
      attention_weights = jnp.where(attention_mask, attention_weights, 0.0)
      dS = attention_weights * (dP_ij - output_dot_grad_block) * scale
      # dS= jnp.where(attention_mask, dS, 0.0)

      # 计算查询梯度
      query_grad_block = jnp.einsum('bqk,bkd->bqd', dS, key_block)
      grad_query_block = grad_query_block + query_grad_block

      # 计算键梯度
      key_grad = jnp.einsum('bqk,bqd->bkd', dS, query_block)
      key_grad_block = lax.dynamic_slice_in_dim(grad_key, kv_start_idx, kv_block_size, axis=1)
      updated_key_grad = key_grad_block + key_grad
      grad_key = lax.dynamic_update_slice_in_dim(grad_key, updated_key_grad, kv_start_idx, axis=1)

      return (grad_query_block, grad_key, grad_value)

    # 处理所有 KV 块
    inner_carry_init = (grad_query_block, grad_key, grad_value)
    grad_query_block, grad_key, grad_value = lax.fori_loop(
        0, num_kv_blocks, kv_block_loop, inner_carry_init)
    # 处理全无效行
    grad_query_block = jnp.where(has_valid_block[..., None], grad_query_block, 0.0)

    # 更新查询梯度
    query_grad_block = lax.dynamic_slice_in_dim(
        grad_query, query_start_idx, query_block_size, axis=1)
    updated_query_grad = query_grad_block + grad_query_block
    grad_query = lax.dynamic_update_slice_in_dim(
        grad_query, updated_query_grad, query_start_idx, axis=1)

    return (grad_query, grad_key, grad_value)

  # 处理所有 Q 块
  init_carry = (grad_query, grad_key, grad_value)
  grad_query, grad_key, grad_value = lax.fori_loop(
      0, num_query_blocks, query_block_loop, init_carry)

  return (grad_query, grad_key, grad_value, None, None, None)  # 返回梯度和占位符

# 3. 主函数


@partial(jax.custom_vjp, nondiff_argnums=(6, 7))  # block_size_q, block_size_kv
def flash_attention(q, k, v, padding_mask_k=None, padding_mask_q=None, causal=False, block_size_q=128, block_size_kv=128):
  """主函数：只返回output
  注意：此函数使用了JAX的custom_vjp装饰器，
  以便定义自定义的前向和反向传播规则。
  该函数的目的是提供一个简单的接口，
  以便在训练过程中使用FlashAttention。
  输入形状:
      q, k, v: [batch_heads, seq_len, head_dim]
  padding_mask_k, padding_mask_q: [batch_heads, seq_len] (可选)
  causal: 是否使用因果掩码
  block_size_q: Q块大小
  block_size_kv: KV块大小
  输出形状:
      output: [batch_heads, q_seq_len, v_dim]
  该实现使用 JAX 的 lax.fori_loop 来处理分块计算，
  并且支持因果掩码和动态掩码。
  注意：此实现假设输入的 q, k, v 已经合并了批次和注意力头维度。
  例如,q 的形状为 [batch_heads, q_seq_len, d_head]。
  其中 batch_heads = batch_size * num_heads。
  该函数的目的是提供一个简单的接口，
  以便在训练过程中使用FlashAttention。
  该函数的前向传播规则将返回输出，
  而反向传播规则将使用静态参数计算梯度。
  这使得后向传播可以使用相同的掩码和参数，
  以避免重复计算。
  该实现假设输入的 q, k, v 已经合并了批次和注意力头维度。
  例如,q 的形状为 [batch_heads, q_seq_len, d_head]。
  其中 batch_heads = batch_size * num_heads。
  该函数的目的是提供一个简单的接口，
  以便在训练过程中使用FlashAttention。
  """
  output, _ = flash_attention_forward(
      q, k, v, padding_mask_k, padding_mask_q, causal, block_size_q, block_size_kv)
  return output


flash_attention.defvjp(flash_attention_forward, flash_attention_backward)


class TestFlashAttention(unittest.TestCase):
  def setUp(self):
    # 设置随机种子
    self.key = jax.random.PRNGKey(42)

    # 基本参数
    self.batch_size = 2
    self.num_heads = 4
    self.seq_len = 64
    self.head_dim = 64

    q_key, k_key, v_key, qmask_key, kv_mask_key = jax.random.split(self.key, 5)
    # 创建输入数据
    self.q = jax.random.normal(
        q_key, (self.batch_size * self.num_heads, self.seq_len, self.head_dim))
    self.k = jax.random.normal(
        k_key, (self.batch_size * self.num_heads, self.seq_len, self.head_dim))
    self.v = jax.random.normal(
        v_key, (self.batch_size * self.num_heads, self.seq_len, self.head_dim))

    # 创建掩码
    # self.padding_mask =jnp.concat([ jnp.ones((self.batch_size, self.seq_len//2), dtype=bool),jnp.zeros((self.batch_size, self.seq_len-self.seq_len//2), dtype=bool)],axis=1)
    self.padding_mask = random.bernoulli(kv_mask_key, 0.1, (self.batch_size, self.seq_len))
    # self.padding_mask =jnp.ones((self.batch_size, self.seq_len), dtype=bool)
    # self.padding_mask =jnp.zeros((self.batch_size, self.seq_len), dtype=bool)
    self.padding_mask = jnp.repeat(self.padding_mask, self.num_heads, axis=0)

    # 编译函数 - 修复：移除partial中的causal参数
    self.flash_attention_jit = jax.jit(partial(
        flash_attention,
        block_size_q=32,
        block_size_kv=32
    ))

  def standard_attention(self, q, k, v, padding_mask_k=None, padding_mask_q=None, causal=False):
    """标准注意力实现"""
    # 计算注意力分数
    scores = jnp.einsum('bqd,bkd->bqk', q, k) / jnp.sqrt(k.shape[-1])

    # 应用掩码
    if causal:
      mask = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
      scores = jnp.where(mask, scores, -jnp.inf)

    if padding_mask_k is not None:
      scores = jnp.where(padding_mask_k[:, None, :], scores, -jnp.inf)
    if padding_mask_q is not None:
      scores = jnp.where(padding_mask_q[:, :, None], scores, -jnp.inf)

    # 处理全无效行
    row_valid = jnp.any(jnp.isfinite(scores), axis=-1, keepdims=True)
    scores = jnp.where(row_valid, scores, 0.0)

    # 计算注意力权重
    attn_weights = jax.nn.softmax(scores)

    # 计算输出
    output = jnp.einsum('bqk,bkd->bqd', attn_weights, v)
    return output

  def test_forward_no_mask(self):
    """测试无掩码的前向传播一致性"""
    # FlashAttention 输出
    flash_output = self.flash_attention_jit(
        self.q, self.k, self.v, None, None, False  # 修复：显式传递causal参数
    )

    # 标准注意力输出
    standard_output = self.standard_attention(
        self.q, self.k, self.v, None, None, False
    )

    # 计算差异
    diff = jnp.max(jnp.abs(flash_output - standard_output))
    print(f"无掩码前向传播差异: {diff:.6f}")

    # 验证差异在可接受范围内
    self.assertLess(diff, 1e-3, "无掩码前向传播差异过大")

  def test_forward_with_mask(self):
    """测试带掩码的前向传播一致性"""
    # FlashAttention 输出
    flash_output = self.flash_attention_jit(
        self.q, self.k, self.v, self.padding_mask, self.padding_mask, False  # 修复：显式传递causal参数
    )

    # 标准注意力输出
    standard_output = self.standard_attention(
        self.q, self.k, self.v, self.padding_mask, self.padding_mask, False
    )

    # 计算差异
    diff = jnp.max(jnp.abs(flash_output - standard_output))
    print(f"带掩码前向传播差异: {diff:.6f}")

    # 验证差异在可接受范围内
    self.assertLess(diff, 1e-3, "带掩码前向传播差异过大")

  def test_forward_causal_mask(self):
    """测试因果掩码的前向传播一致性"""
    # FlashAttention 输出
    flash_output = self.flash_attention_jit(
        self.q, self.k, self.v, None, None, True  # 修复：显式传递causal参数
    )

    # 标准注意力输出
    standard_output = self.standard_attention(
        self.q, self.k, self.v, None, None, True
    )

    # 计算差异
    diff = jnp.max(jnp.abs(flash_output - standard_output))
    print(f"因果掩码前向传播差异: {diff:.6f}")

    # 验证差异在可接受范围内
    self.assertLess(diff, 1e-3, "因果掩码前向传播差异过大")

  def test_backward_no_mask(self):
    """测试无掩码的反向传播一致性"""
    # 定义损失函数
    def loss_flash(q, k, v):
      return jnp.sum(self.flash_attention_jit(q, k, v, None, None, False))  # 修复：显式传递causal参数

    def loss_standard(q, k, v):
      return jnp.sum(self.standard_attention(q, k, v, None, None, False))

    # 计算梯度
    grad_flash = jax.grad(loss_flash, (0, 1, 2))(self.q, self.k, self.v)
    grad_standard = jax.grad(loss_standard, (0, 1, 2))(self.q, self.k, self.v)

    # 计算梯度差异
    diff_q = jnp.max(jnp.abs(grad_flash[0] - grad_standard[0]))
    diff_k = jnp.max(jnp.abs(grad_flash[1] - grad_standard[1]))
    diff_v = jnp.max(jnp.abs(grad_flash[2] - grad_standard[2]))

    print(f"无掩码梯度差异 - Q: {diff_q:.6f}, K: {diff_k:.6f}, V: {diff_v:.6f}")

    # 验证差异在可接受范围内
    self.assertLess(diff_q, 1e-3, "Q梯度差异过大")
    self.assertLess(diff_k, 1e-3, "K梯度差异过大")
    self.assertLess(diff_v, 1e-3, "V梯度差异过大")

  def test_backward_with_mask(self):
    """测试带掩码的反向传播一致性"""
    # 定义损失函数
    def loss_flash(q, k, v):
      return jnp.sum(self.flash_attention_jit(q, k, v, self.padding_mask, self.padding_mask, False))  # 修复

    def loss_standard(q, k, v):
      return jnp.sum(self.standard_attention(q, k, v, self.padding_mask, self.padding_mask, False))

    # 计算梯度
    grad_flash = jax.grad(loss_flash, (0, 1, 2))(self.q, self.k, self.v)
    grad_standard = jax.grad(loss_standard, (0, 1, 2))(self.q, self.k, self.v)

    # 计算梯度差异
    diff_q = jnp.max(jnp.abs(grad_flash[0] - grad_standard[0]))
    diff_k = jnp.max(jnp.abs(grad_flash[1] - grad_standard[1]))
    diff_v = jnp.max(jnp.abs(grad_flash[2] - grad_standard[2]))

    print(f"带掩码梯度差异 - Q: {diff_q:.6f}, K: {diff_k:.6f}, V: {diff_v:.6f}")

    # 验证差异在可接受范围内
    self.assertLess(diff_q, 1e-3, "Q梯度差异过大")
    self.assertLess(diff_k, 1e-3, "K梯度差异过大")
    self.assertLess(diff_v, 1e-3, "V梯度差异过大")

  def test_backward_causal_mask(self):
    """测试因果掩码的反向传播一致性"""
    # 定义损失函数
    def loss_flash(q, k, v):
      return jnp.sum(self.flash_attention_jit(q, k, v, None, None, True))  # 修复：显式传递causal参数

    def loss_standard(q, k, v):
      return jnp.sum(self.standard_attention(q, k, v, None, None, True))

    # 计算梯度
    grad_flash = jax.grad(loss_flash, (0, 1, 2))(self.q, self.k, self.v)
    grad_standard = jax.grad(loss_standard, (0, 1, 2))(self.q, self.k, self.v)

    # 计算梯度差异
    diff_q = jnp.max(jnp.abs(grad_flash[0] - grad_standard[0]))
    diff_k = jnp.max(jnp.abs(grad_flash[1] - grad_standard[1]))
    diff_v = jnp.max(jnp.abs(grad_flash[2] - grad_standard[2]))

    print(f"因果掩码梯度差异 - Q: {diff_q:.6f}, K: {diff_k:.6f}, V: {diff_v:.6f}")

    # 验证差异在可接受范围内
    self.assertLess(diff_q, 1e-3, "Q梯度差异过大")
    self.assertLess(diff_k, 1e-3, "K梯度差异过大")
    self.assertLess(diff_v, 1e-3, "V梯度差异过大")

  def test_numerical_stability(self):
    """测试数值稳定性"""
    # 创建极端输入
    q = jnp.full_like(self.q, 1000.0)
    k = jnp.full_like(self.k, -1000.0)
    v = jnp.full_like(self.v, 1e6)

    # 计算输出
    output = self.flash_attention_jit(
        q, k, v, self.padding_mask, self.padding_mask, False
    )

    # 验证输出值范围
    self.assertTrue(jnp.all(jnp.isfinite(output)))
    self.assertFalse(jnp.any(jnp.isnan(output)))
    self.assertFalse(jnp.any(jnp.isinf(output)))
    print("数值稳定性测试通过")

  def test_different_block_sizes(self):
    """测试不同块大小的输出一致性"""
    # 不同块大小配置
    block_sizes = [
        (16, 16),
        (32, 32),
        (64, 64),
        (16, 32),
        (32, 16)
    ]

    outputs = []

    for block_size_q, block_size_kv in block_sizes:
      output = flash_attention(
          self.q, self.k, self.v,
          self.padding_mask, self.padding_mask,
          False,
          block_size_q,
          block_size_kv
      )
      outputs.append(output)

    # 比较不同块大小的输出
    for i in range(1, len(outputs)):
      diff = jnp.max(jnp.abs(outputs[0] - outputs[i]))
      self.assertLess(diff, 1e-3, f"块大小 {block_sizes[0]} 和 {block_sizes[i]} 的输出差异过大")
    print("不同块大小输出一致性测试通过")


if __name__ == "__main__":
  unittest.main()
