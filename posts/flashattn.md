---
title: Flash Attention记录
toc: true
mathjax: true
categories:
  - 推理框架
date: 2025-09-26 23:36:24
tags:
- 算子
---

简单记录一下flash attention的推导和实现。

<!--more-->

# Naive Attention

$$
\begin{aligned}
  S &= QK^T \in \mathbb{R}^{N\times N} \\
  P &= softmax(S) \\ 
  O &= PV \in \mathbb{R}^{N \times d}
\end{aligned} 
$$

本质上就是两个matmul中间有一个伪elementwise操作。 主要由于softmax每个输出点依赖了他的所有input，导致无法进行tiling fusion。原因如下

# SoftMax

## naive softmax实现：

$$
\begin{aligned}
softmax(x_i, \ldots, x_N) = \frac{e ^ {x_i}}{\sum_{j=1}^{N} e^{x_j}}, i \in [1, N]
\end{aligned}
$$

由于$e^x$会很大，容易出现数值溢出，因此出现safe softmax

## safe softmax实现

$$
\begin{aligned}
  max_{N} &= max(x_i), \ i \in [1, N] \\
  softmax(x_i, \ldots, x_N) &= \frac{e ^ {x_i - max_{N}}}{\sum_{j=1}^{N} e^{x_j - max_{N}}}, i \in [1, N]
\end{aligned}
$$

但是在实现上需要循环三次，因为$max_N$和$x_{sum}$都需要单独的循环

$$
\begin{aligned}
  max_i &= max(m_{i-1}, x_i)\\
  sum_i &= sum_{i-1} + e^{x_i - max_N} \\
  a_i &= \frac{e^{x_i - max_N}}{sum_N}, \ i \in [1, N] 
\end{aligned}
$$


## 2-pass softmax

说实话，让我是没办法能想出来融合max以及sum的公式，可能作者也是受Welford方法启发的。 首先我们考虑把$sum_N$的公式展开，通过$exp$的计算性质把$- max_N$这个拆分为两个部分 
$$
\begin{aligned}
  sum_{N} &= \sum_{j = 1} ^ N e^{x_j - max_N} \\ \\
    &= \sum_{j = 1} ^ {N-1} e^{x_j - max_N} + e^{x_N - max_N}  \\
    &= \sum_{j = 1} ^ {N-1} e^{x_j - max_{N-1} + max_{N-1} - max_N} + e^{x_N - max_N} \\
    &= (\sum_{j = 1} ^ {N-1} e^{x_j - max_{N-1}}) e^{max_{N-1} - max_N} +e^{x_N - max_N} \\
\end{aligned}
$$

观察上面的公式，发现如果从另一个视角去定义变量就可以让他们递归起来
$$
\begin{aligned}
 \text{let}\ sum_{N}^\prime &=\sum_{j=1}^{N} e^{x_j - max_N} = sum_{N} \\
  &= (\sum_{j = 1} ^ {N-1} e^{x_j - max_{N-1}}) e^{max_{N-1} - max_N} +e^{x_N - max_N}  \\
  &= sum_{N-1}^\prime e^{max_{N-1} - max_N} +e^{x_N - max_N} \\
  sum_i ^\prime &= sum_{i-1} ^\prime e^{max_{i-1} - max_i} +e^{x_i - max_i} 
\end{aligned}
$$

通过视角的转换将$max_N$与$sum_{i}$进行了解耦，并且当迭代到最后时$sum_{N}^\prime = sum_{N}$。 虽然2-pass的方式需要在每次迭代添加额外的乘$e^{max_{i-1} - max_i}$的运算，但显然比访存开销低很多。


# Flash Attention

## 2-pass Attention

首先使用2-pass的softmax来实现一个attention,这里为了不混淆`query len`和`seq len`， 分别用`k`和`i`来表示。

$$
\begin{aligned}
\text{for i in [1, N]}:&\\
  x_i &= Q[k, :]K^T[:, i]\\
  max_i &= \max(max_{i-1}, x_i) \\
  sum_i^\prime &= sum_{i-1} ^\prime e^{max_{i-1} - max_i} +e^{x_i - max_i}\\
\text{end} \qquad \qquad \\
\text{for i in [1, N]}:&\\
  a_i &= \frac{e^{x_i - max_N}}{sum_N^\prime} \\
  o_i &= o_{i-1} + a_i V[i,:] \\
\text{end} \qquad \qquad \\
  O[k,:] & = o_N 
\end{aligned}
$$
<!-- 这里的o_{i-1} + a_i 是因为第二个矩阵乘k维度对应的N，也就是要进行mul add的操作 -->

## 1-pass Attention

在和V做矩阵乘时，每一个$o_i$还是依赖了$max_N$。 接下来就是找到办法把$max_N$的依赖消除。参考`2-pass softmax`的套路先定义：
$$
\begin{aligned}
o_N^\prime &= \sum_{i = 1} ^ {N} a_i V[i,:] \\
           &= \sum_{i = 1} ^ {N} \frac{e^{x_i - max_N}}{sum_N^\prime} V[i,:] \\
           &= (\sum_{i = 1} ^ {N-1} \frac{e^{x_i - max_N}}{sum_N^\prime} V[i,:]) + \frac{e^{x_N - max_N}}{sum_N^\prime} V[N,:] \\
           &= (\sum_{i = 1} ^ {N-1} \frac{e^{x_i - max_N}}{sum_N^\prime} \frac{sum_{N-1}^\prime}{sum_{N-1}^\prime} \frac{e^{x_i - max_{N-1}}}{e^{x_i - max_{N-1}}}  V[i,:]) + \frac{e^{x_N - max_N}}{sum_N^\prime} V[N,:] \\
           &= (\sum_{i = 1} ^ {N-1} \frac{e^{x_i - max_{N-1}}}{sum_{N-1}^\prime} V[i,:]) \frac{sum_{N-1}^\prime}{sum_{N}^\prime}\frac{e^{x_i - max_N}}{e^{x_i - max_{N-1}}} + \frac{e^{x_N - max_N}}{sum_N^\prime} V[N,:] \\
           &= (\sum_{i = 1} ^ {N-1} \frac{e^{x_i - max_{N-1}}}{sum_{N-1}^\prime} V[i,:]) \frac{sum_{N-1}^\prime}{sum_{N}^\prime}e^{max_{N-1} - max_N} + \frac{e^{x_N - max_N}}{sum_N^\prime} V[N,:] \\
           &= o_{N-1}^\prime \frac{sum_{N-1}^\prime}{sum_{N}^\prime}e^{max_{N-1} - max_N} + \frac{e^{x_N - max_N}}{sum_N^\prime} V[N,:] \\
\end{aligned}
$$
然后归纳得到不包含$max_N$的$o_i^\prime$公式为：
$$
\begin{aligned}
  o_i^\prime &= o_{i-1}^\prime \frac{sum_{i-1}^\prime}{sum_{i}^\prime}e^{max_{i-1} - max_i} + \frac{e^{x_i - max_i}}{sum_i^\prime} V[i,:]
\end{aligned}
$$

最终列出标量化的1-pass Attention形式：
$$
\begin{aligned}
\text{for i in [1, N]}:&\\
  x_i &= Q[k, :]K^T[:, i]\\
  max_i &= \max(max_{i-1}, x_i) \\
  sum_i^\prime &= sum_{i-1} ^\prime e^{max_{i-1} - max_i} +e^{x_i - max_i}\\
    o_i^\prime &= o_{i-1}^\prime \frac{sum_{i-1}^\prime}{sum_{i}^\prime}e^{max_{i-1} - max_i} + \frac{e^{x_i - max_i}}{sum_i^\prime} V[i,:] \\
\text{end} \qquad \qquad\\
  O[k,:] & = o_N 
\end{aligned}
$$

## Flash Attention v1

上面推导出来的1-pass attention是基于标量循环的，对于flash attention是需要按tile进行计算的，所以具体的公式还需要稍作修改。

首先列出普通的softmax计算公式：

$$
\begin{aligned}
X & =  [x_1, \ldots , x_N] \\
max_N & = \max(X) \\
  &= \max([x_1, \ldots , x_N]) \\
f(X) & = [f(x_1), \ldots , f(x_N)] \\
  &= [e^{x_1 - max_N}, \ldots, e^{x_N - max_N}] \\
sum_N & = \sum_{i = 1}^N f(x_i) \\
  & = \sum_{i = 1}^N e^{x_i - max_N} \\
softmax(X) & = \frac{ f(X)}{sum_N}
\end{aligned}
$$

现在来推导tiled softmax的计算公式， 那么假设现在的$X$是由两个长度为$N$的子向量组成的, 那么首先把它看成单个向量计算，然后拆分转换为可分治的公式：
$$
\begin{aligned}
X & = [x^1, x^2] \\
max_{2N} & = \max([\max(x^1),\max(x^2)]) \\
  & = \max([max_N^1, max_N^2]) \\
f(X) &= \left[ [e^{x_1^1 - max_{2N}},\ldots, e^{x_N^1 - max_{2N}}] , [e^{x_1^2 - max_{2N}},\ldots, e^{x_N^2 - max_{2N}}] \right] \\
    &= \left[ e^{max_N^1 - max_{2N}} [e^{x_1^1 - max_N^1},\ldots, e^{x_N^1 - max_N^1}] , e^{max_N^2 - max_{2N}} [e^{x_1^2 - max_N^2},\ldots, e^{x_N^2 - max_N^2}] \right] \\
    &= \left[ e^{max_N^1 - max_{2N}} f(x^1) , e^{max_N^2 - max_{2N}} f(x^2) \right] \\
sum_{2N} & = \sum_{i = 1}^N e^{x_i^1 - max_{2N}} + \sum_{i = 1}^N e^{x_i^2 - max_{2N}} \\
         & = e^{max_N^1 - max_{2N}} \sum_{i = 1}^N e^{x_i^1 - max_{N}^1} + e^{max_N^2 - max_{2N}} \sum_{i = 1}^N e^{x_i^2 - max_{N}^2} \\
         & = e^{max_N^1 - max_{2N}} sum_N^1 + e^{max_N^2 - max_{2N}} sum_N^2 \\
softmax(X) &= \frac{f(X)}{sum_{2N}}
\end{aligned}
$$

此时可以发现，除了每个子向量的 $max^j$ 用于计算 $f(x^j), sum^j$，还需要维护整体的$max, sum$用于计算最终的结果。

flash attention的tiling就是将$x_i$向量化,基于1-pass attention的公式，结合tiled softmax公式，只需要略微修改$max_i,sum_i$的计算即可得到flash attention的公式：
$$
\begin{aligned}
\text{for i in [1, N/b]}:&\\
  x_i &= Q[k, :]K^T[:, i:i+b]\\
  max_i^{local} &= max(x_i) \\
  max_i &= \max(max_{i-1}, max_i^{local}) \\
  sum_i^\prime &= sum_{i-1} ^\prime e^{max_{i-1} - max_i} + \sum_{j=1}^{b} e^{x_i[j] - max_i}\\
  o_i^\prime &= o_{i-1}^\prime \frac{sum_{i-1}^\prime}{sum_{i}^\prime}e^{max_{i-1} - max_i} + \sum_{j=1}^b \frac{e^{x_i[j] - max_i}}{sum_i^\prime} V[(i-1)b+j,:] \\
\text{end}\qquad \qquad\\
  O[k,:] & = o_N 
\end{aligned}
$$

附上一个简易的flash attention实现供参考：

```python
import pytest
import torch
import torch.nn.functional as F
import math
import numpy as np


np.set_printoptions(suppress=True)


def flash_attn(query: np.ndarray, key: np.ndarray, value: np.ndarray, attn_mask=None, dropout_p=0.0,
               is_causal=False, scale=None, enable_gqa=False):
  L, S = query.shape[-2], key.shape[-2]
  scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
  attn_bias = np.zeros((L, S), dtype=query.dtype)
  if is_causal:
    assert attn_mask is None
    temp_mask = np.tril(np.ones((L, S), dtype=np.bool_), k=0)
    attn_bias[False == temp_mask] = float("-inf")

  if attn_mask is not None:
    if attn_mask.dtype == np.bool_:
      attn_bias[False == attn_mask] = -np.inf
    else:
      attn_bias = attn_mask + attn_bias

  assert enable_gqa is False, "GQA not implemented"

  for head in range(query.shape[0]):
    Q = query[head]  # [query_len, dim]
    K = key[head]  # [seq_len, dim]
    V = value[head]  # [seq_len, dim]
    O = np.zeros_like(Q, dtype=np.float32)  # [query_len, dim]
    Tc = 4
    Tr = 16
    assert L % Tr == 0
    assert S % Tc == 0

    global_maxs = np.zeros([Tr, L // Tr], dtype=np.float32)
    global_sums = np.zeros([Tr, L // Tr], dtype=np.float32)
    for j in range(0, S, Tc):
      # outer loop is seq_len, because seq_len is `K` dimension, we can reuse Kj,Vj `query_len/Tc` times
      Kj = K[j:j + Tc, :]  # [Tc, dim]
      Vj = V[j:j + Tc, :]  # [Tc, dim]
      for (ii, i) in enumerate(range(0, L, Tr)):
        # load
        Qi = Q[i:i + Tr, :]  # [Tr, dim]
        O_last = O[i:i + Tr, :]  # [Tr, dim]
        max_last = (np.zeros([Tr, 1], dtype=np.float32) - np.inf) if j == 0 else global_maxs[:, ii:ii + 1]
        sum_last = np.zeros([Tr, 1], dtype=np.float32) if j == 0 else global_sums[:, ii:ii + 1]

        a_ij = (Qi @ Kj.T) * scale_factor  # [Tr, Tc]
        a_ij += attn_bias[i:i + Tr, j:j + Tc]
        max_local = np.max(a_ij, axis=1, keepdims=True)  # [Tr, 1]
        max_i = np.maximum(max_last, max_local)  # [Tr, 1]
        p_ij = np.exp(a_ij - max_i)  # [Tr, Tc]
        sum_i = sum_last * np.exp(max_last - max_i) + np.sum(p_ij, axis=1, keepdims=True)  # [Tr, 1]
        O_i = O_last * (sum_last * np.exp(max_last - max_i)) / sum_i + (p_ij / sum_i) @ Vj  # [Tr, dim]

        # store
        O[i:i + Tr, :] = O_i
        global_maxs[:, ii:ii + 1] = max_i
        global_sums[:, ii:ii + 1] = sum_i
    return O


@pytest.mark.parametrize("head_q, head_kv", [(1, 1)])
@pytest.mark.parametrize("query_len, seq_len", [(64, 64)])
@pytest.mark.parametrize("dim", [128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("scale", [1.0])
def test_flash_attention(head_q, head_kv, query_len, seq_len, dim, is_causal, scale):
  query = np.random.rand(head_q, query_len, dim).astype(np.float32)  # [head_q, query_len, dim]
  key = np.random.rand(head_kv, seq_len, dim).astype(np.float32)  # [head_kv, seq_len, dim]
  value = np.random.rand(head_kv, seq_len, dim).astype(np.float32)  # [head_kv, seq_len, dim]

  o = F.scaled_dot_product_attention(
      torch.tensor(query), torch.tensor(key), torch.tensor(value), is_causal=is_causal, scale=scale)
  o_np = o.numpy()  # [q_head,query,dim]

  o_actual = flash_attn(query, key, value, is_causal=is_causal, scale=scale)

  assert np.allclose(o_np, o_actual, atol=1e-7)


if __name__ == "__main__":
  pytest.main([__file__, "-vvs"])
```