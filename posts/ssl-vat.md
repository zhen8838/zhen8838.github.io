---
title: 半监督学习：Virtual Adversarial Training
mathjax: true
toc: true
categories:
  - 深度学习
date: 2020-01-31 15:10:42
tags:
- 半监督学习
- Tensorflow
---

第四个算法`Virtual Adversarial Training`(虚拟对抗训练)，出自论文`Virtual Adversarial Training:A Regularization Method for Supervised and Semi-Supervised Learning`,下面简称为`vat`。


<!--more-->

# 算法理论

经过三篇半监督算法的学习，可以发现半监督学习的精髓大概率就在于正则化(一致性)上面，`vat`是一种基于熵最小化出发的正则方法。他的出发点在于，我们要学习一致性那就得添加扰动，通常添加的随机扰动无法模拟各种复杂情况的输入，所以需要添加合适的扰动。因此根据一定的理论基础提出了如何找到合适的扰动，以及计算扰动的方法。我们记模型的损失函数为$J(\theta ;x;y)$，其中负梯度方向$− \nabla J_x(\theta ;x;y)$是模型的损失下降最快的方向，那么也就是说负梯度上模型优化最快， 为了使$\hat{x}$对模型的输出分布产生最大的改变，正梯度方向也就是模型梯度下降最慢的方向定为扰动方向，这就是`vat`需要添加扰动的方向。

-----

以下内容来自原论文：

`vat`的引入了`虚拟对抗方向`的概念，即扰动的方向，在输入数据中加入此方向上的扰动可以最大程度的影响模型分类输出概率分布，根据`虚拟对抗方向`的定义，可以在不使用监督信号的情况下，量化模型在每个输入点的局部各向异性，将`Local Distributional Smoothness (LDS) `定义为针对`虚拟对抗方向`的模型基于散度的分布鲁棒性，提出了一种新颖的使用有效的近似值，以最大程度地提高模型的熵，同时在每个训练输入数据点上提升模型的`LDS`，此方法成为`vat`。

`vat`的优点：
-   适用于半监督学习任务
-   适用于任何我们可以评估输入和参数梯度的参数模型
-   超参数数量少
-   参数化不变正则化

## 方法细节

首先定义符号：令$x\in R^I,y\in Q$表示输入向量与输出标签，$I$表示输入维度，$Q$表示标签空间。此外，我们将输出分布通过$\theta$参数化为$p(y|x,\theta)$，我们使用$\hat{\theta}$来表示模型参数在训练过程的特定迭代步骤的向量。使用$\mathcal{D}_{l}=\left\{x_{l}^{(n)}, y_{l}^{(n)} | n=1, \ldots, N_{l}\right\}$表示带标签数据集，$\mathcal{D}_{ul}=\left\{x_{ul}^{(m)}, y_{l}^{(m)} | m=1, \ldots, N_{ul}\right\}$表示无标签数据集，我们使用$\mathcal{D}_l,\mathcal{D}_{ul}$训练模型$p(y|x,\theta)$。

## 对抗训练

因为`vat`继承自`对抗训练`，因此，在介绍它之前，需要介绍对抗训练。对抗训练的损失函数可以写成：
$$
\begin{equation}
L_{\mathrm{adv}}\left(x_{l}, \theta\right):=D\left[q\left(y | x_{l}\right), p\left(y | x_{l}+r_{\mathrm{adv}}, \theta\right)\right]
\end{equation}\tag{1}
$$

$$
\begin{equation}
{\text { where } r_{\text {adv }}:=\underset{r ;\|r\| \leq \epsilon}{\arg \max } D\left[q\left(y | x_{l}\right), p\left(y | x_{l}+r, \theta\right)\right]}
\end{equation}\tag{2}
$$

其中$D$为分布$p$和$p'$间的差异度量函数，通常，我们无法获得精确的对抗性扰动$r_{adv}$的封闭形式，不过我们可以通过公式$2$中的度量$D$来线性近似$r$。当使用`l2`正则时，对抗扰动可以通过此公式近似：
$$
\begin{align}
r_{\mathrm{adv}} \approx \epsilon \frac{g}{\|g\|_{2}}, \text { where } g=\nabla_{x_{l}} D\left[h\left(y ; y_{l}\right), p\left(y | x_{l}, \theta\right)\right]
\end{align}\tag{3}
$$

当使用$L_{\infty}$正则时，对抗扰动可以通过此公式近似：
$$
\begin{aligned}
    r_{adv}\approx\epsilon sign(g)
\end{aligned}\tag{4}
$$

其中$g$和公式$3$相同。传统的对抗训练一般使用公式$3$来计算。

## 虚拟对抗训练

对抗训练是一种成功的方法，可以解决任何有监督的问题。但并非始终都有完整的标签信息。 令$x_*$代表任一$x_l$或$x_{ul}$，我们的目标函数现在为：
$$
\begin{align}
\begin{aligned}
&D\left[q\left(y | x_{*}\right), p\left(y | x_{*}+r_{\mathrm{qadv}}, \theta\right)\right]\\
&\text { where } r_{\text {qadv }}:=\underset{r ;\|r\| \leq \epsilon}{\arg \max } D\left[q\left(y | x_{*}\right), p\left(y | x_{*}+r, \theta\right)\right]
\end{aligned}
\end{align}
$$

实际上没有关于$q(x|x_{ul})$直接的标签信息，因此，我们采取了用当前近似值$p(y|x,\theta)$代替$q(y|x)$的策略，这种近似不一定是`naive`的，因为当带标签的训练样本数量很大时，$p(y|x,\theta)$应该接近$q(y|x)$。从字面上看，我们使用从$p(y|x,\theta)$概率生成的`虚拟`标签代替用户不知道的标签，并根据虚拟标签计算对抗方向。因此，使用当前估计值$p(y|x,\hat{\theta})$代替$q(y|x)$。有了这种折衷，我们得出了新的公式$2$的表达式：
$$
\begin{align}
\operatorname{LDS}\left(x_{*}, \theta\right):=D\left[p\left(y | x_{*}, \hat{\theta}\right), p\left(y | x_{*}+r_{\mathrm{vadv}}, \theta\right)\right]
\end{align}\tag{5}
$$

$$
\begin{align}
r_{\text {vadv }}:=\underset{r ;\|r\|_{2} \leq \epsilon}{\arg \max } D\left[p\left(y | x_{*}, \hat{\theta}\right), p\left(y | x_{*}+r\right)\right]
\end{align}\tag{6}
$$

$r_{\text {vadv }}$定义了我们的虚拟采样扰动，损失函数$\operatorname{LDS}( x_{*}, \theta)$可以视为对每个输入样本$x_{*}$当前模型的局部平滑度的否定度量，度量的减少将使模型在每个样本点处平滑。同时此损失的正则化项是所有输入样本点上的$\operatorname{LDS}(x_{*}, \theta)$的平均值：
$$
\begin{align}
\mathcal{R}_{\mathrm{vadv}}\left(\mathcal{D}_{l}, \mathcal{D}_{u l}, \theta\right):=\frac{1}{N_{l}+N_{u l}} \sum_{x_{*} \in \mathcal{D}_{l}, \mathcal{D}_{u l}} \operatorname{LDS}\left(x_{*}, \theta\right)
\end{align}\tag{7}
$$

最终得到完整的目标函数为：

$$
\begin{align}
\ell\left(\mathcal{D}_{l}, \theta\right)+\alpha \mathcal{R}_{\mathrm{vadv}}\left(\mathcal{D}_{l}, \mathcal{D}_{u l}, \theta\right)
\end{align}\tag{8}
$$

其中$\ell(\mathcal{D}_{l}, \theta)$是标记数据集的负对数似然。`vat`是使用正则化$\mathcal{R}_{\mathrm{vadv}}$的训练方法。

`vat`的一个显着优势是仅有两个标量值超参数：(1)对抗方向的范数约束$\epsilon>0$；(2)控制负对数似然与正则化器$\mathcal{R}_{\mathrm{vadv}}$之间相对平衡的正则化系数$\alpha>0$。实际上，根据实验，`vat`仅通过调整超参数(固定$\alpha=1$)而获得了出色的性能。



## $r_{\text{vadv}}$的快速逼近方法和目标函数的导数

为了简单起见，我们把$D\left[p\left(y | x_{*}, \hat{\theta}\right), p\left(y | x_{*}+r_{\mathrm{vadv}}, \theta\right)\right]$表示成$D(r,x_*,\theta)$
$r_{\text{vadv}}$。我们假设$p(y|x_*,\theta)$相对于$\theta$和几乎各处的$x$的差是两倍。因为$D(r,x_*,\theta)$取最小值时$r=0$，其一阶导数$\left.\nabla_{r} D(r, x, \hat{\theta})\right|_{r=0}$是`0`，因此$D$的第二项泰勒级数近似为：

$$
\begin{align}
D(r, x, \hat{\theta})=f(0)+f'(0)r+\frac{f''(0)}{2!}r^2 \approx \frac{1}{2} r^{T} H(x, \hat{\theta}) r
\end{align}\tag{9}
$$


$H(x, \hat{\theta})$的`Hessian矩阵`为$H(x,\hat{\theta}) = \nabla \nabla_r D(r,x,\hat{\theta}) \vert_{r=0}$。此时$r_{\text{vadv}}$成为$H(x, \hat{\theta})$的第一个特征向量$u(x,\theta)$，并且幅度为$\epsilon$(二次型在单位元上的最大值和最小值分别对应其最大特征值和最小特征值，此时$r$等于其对应的特征向量，这个具体的证明将Hermite矩阵正交对角化)：
$$
\begin{align}
\begin{aligned}
r_{\text {vadv }} & \approx \underset{r}{\arg \max }\left\{r^{T} H(x, \hat{\theta}) r ;\|r\|_{2} \leq \epsilon\right\} \\
&=\overline{\epsilon u(x, \hat{\theta})}
\end{aligned}
\end{align}\tag{10}
$$

其中$\overline{v}$表示方向与其参数向量$v$相同的单位向量，即$\bar{v} \equiv \frac{v}{\|v\|_{2}}$。下面为简单起见，用$H$表示$H(x, \hat{\theta})$。接下来，我们需要解决计算`Hessian矩阵`特征向量所需的$O(I^3)$运行时间。通过幂迭代法和有限差分法通过逼近来解决此问题。设$d$为随机采样的单位向量，$d$如果不垂直于主特征向量$u$，则迭代计算：
$$
\begin{equation}
d \gets \overline{Hd}
\end{equation}\tag{11}
$$

此时$d$是收敛到主特征向量$u$的，对于$H$的计算，不需要直接计算，而是计算近似有限差分：
$$
\begin{equation}
\begin{aligned}
Hd &\approx \frac{\nabla_r D(r,x,\hat{\theta}) \vert_{r=\xi d} -\nabla_r D(r,x,\hat{\theta})\vert_{r=0}}{\xi} \\
&= \frac{\nabla_r D(r,x,\hat{\theta})\vert_{r=\xi d}}{\xi}
\end{aligned}
\end{equation}\tag{12}
$$

其中$\xi \neq 0$，在上面的计算中，我们可以再次利用$\left.\nabla_{r} D(r, x, \hat{\theta})\right|_{r=0}=0$。总而言之，我们可以通过以下更新的重复应用来近似$r_{\text{vadv}}$：
$$
\begin{align}
d \leftarrow \overline{\nabla_{r} D(r, x, \hat{\theta})|_{r=\xi d}}
\end{align}\tag{13}
$$

在幂迭代下，这种近似可以由迭代次数$K$来单调改善，在实验中$K=1$就可以实现较好的结果了，此时可以对$r_{\text{vadv}}$进一步改写为：
$$
\begin{aligned}
    r_{\text{vadv}} \approx \epsilon\frac{g}{\|g\|_2}
\end{aligned}\tag{14}
$$

$$
\begin{aligned}
    \text{where}\ g=\left. \nabla_{r} D[p(y | x, \hat{\theta}), p(y | x+r, \hat{\theta})]\right|_{r=\xi d}
\end{aligned}\tag{15}
$$

计算$r_{\text{vadv}}$之后，可以使用神经网络中进行的正向和反向传播轻松计算$r_{\text{vadv}}$的导数。但是，加入$r_{\text{vadv}}$相对于参数的导数，不仅无用并且计算代价高，而且还为梯度引入了另一种方差来源，并对算法的性能产生负面影响。因此`vat`忽略了$r_{\text{vadv}}$对于$\theta$的依赖性。总体而言，包括对数似然项公式$8$在内的全目标函数的导数可以用$K + 2$组反向传播来计算。具体迭代过程伪代码如下：

![](ssl-vat/vat-1.png)

对于幂迭代次数$K$，可以对`vat`的正则项做一个表述：
$$
\begin{equation}
\mathcal R^{(K)}(\theta ,\mathcal D_l,\mathcal D_{ul}) := \frac{1}{N_l + N_{ul}} \sum_{x \in \mathcal D_l,\mathcal D_{ul}} \mathbb E_{r_K}[D[p(y \vert x,\hat{\theta}), p(y \vert x+r_K,\theta)]]
\end{equation}\tag{16}
$$

对于`vat`就是幂迭代次数大于等于1次，即$K\geq1$。当$K=0$时，也就是不采用幂迭代求解$r_{\text{vadv}}$，称这种方法为`rpt`，`rpt`是`vat`的降级版本， 不执行幂迭代，`rpt`仅在每个输入数据点周围各向同性地平滑函数。


# 代码

1.  总体流程

```python
hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
xt_in = tf.placeholder(tf.float32, [batch] + hwc, 'xt')  # For training
x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
y_in = tf.placeholder(tf.float32, [batch] + hwc, 'y')
l_in = tf.placeholder(tf.int32, [batch], 'labels')
wd *= lr
warmup = tf.clip_by_value(tf.to_float(self.step) / (warmup_pos * (FLAGS.train_kimg << 10)), 0, 1)

classifier = lambda x, **kw: self.classifier(x, **kw, **kwargs).logits
l = tf.one_hot(l_in, self.nclass)
# 带标签数据概率分布
logits_x = classifier(xt_in, training=True)
post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Take only first call to update batch norm.
logits_y = classifier(y_in, training=True) # 无标签数据概率分布
# 计算对当前无标签数据需要添加的扰动
delta_y = vat_utils.generate_perturbation(y_in, logits_y, lambda x: classifier(x, training=True), vat_eps)
# （无标签数据+扰动）=> 学生模型输出概率分布
logits_student = classifier(y_in + delta_y, training=True)
# （无标签数据）=> 教师模型输出概率分布
logits_teacher = tf.stop_gradient(logits_y)
# 利用kl散度损失学习一致性
loss_vat = layers.kl_divergence_from_logits(logits_student, logits_teacher)
loss_vat = tf.reduce_mean(loss_vat)
# 最小化无监督概率分布的熵
loss_entmin = tf.reduce_mean(tf.distributions.Categorical(logits=logits_y).entropy())

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=l, logits=logits_x)
loss = tf.reduce_mean(loss)
tf.summary.scalar('losses/xe', loss)
tf.summary.scalar('losses/vat', loss_vat)
tf.summary.scalar('losses/entmin', loss_entmin)

ema = tf.train.ExponentialMovingAverage(decay=ema)
ema_op = ema.apply(utils.model_vars())
ema_getter = functools.partial(utils.getter_ema, ema)
post_ops.append(ema_op)
post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

train_op = tf.train.AdamOptimizer(lr).minimize(loss + loss_vat * warmup * vat + entmin_weight * loss_entmin,
                                                colocate_gradients_with_ops=True)
with tf.control_dependencies([train_op]):
    train_op = tf.group(*post_ops)
```


2.  扰动计算

```python
def generate_perturbation(x, logit, forward, epsilon, xi=1e-6):
    """Generate an adversarial perturbation.

    Args:
        x: Model inputs.
        logit: Original model output without perturbation.
        forward: Callable which computs logits given input.
        epsilon: Gradient multiplier.
        xi: Small constant.

    Returns:
        Aversarial perturbation to be applied to x.
    """
    d = tf.random_normal(shape=tf.shape(x))
    # 迭代次数为1
    for _ in range(1):
        # 向量d需要单位化
        d = xi * get_normalized_vector(d)
        logit_p = logit
        logit_m = forward(x + d) # 无标签样本+随机噪声的概率分布输出
        dist = kl_divergence_with_logit(logit_p, logit_m) # 计算分布距离度量
        # 求得度量梯度（累积求和）
        grad = tf.gradients(tf.reduce_mean(dist), [d], aggregation_method=2)[0]
        # 删除此操作产生的梯度
        d = tf.stop_gradient(grad)
    # 输出向量r_adv也是单位化向量，同时乘epsilon
    return epsilon * get_normalized_vector(d)

def get_normalized_vector(d):
    """Normalize d by infinity and L2 norms."""
    d /= 1e-12 + tf.reduce_max(
        tf.abs(d), list(range(1, len(d.get_shape()))), keepdims=True
    )
    d /= tf.sqrt(
        1e-6
        + tf.reduce_sum(
            tf.pow(d, 2.0), list(range(1, len(d.get_shape()))), keepdims=True
        )
    )
    return d
```


# 测试结果


使用默认参数以及cifar10中250张标注样本训练128个epoch，得到测试集准确率如下，比之前的算法的确有提升：

```
"last01": 54.720001220703125,
"last10": 54.28499984741211,
"last20": 54.27000045776367,
"last50": 53.93499946594238
```