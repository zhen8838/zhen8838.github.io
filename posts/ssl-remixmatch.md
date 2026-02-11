---
title: 半监督学习：ReMixMatch
mathjax: true
toc: true
categories:
  - 深度学习
date: 2020-02-05 15:39:31
tags:
- 半监督学习
- Tensorflow
---

第九个算法`ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring`，这也是谷歌`MixMatch`的同一作者提出的，是对`MixMatch`的改进。

<!--more-->

# 算法理论

通过引入两个技术:`Distribution Alignment`和`Augmentation Anchoring`改进了`MixMatch`。

## Distribution Alignment

分布对齐的目标是使无标签数据的预测汇总与提供的标签数据分布相匹配。这个概念是25年前的`Unsupervised classifiers, mutualinformation and’phantom targets`所引入的，但是在`ReMixMatch`之前还没有人在半监督学习中用过这个方法。

![分布对齐.根据经验性的ground-truth类别分布除以未标记数据的平均预测的比例调整猜测标签的分布](ssl-remixmatch/remixmatch-1.png)

半监督算法的主要目标是利用未标记数据提升模型性能，`bridle`等人首先提出一种这种直觉形式化的方法，最大化无标签数据的输入与输出间的互信息。将此操作公式化为如下，以下公式推导时，假设了$p(x),p(y)$相互独立则有$p(y) = \int p(x)p(y|x)\ dx$。：


$$
\begin{align}
\mathcal{I}(y ; x) &=\iint p(y, x) \log \frac{p(y, x)}{p(y) p(x)} \mathrm{d} y \mathrm{d} x\\
&=\int p(x) \mathrm{d} x \int p(y | x) \log \frac{p(y | x)}{p(y)} \mathrm{d} y\\
&=\int p(x) \mathrm{d} x \int p(y | x) \log \frac{p(y | x)}{\int p(x) p(y | x) \mathrm{d} x} \mathrm{d} y\\
&=\mathbb{E}_{x}\left[\int p(y | x) \log \frac{p(y | x)}{\mathbb{E}_{x}[p(y | x)]} \mathrm{d} y\right]\\
\text{离散化:}\\
&=\mathbb{E}_{x}\left[\sum_{i=1}^{L} p\left(y_{i} | x\right) \log \frac{p\left(y_{i} | x\right)}{\mathbb{E}_{x}\left[p\left(y_{i} | x\right)\right]}\right]\\
&=\mathbb{E}_{x}\left[\sum_{i=1}^{L} p\left(y_{i} | x\right) \log p\left(y_{i} | x\right)\right]-\mathbb{E}_{x}\left[\sum_{i=1}^{L} p\left(y_{i} | x\right) \log \mathbb{E}_{x}\left[p\left(y_{i} | x\right)\right]\right]\\
&=\mathbb{E}_{x}\left[\sum_{i=1}^{L} p\left(y_{i} | x\right) \log p\left(y_{i} | x\right)\right]-\sum_{i=1}^{L} \mathbb{E}_{x}\left[p\left(y_{i} | x\right)\right] \log \mathbb{E}_{x}\left[p\left(y_{i} | x\right)\right]
\end{align}\tag{1}
$$
$$
\begin{align}
 &=\mathcal{H}\left(\mathbb{E}_{x}\left[p_{\text {model }}(y | x ; \theta)\right]\right)-\mathbb{E}_{x}\left[\mathcal{H}\left(p_{\text {model }}(y | x ; \theta)\right)\right] 
\end{align}\tag{2}
$$


其中$\mathcal{H}(\cdot)$为熵。其中公式2是熟悉的熵最小化目标，它简单的鼓励模型输出具有更低的熵(对当前类别标签置信度更高)。但其中公式1并未广泛使用，其目标是鼓励模型在整个训练集平均预测每个类别的频率相同，`bridle`等人称之为公平性。


在`MixMatch`中已经使用了`sharpening`函数使猜测标签的熵最小化。现在要通过互信息的概念引入`公平性`这一原则，注意目标$\mathcal{H}\left(\mathbb{E}_{x}\left[p_{\text {model }}(y | x ; \theta)\right]\right)$本来已经暗示了它应该以相同的频率预测每个标签，但如果数据集中的$p(y)$的分布并不是均匀的，那这个目标就不一定有效了。虽然可以按`batch`最小化这个目标，但是为了不引入更多的超参数，因此为了解决以上问题，引入了另外一种`公平性`形式`Distribution Alignment`。其过程如下：

训练过程中维持模型对未标记数据的预测结果的平均值$\tilde{p}(y)$,给定模型在未标记数据$u$上的预测为$q=P_{\text{model}}(y|u,\theta)$，我们将利用$\frac{p(y)}{\tilde{p}(y)}$作为比例缩放$q$，然后在重新放大到有效的概率分布区间： $\tilde{q}=\text{Normalize}(q\times\frac{p(y)}{\tilde{p}(y)})$，其中`Normalize`为$\text{Normalize}(x)_i=\frac{x_i}{\sum_j x_j}$。 然后我们使用$\tilde{q}$作为$u$的猜测标签，然后可以再用`sharpening`或其他的处理方式。实际操作中，将计算过去`128`个`batch`中无标签数据预测值的滑动平均作为$\tilde{p}(y)$，如果我们直接知道$\tilde{p}(y)$的某些先验分布，那么应该还可以更好。



## 改进一致性正则化

论文中说，使用了最新提出`AutoAugment`数据增强算法来代替原本`MixMatch`中的数据弱增强看看能不能提高性能，但是发现训练并不能收敛。因此提出了一个解决方法`Augmentation Anchoring`，它的基本想法是将模型对弱增强的未标记图像的预测结果作为同一图像的强增强的猜测标签。

同时因为`AutoAugment`是使用强化学习策略来搜索的，需要对有监督模型做多次尝试。在半监督学习中难以做到，为了解决这个问题，提出了一个名为`CTAugment`的方法，使用控制理论的思想在线适应，而无需任何形式的基于强化学习的训练。

### Augmentation Anchoring

我们假设带有`AutoAugment`的`MixMatch`不稳定的原因是`MixMatch`对$K$个的预测取了平均值。由于增强效果可能会导致不同的预测，因此其平均值可能不是有意义的目标，取而代之的是，给定一个未标记的输入$u$，我们首先通过对其应用弱增强来生成一个`Anchor`。然后使用`CTAugment`生成$K$个$u$的增强，然后将(经过`distribution alignment`和`sharpening`后的)猜测标签作为$K$个增强后的目标。

![](ssl-remixmatch/remixmatch-2.png)

在实验中发现，使用`Augmentation Anchoring`之后，可以直接使用交叉熵代替原本的`mse`损失，更易于实现，同时$K=2$即可取得不错的效果，当然$K>8$效果更好。

### Control Theory Augment

像`AutoAugment`一样，`CTAugment`均匀的随机采样要实施的变换，但是会在训练过程中动态推断每次变换的幅度大小。由于`CTAugment`具有不敏感的超参数，因此可以直接包含在半监督模型中。直观的，对于每个建议的参数，`CTAugment`都知道它将产生被分类正确标签的图像的概率，然后使用这些概率，仅对网络可忍受范围内的误差进行采样。这个过程在`FastAutoAugment`中被称为`density-matching`。

首先，`CTAugment`将每个变化的每个参数范围划分为数个分组，在开始训练时将每个分组的权重设置为`1`，然后令权重向量$m$向某些分组变化，这些权重决定了那些幅度级别是需要实施变化的。在每个训练`step`中，对于每个图像随机地均匀采样两个变换，用于图像增强。使用改变过的权重参数$\bar{m}$，其中$\bar{m}_i=m_i\ \ \text{if}\ \  m_i>0.8\ \ \text{and}\ \ \bar{m}_i =0 $，否则使用$\bar{m}$作为权重进行随机分类采样。为了更新权重，首先随机地对每个转换参数均匀的采样一个$m_i$，将结果转换应用于带标签样本$x$以获得增强版本$\bar{x}$，然后测量模型的预测与标签的匹配程度为$\omega=1-\frac{1}{2L}\sum|p_{\text{model}}(y|\bar{x};\theta)-p|$，每个采样权重的权重更新为$m_i=\rho m_i+(1-\rho)\omega$，其中$\rho=0.99$是固定的指数衰减超参数。

### 综合

综合算法流程如下：


![](ssl-remixmatch/remixmatch-3.png)


主要是生成两个集合$\mathcal{X}'$和$\mathcal{U}'$，由增强后的带标记的有标签无标签数据`mixup`生成。$\mathcal{X}'$和$\mathcal{U}'$的标签与猜测标签根据模型预测输入到标准的交叉熵损失中。还有$\mathcal{U}_1$是由无标签数据经过单个强增强组成的，并且他的猜测标签没有应用`mixup`，$\mathcal{U}_1$是用在两个额外的损失项中，它能提供很大的改善。

`Pre-mixup unlabeled loss`： 将$\mathcal{U}_1$的猜测标签和预测输入一个单独的交叉熵损失项。

`Rotation loss` ：最近的结果表明，将自我监督学习的思想应用于半监督学习可以产生出色的性能( Self-supervised semi-supervised learning)。将这个想法通过旋转每个图像$\text{Rotate}(u,r) \in \mathcal{U}_1$来整合，$r \sim {0,90,180,270}$，然后要求模型将旋转量预测为四类分类问题。

$$
\begin{align}
\begin{aligned}
\sum_{x, p \in \mathcal{X}^{\prime}} \mathrm{H}\left(p, p_{\text {model }}(y | x ; \theta)\right)+\lambda_{\mathcal{U}} \sum_{u, q \in \mathcal{U}^{\prime}} \mathrm{H}\left(q, p_{\text {model }}(y | u ; \theta)\right) \\
+\lambda_{\hat{u}_{1}} \sum_{u, q \in \hat{\mathcal{U}}_{1}} \mathrm{H}\left(q, p_{\text {model }}(y | u ; \theta)\right)+\lambda_{r} \sum_{u \in \hat{\mathcal{U}}_{1}} \mathrm{H}\left(r, p_{\text {model }}(r | \text { Rotate }(u, r) ; \theta)\right)
\end{aligned}
\end{align}
$$

根据消融测试结果：

![](ssl-remixmatch/remixmatch-4.png)

如果没有弱增强和强增强间的`augment anchoring`错误率就立马上升非常多。其次是将`guess label`的损失从交叉熵变成`l2 loss`，不过这里我挺奇怪的，之前其他的算法都是说`l2 loss`的约束性更大，效果会更好。

# 代码

```python
def classifier_rot(self, x):
    # 旋转目标分类器
    with tf.variable_scope('classify_rot', reuse=tf.AUTO_REUSE):
        return tf.layers.dense(x, 4, kernel_initializer=tf.glorot_normal_initializer())

def guess_label(self, logits_y, p_data, p_model, T, use_dm, redux, **kwargs):
    del kwargs
    if redux == 'swap':
        p_model_y = tf.concat([tf.nn.softmax(x) for x in logits_y[1:] + logits_y[:1]], axis=0)
    elif redux == 'mean':
        p_model_y = sum(tf.nn.softmax(x) for x in logits_y) / len(logits_y)
        p_model_y = tf.tile(p_model_y, [len(logits_y), 1])
    elif redux == '1st':
        p_model_y = tf.nn.softmax(logits_y[0])
        p_model_y = tf.tile(p_model_y, [len(logits_y), 1])
    else:
        raise NotImplementedError()

    # 计算目标分布
    # 默认是使用分布match的
    if use_dm:
        p_ratio = (1e-6 + p_data) / (1e-6 + p_model)
        p_weighted = p_model_y * p_ratio
        p_weighted /= tf.reduce_sum(p_weighted, axis=1, keep_dims=True)
    else:
        p_weighted = p_model_y
    # 再进行锐化
    p_target = tf.pow(p_weighted, 1. / T)
    p_target /= tf.reduce_sum(p_target, axis=1, keep_dims=True)
    return EasyDict(p_target=p_target, p_model=p_model_y)

def model(self, batch, lr, wd, beta, w_kl, w_match, w_rot, K, use_xe, warmup_kimg=1024, T=0.5,
          mixmode='xxy.yxy', dbuf=128, ema=0.999, **kwargs):
    hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
    xt_in = tf.placeholder(tf.float32, [batch] + hwc, 'xt')  # For training
    x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
    # K个强增强 + 1一个弱增强
    y_in = tf.placeholder(tf.float32, [batch, K + 1] + hwc, 'y')
    l_in = tf.placeholder(tf.int32, [batch], 'labels')

    w_match *= tf.clip_by_value(tf.cast(self.step, tf.float32) / (warmup_kimg << 10), 0, 1)
    lrate = tf.clip_by_value(tf.to_float(self.step) / (FLAGS.train_kimg << 10), 0, 1)
    lr *= tf.cos(lrate * (7 * np.pi) / (2 * 8))
    tf.summary.scalar('monitors/lr', lr)
    augment = layers.MixMode(mixmode)
    # 使用多gpu并行分类
    gpu = utils.get_gpu()

    def classifier_to_gpu(x, **kw):
        with tf.device(next(gpu)):
            return self.classifier(x, **kw, **kwargs).logits

    def random_rotate(x):
        b4 = batch // 4
        x, xt = x[:2 * b4], tf.transpose(x[2 * b4:], [0, 2, 1, 3])
        l = np.zeros(b4, np.int32)
        l = tf.constant(np.concatenate([l, l + 1, l + 2, l + 3], axis=0))
        return tf.concat([x[:b4], x[b4:, ::-1, ::-1], xt[:b4, ::-1], xt[b4:, :, ::-1]], axis=0), l

    # 当前估计标签的分布滑动平均值
    p_model = layers.PMovingAverage('p_model', self.nclass, dbuf)
    # 当前标签的分布滑动平均值，用于绘图观察真实标签的分布情况
    p_target = layers.PMovingAverage('p_target', self.nclass, dbuf)  # Rectified distribution (only for plotting)

    # 推断真实无标签数据的分布
    p_data = layers.PData(self.dataset)
    
    # 旋转数据并进行分类
    if w_rot > 0:
        rot_y, rot_l = random_rotate(y_in[:, 1])
        with tf.device(next(gpu)):
            rot_logits = self.classifier_rot(self.classifier(rot_y, training=True, **kwargs).embeds)
        loss_rot = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(rot_l, 4), logits=rot_logits)
        loss_rot = tf.reduce_mean(loss_rot)
        tf.summary.scalar('losses/rot', loss_rot)
    else:
        loss_rot = 0
    # 这里是对于guess label处理过程，因为有多个y的预测值，可以选择其一也可以平均他。
    if kwargs['redux'] == '1st' and w_kl <= 0:
        logits_y = [classifier_to_gpu(y_in[:, 0], training=True)] * (K + 1)
    elif kwargs['redux'] == '1st':
        logits_y = [classifier_to_gpu(y_in[:, i], training=True) for i in range(2)]
        logits_y += logits_y[:1] * (K - 1)
    else:
        logits_y = [classifier_to_gpu(y_in[:, i], training=True) for i in range(K + 1)]
    # 做augment anchor损失
    guess = self.guess_label(logits_y, p_data(), p_model(), T=T, **kwargs)
    ly = tf.stop_gradient(guess.p_target)
    if w_kl > 0:
        w_kl *= tf.clip_by_value(tf.cast(self.step, tf.float32) / (warmup_kimg << 10), 0, 1)
        loss_kl = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ly[:batch], logits=logits_y[1])
        loss_kl = tf.reduce_mean(loss_kl)
        tf.summary.scalar('losses/kl', loss_kl)
    else:
        loss_kl = 0
    del logits_y
    # mixup
    lx = tf.one_hot(l_in, self.nclass)
    xy, labels_xy = augment([xt_in] + [y_in[:, i] for i in range(K + 1)], [lx] + tf.split(ly, K + 1),
                            [beta, beta])
    x, y = xy[0], xy[1:]
    labels_x, labels_y = labels_xy[0], tf.concat(labels_xy[1:], 0)
    del xy, labels_xy

    batches = layers.interleave([x] + y, batch)
    logits = [classifier_to_gpu(yi, training=True) for yi in batches[:-1]]
    skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    logits.append(classifier_to_gpu(batches[-1], training=True))
    post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]
    logits = layers.interleave(logits, batch)
    logits_x = logits[0]
    logits_y = tf.concat(logits[1:], 0)
    del batches, logits
    # 分类损失
    loss_xe = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_x, logits=logits_x)
    loss_xe = tf.reduce_mean(loss_xe)
    if use_xe:
        loss_xeu = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_y, logits=logits_y)
    else:
        loss_xeu = tf.square(labels_y - tf.nn.softmax(logits_y))
    loss_xeu = tf.reduce_mean(loss_xeu)
    tf.summary.scalar('losses/xe', loss_xe)
    tf.summary.scalar('losses/%s' % ('xeu' if use_xe else 'l2u'), loss_xeu)
    self.distribution_summary(p_data(), p_model(), p_target())

    # L2 regularization
    loss_wd = sum(tf.nn.l2_loss(v) for v in utils.model_vars('classify') if 'kernel' in v.name)
    tf.summary.scalar('losses/wd', loss_wd)

    ema = tf.train.ExponentialMovingAverage(decay=ema)
    ema_op = ema.apply(utils.model_vars())
    ema_getter = functools.partial(utils.getter_ema, ema)
    post_ops.extend([ema_op,
                     p_model.update(guess.p_model),
                     p_target.update(guess.p_target)])
    if p_data.has_update:
        post_ops.append(p_data.update(lx))

    train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(
        loss_xe + w_kl * loss_kl + w_match * loss_xeu + w_rot * loss_rot + wd * loss_wd,
        colocate_gradients_with_ops=True)
    with tf.control_dependencies([train_op]):
        train_op = tf.group(*post_ops)
```


# 实验