---
title: 半监督学习：FixMatch
mathjax: true
toc: true
categories:
  - 深度学习
date: 2020-02-06 14:45:13
tags:
- 半监督学习
- Tensorflow
---

第十个算法`FixMatch: Simplifying Semi-Supervised Learning withConsistency and Confidence`，这依旧是谷歌研究组的作者提出的，是对`MixMatch`的改进。


<!--more-->

# 算法理论

`FixMatch`简而言之是一致性正则与伪标签的简单组合，他的主要创新点在于如何结合，以及在执行一致性正则是单独使用弱增强与强增强。这里的符号还是和之前的文章一样，新添加强增强符号为$\mathcal{A}(\cdot)$和弱增强$\alpha(\cdot)$。他的损失函数只由两个交叉熵组成：有监督的交叉熵$\ell_{s}$和无监督交叉熵$\ell_{u}$。对于有标签数据只使用弱增强并计算交叉熵：

$$
\begin{align}
\ell_{s}=\frac{1}{B} \sum_{b=1}^{B} \mathrm{H}\left(p_{b}, p_{\mathrm{m}}\left(y | \alpha\left(x_{b}\right)\right)\right)
\end{align}
$$


对于无标签数据，首先计算人工标签再用它来计算交叉熵。为了获得人工标签，首先计算给定的弱增强无标签图像的分布：$q_b=\_m(y|\alpha(u_b))$。然后使用$\hat{q}_{b}=\arg \max \left(q_{b}\right)$作为伪标签，接着使模型学习强增强的无标签样本的分类类别：
$$
\begin{align}
\ell_{u}=\frac{1}{\mu B} \sum_{b=1}^{\mu B} \mathbb{1}\left(\max \left(q_{b}\right) \geq \tau\right) \mathrm{H}\left(\hat{q}_{b}, p_{\mathrm{m}}\left(y | \mathcal{A}\left(u_{b}\right)\right)\right)
\end{align}
$$

其中$\tau$为保留阈值，如果伪标签的置信度不够高，那么需要丢弃这部分损失。综合损失为如下：$\ell_s+\lambda_u\ell_u$，其中$\lambda_u$为权重参数。


## FixMatch中的增强

其中弱增强为50%随机水平翻转，12.5%随机上下翻转，并添加一部分水平平移。对于强增强，使用了`RandAugment`和`CTAugment`。

对于`RandAugment`他使用一个全局的幅度来控制增强，幅度是通过验证集来优化的。不过论文中发现每个`step`直接从预定义范围中随机采样幅度即可获得良好的效果，方法类似与`UDA`。

对于`CTAugment`，在`ReMixMatch`讲过了。

## 其他重要因素

发现对于优化器，使用`Adam`反而效果不好。对于学习率衰减，使用余弦衰减比较好。

## 相关工作总结

| 算法                         | 人工标签增强 | 预测增强 | 人工标签后处理 | 备注                   |
| ---------------------------- | ------------ | -------- | -------------- | ---------------------- |
| Π-Model                      | 弱           | 弱       | 无             |                        |
| Temporal Ensembling          | 弱           | 弱       | 无             | 使用较早训练的模型     |
| Mean Teacher                 | 弱           | 弱       | 无             | 使用参数指数移动平滑   |
| Virtual Adversarial Training | 无           | 对抗     | 无             |                        |
| UDA                          | 无           | 强       | 锐化           | 忽略低置信度的人工标签 |
| MixMatch                     | 弱           | 弱       | 锐化           | 平均多个人工标签       |
| ReMixMatch                   | 弱           | 强       | 锐化           | 汇总多个预测的损失     |
| FixMatch                     | 弱           | 强       | 伪标签         |                        |
实际上经过`MixMatch`，`ReMixMatch`中大量的消融测试，作者应该已经找到了其中最重要的几个因素，所以这篇论文的理论部分比较少，因为之前的论文都已经介绍过了。实际上我认为主要加强点还是在于如何更好的使用一致性正则化，弱增强与弱增强间的所能学习到的一致性还不够，需要在弱增强与强增强间学习。同时对于人工标签的处理方式相当于选择如何进行熵最小化，也是次重要的。

## 实验结果


![](ssl-fixmatch/fixmatch-1.png)


## 消融测试

这里的消融测试主要就是在优化器上的了。

![](ssl-fixmatch/fixmatch-2.png)


# 代码

实际上他这里的训练流程代码不算复杂，但是他的数据增强部分我暂时也没有搞懂，需要花点时间仔细看看。

```python
AUGMENT_POOL_CLASS = AugmentPoolCTACutOut

def train(self, train_nimg, report_nimg):
    if FLAGS.eval_ckpt:
        self.eval_checkpoint(FLAGS.eval_ckpt)
        return
    batch = FLAGS.batch
    train_labeled = self.dataset.train_labeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
    train_labeled = train_labeled.batch(batch).prefetch(16).make_one_shot_iterator().get_next()
    train_unlabeled = self.dataset.train_unlabeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
    train_unlabeled = train_unlabeled.batch(batch * self.params['uratio']).prefetch(16)
    train_unlabeled = train_unlabeled.make_one_shot_iterator().get_next()
    scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=FLAGS.keep_ckpt,
                                                      pad_step_number=10))

    with tf.Session(config=utils.get_config()) as sess:
        self.session = sess
        self.cache_eval()

    with tf.train.MonitoredTrainingSession(
            scaffold=scaffold,
            checkpoint_dir=self.checkpoint_dir,
            config=utils.get_config(),
            save_checkpoint_steps=FLAGS.save_kimg << 10,
            save_summaries_steps=report_nimg - batch) as train_session:
        self.session = train_session._tf_sess()
        gen_labeled = self.gen_labeled_fn(train_labeled)
        gen_unlabeled = self.gen_unlabeled_fn(train_unlabeled)
        self.tmp.step = self.session.run(self.step)
        while self.tmp.step < train_nimg:
            loop = trange(self.tmp.step % report_nimg, report_nimg, batch,
                          leave=False, unit='img', unit_scale=batch,
                          desc='Epoch %d/%d' % (1 + (self.tmp.step // report_nimg), train_nimg // report_nimg))
            for _ in loop:
                self.train_step(train_session, gen_labeled, gen_unlabeled)
                while self.tmp.print_queue:
                    loop.write(self.tmp.print_queue.pop(0))
        while self.tmp.print_queue:
            print(self.tmp.print_queue.pop(0))

def model(self, batch, lr, wd, wu, confidence, uratio, ema=0.999, **kwargs):
    hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
    xt_in = tf.placeholder(tf.float32, [batch] + hwc, 'xt')  # Training labeled
    x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')  # Eval images
    y_in = tf.placeholder(tf.float32, [batch * uratio, 2] + hwc, 'y')  # Training unlabeled (weak, strong)
    l_in = tf.placeholder(tf.int32, [batch], 'labels')  # Labels

    lrate = tf.clip_by_value(tf.to_float(self.step) / (FLAGS.train_kimg << 10), 0, 1)
    lr *= tf.cos(lrate * (7 * np.pi) / (2 * 8))
    tf.summary.scalar('monitors/lr', lr)

    # Compute logits for xt_in and y_in
    classifier = lambda x, **kw: self.classifier(x, **kw, **kwargs).logits
    skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    logits = utils.para_cat(lambda x: classifier(x, training=True), tf.concat([xt_in, y_in[:, 0], y_in[:, 1]], 0))
    post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]
    logits_x = logits[:batch]
    logits_weak, logits_strong = tf.split(logits[batch:], 2)
    del logits, skip_ops

    # Labeled cross-entropy
    loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=l_in, logits=logits_x)
    loss_xe = tf.reduce_mean(loss_xe)
    tf.summary.scalar('losses/xe', loss_xe)

    # Pseudo-label cross entropy for unlabeled data
    pseudo_labels = tf.stop_gradient(tf.nn.softmax(logits_weak))
    loss_xeu = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(pseudo_labels, axis=1),
                                                              logits=logits_strong)
    pseudo_mask = tf.to_float(tf.reduce_max(pseudo_labels, axis=1) >= confidence)
    tf.summary.scalar('monitors/mask', tf.reduce_mean(pseudo_mask))
    loss_xeu = tf.reduce_mean(loss_xeu * pseudo_mask)
    tf.summary.scalar('losses/xeu', loss_xeu)

    # L2 regularization
    loss_wd = sum(tf.nn.l2_loss(v) for v in utils.model_vars('classify') if 'kernel' in v.name)
    tf.summary.scalar('losses/wd', loss_wd)

    ema = tf.train.ExponentialMovingAverage(decay=ema)
    ema_op = ema.apply(utils.model_vars())
    ema_getter = functools.partial(utils.getter_ema, ema)
    post_ops.append(ema_op)

    train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(
        loss_xe + wu * loss_xeu + wd * loss_wd, colocate_gradients_with_ops=True)
    with tf.control_dependencies([train_op]):
        train_op = tf.group(*post_ops)
```
# 测试结果

使用默认参数以及cifar10中250张标注样本训练128个epoch，得到测试集准确率如下，比`UDA`高一些：

```
"last01": 84.94000244140625,
"last10": 84.79999923706055,
"last20": 84.40499877929688,
"last50": 83.69499969482422
```

相比于`MixMatch`是又快又好。