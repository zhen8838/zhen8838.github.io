---
title: Tensorflow中动态学习率无效
categories:
  - 深度学习
date: 2019-01-27 15:03:46
tags:
-   踩坑经验
-   Tensorflow
---

我昨天刚刚上传了`Mobilenet Flowers`项目,今天在修改勘智官方的的`demo`,我简单粗暴的把他的代码改成我的写法,然后测试,忽然发现我的动态学习率一直不变.找了半天才解决.

<!--more-->

## 问题解决
我的是把`train_op`使用一个函数输入的,问题就是在这个函数的`minimize`的时候,我忘记把第二个参数`global_step`传入了.在我以前没有被函数包裹的情况下,可以不传入这个参数,现在必须传入.
```python
    train_op = create_train_op(total_loss, global_step, args.optimizer,
                               current_learning_rate, args.moving_average_decay)
    .
    .
    .
    
def create_train_op(total_loss, global_step, optimizer, learning_rate, moving_average_decay):
    # Generate moving averages of all losses and associated summaries.
    # loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    # with tf.control_dependencies([loss_averages_op]):
    if optimizer == 'ADAGRAD':
        opt = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer == 'ADADELTA':
        opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
    elif optimizer == 'ADAM':
        opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
    elif optimizer == 'RMSPROP':
        opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
    elif optimizer == 'MOM':
        opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True)
    else:
        raise ValueError('Invalid optimization algorithm')
    # ! 因为minimize中包含了compute_gradients和apply_gradients,所以他们的思路是:
    # ! 首先计算梯度,然后给梯度增加滑动平均,最后把滑动平均的梯度应用到梯度下降
    #     grads = opt.compute_gradients(total_loss, update_gradient_vars)

    # # Apply gradients.
    # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # # Track the moving averages of all trainable variables.
    # variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    # variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    #     train_op = tf.no_op(name='train')
    # ! 我先用原始的方式进行训练
    train_op = opt.minimize(total_loss, global_step)

    return train_op

```
