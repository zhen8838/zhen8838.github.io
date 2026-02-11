---
title: Lookahead优化器的tf.Keras实现
mathjax: true
toc: true
categories:
  - 深度学习
date: 2019-10-20 20:09:27
tags:
- Tensorflow
- 优化器
---

论文《[Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610)》的`tf.Keras`实现.

参考自苏剑林的[repo](https://github.com/bojone/keras_lookahead)


<!--more-->

# tf 1.14 的实现

因为`tf.keras`的`keras`改动有点大,所以这里的实现和原本的不一样.

```python
# NOTE from https://github.com/bojone/keras_lookahead
class Lookahead(object):
    """Add the [Lookahead Optimizer](https://arxiv.org/abs/1907.08610) functionality for [keras](https://keras.io/).
    """

    def __init__(self, k=5, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.count = 0

    def inject(self, model: keras.models.Model):
        """Inject the Lookahead algorithm for the given model.
        The following code is modified from keras's _make_train_function method.
        See: https://github.com/keras-team/keras/blob/master/keras/engine/training.py#L497
        """
        if not hasattr(model, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')

        model._check_trainable_weights_consistency()
        metrics_tensors = [
            model._all_metrics_tensors[m] for m in model.metrics_names[1:]
        ]
        if model.train_function is None:
            inputs = (model._feed_inputs +
                      model._feed_targets +
                      model._feed_sample_weights)
            if not isinstance(K.symbolic_learning_phase(), int):
                inputs += [K.symbolic_learning_phase()]
            fast_params = model._collected_trainable_weights

            with K.name_scope('training'):
                with K.name_scope(model.optimizer.__class__.__name__):
                    training_updates = model.optimizer.get_updates(
                        params=fast_params,
                        loss=model.total_loss)
                    slow_params = [K.variable(p) for p in fast_params]

                fast_updates = (model.updates +
                                training_updates +
                                model.get_updates_for(None) +
                                model.get_updates_for(model.inputs))

                slow_updates, copy_updates = [], []
                for p, q in zip(fast_params, slow_params):
                    slow_updates.append(K.update(q, q + self.alpha * (p - q)))
                    copy_updates.append(K.update(p, q))

                # Gets loss and metrics. Updates weights at each call.
                fast_train_function = K.function(
                    inputs, [model.total_loss] + metrics_tensors,
                    updates=fast_updates,
                    name='fast_train_function',
                    **model._function_kwargs)

                def F(inputs):
                    self.count += 1
                    R = fast_train_function(inputs)
                    if self.count % self.k == 0:
                        K.batch_get_value(slow_updates)
                        K.batch_get_value(copy_updates)
                    return R

                model.train_function = F
```


# tf 1.15 的实现

因为新版本的`tf.keras`的`keras`改动又有点大,所以这里的实现和原本的又不一样.

```python
class Lookahead(object):
    """Add the [Lookahead Optimizer](https://arxiv.org/abs/1907.08610) functionality for [keras](https://keras.io/).
    """

    def __init__(self, k=5, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.count = 0

    def inject(self, model: keras.models.Model):
        """ from tensorflow.keras `_make_train_function` refer from
         https://github.com/tensorflow/tensorflow/blob/590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b/tensorflow/python/keras/engine/training.py#L2091 and https://github.com/bojone/keras_lookahead/blob/master/lookahead.py
        """
        has_recompiled = model._recompile_weights_loss_and_weighted_metrics()
        model._check_trainable_weights_consistency()
        if isinstance(model.optimizer, list):
            raise ValueError('The `optimizer` in `compile` should be a single '
                             'optimizer.')
        # If we have re-compiled the loss/weighted metric sub-graphs then create
        # train function even if one exists already. This is because
        # `_feed_sample_weights` list has been updated on re-copmpile.
        if getattr(self, 'train_function', None) is None or has_recompiled:
            # Restore the compiled trainable state.
            current_trainable_state = model._get_trainable_state()
            model._set_trainable_state(model._compiled_trainable_state)

            inputs = (model._feed_inputs +
                      model._feed_targets +
                      model._feed_sample_weights)
            if not isinstance(K.symbolic_learning_phase(), int):
                inputs += [K.symbolic_learning_phase()]

            fast_params = model._collected_trainable_weights

            with K.get_graph().as_default():
                with K.name_scope('training'):
                    # Training updates
                    training_updates = model.optimizer.get_updates(
                        params=fast_params, loss=model.total_loss)
                    slow_params = [K.variable(p) for p in fast_params]

                    fast_updates = (
                        training_updates +
                        # Unconditional updates
                        model.get_updates_for(None) +
                        # Conditional updates relevant to this model
                        model.get_updates_for(model.inputs))

                metrics = model._get_training_eval_metrics()
                metrics_tensors = [
                    m._call_result for m in metrics if hasattr(m, '_call_result')  # pylint: disable=protected-access
                ]

            with K.name_scope('training'):
                slow_updates, copy_updates = [], []
                for p, q in zip(fast_params, slow_params):
                    slow_updates.append(K.update(q, q + self.alpha * (p - q)))
                    copy_updates.append(K.update(p, q))

                # Gets loss and metrics. Updates weights at each call.
                fast_train_function = K.function(
                    inputs, [model.total_loss] + metrics_tensors,
                    updates=fast_updates,
                    name='train_function',
                    **model._function_kwargs)

                def F(inputs):
                    self.count += 1
                    R = fast_train_function(inputs)
                    if self.count % self.k == 0:
                        K.batch_get_value(slow_updates)
                        K.batch_get_value(copy_updates)
                    return R

                setattr(model, 'train_function', F)

            # Restore the current trainable state
            model._set_trainable_state(current_trainable_state)
```

# tf2.0的实现

```python
# NOTE from https://github.com/bojone/keras_lookahead
class Lookahead(object):
    """Add the [Lookahead Optimizer](https://arxiv.org/abs/1907.08610) functionality for [keras](https://keras.io/).
    """

    def __init__(self, k=5, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.count = 0

    def inject(self, model: k.models.Model):
        has_recompiled = model._recompile_weights_loss_and_weighted_metrics()
        model._check_trainable_weights_consistency()
        if isinstance(model.optimizer, list):
            raise ValueError('The `optimizer` in `compile` should be a single '
                             'optimizer.')
        # If we have re-compiled the loss/weighted metric sub-graphs then create
        # train function even if one exists already. This is because
        # `_feed_sample_weights` list has been updated on re-copmpile.
        if getattr(model, 'train_function', None) is None or has_recompiled:
            current_trainable_state = model._get_trainable_state()
            model._set_trainable_state(model._compiled_trainable_state)

            inputs = (model._feed_inputs +
                      model._feed_targets +
                      model._feed_sample_weights)
            if not isinstance(K.symbolic_learning_phase(), int):
                inputs += [K.symbolic_learning_phase()]

            with K.get_graph().as_default():
                with K.name_scope('training'):
                    # Training updates
                    fast_params = model._collected_trainable_weights
                    training_updates = model.optimizer.get_updates(
                        params=fast_params, loss=model.total_loss)
                    slow_params = [K.variable(p) for p in fast_params]

                    fast_updates = (
                        training_updates +
                        model.get_updates_for(None) +
                        model.get_updates_for(model.inputs)
                    )
                metrics = model._get_training_eval_metrics()
                metrics_tensors = [
                    m._call_result for m in metrics if hasattr(m, '_call_result')  # pylint: disable=protected-access
                ]

            with K.name_scope('training'):
                slow_updates, copy_updates = [], []
                for p, q in zip(fast_params, slow_params):
                    slow_updates.append(K.update(q, q + self.alpha * (p - q)))
                    copy_updates.append(K.update(p, q))

                # Gets loss and metrics. Updates weights at each call.
                fast_train_function = K.function(
                    inputs, [model.total_loss] + metrics_tensors,
                    updates=fast_updates,
                    name='fast_train_function',
                    **model._function_kwargs)

                def F(inputs):
                    self.count += 1
                    R = fast_train_function(inputs)
                    if self.count % self.k == 0:
                        K.batch_get_value(slow_updates)
                        K.batch_get_value(copy_updates)
                    return R

                setattr(model, 'train_function', F)
            # Restore the current trainable state
            model._set_trainable_state(current_trainable_state)
```