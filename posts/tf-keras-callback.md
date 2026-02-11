---
title: 测试tf.keras中callback的运行状态
mathjax: true
toc: true
categories:
  - 深度学习
date: 2019-11-13 16:03:57
tags:
-   Tensorflow
-   Yolo
-   Keras
---

要给`yolo`添加多尺度训练,因为`tf.keras`无法对`dataset`对象进行`callback`操作这也就算了,但是我没法得知`dataset`对象目前在生成训练数据还是测试数据,这个就很蛋疼,需要能在尽量不大改代码的同时添加多尺度训练方式,所以还得看`tf.keras.callback`.

<!--more-->

# 测试1

最重要的就是能得到目前是训练还是测试状态,我写了个小程序去测试:

```python
class T(k.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_train_begin(self, logs=None):
        print('on_train_begin')

    def on_train_end(self, logs=None):
        print('on_train_end')

    def on_test_begin(self, logs=None):
        print('on_test_begin')

    def on_test_batch_end(self, batch, logs=None):
        print('on_test_batch_end')

    def on_test_end(self, logs=None):
        print('on_test_begin')

    def on_predict_begin(self, logs=None):
        print('predict')


def test_train_callback():
    train_x = np.random.randn(1000, 10).astype(np.float32)
    train_y = np.random.randn(1000, 1).astype(np.float32)

    test_x = np.random.randn(1000, 10).astype(np.float32)
    test_y = np.random.randn(1000, 1).astype(np.float32)
    train_ds = (tf.data.Dataset.from_tensor_slices((train_x, train_y)).
                shuffle(5000).
                repeat().
                batch(400, True).
                map(lambda x, y: ((x), y)))
    test_ds = (tf.data.Dataset.from_tensor_slices((test_x, test_y)).
               shuffle(5000).
               repeat().
               batch(400, True).
               map(lambda x, y: ((x), y)))

    model = k.Sequential([kl.Dense(1)])
    model.compile(k.optimizers.Adam(), 'mse')
    model.fit(train_ds, epochs=3, steps_per_epoch=4, validation_data=test_ds, validation_steps=4, callbacks=[T()], verbose=0)


test_train_callback()
```

获得:
```sh
on_train_begin
on_test_begin
on_test_batch_end
on_test_batch_end
on_test_batch_end
on_test_batch_end
on_test_begin
on_test_begin
on_test_batch_end
on_test_batch_end
on_test_batch_end
on_test_batch_end
on_test_begin
on_test_begin
on_test_batch_end
on_test_batch_end
on_test_batch_end
on_test_batch_end
on_test_begin
on_train_end
```

这里可以分析得整一个周期都是`train`,但在验证过程中只有`on_test_begin`,没有`on_test_end`,只有`on_test_batch_end`.这就很难受了.除非我每个`on_test_batch_end`都调用一下禁止多尺度训练.

不过使用`dataset`至少还是有个好处的,可以使用`validation_steps`统计`on_test_batch_end`次数~


# 测试2

我发现其实在进入测试阶段之后,`on_train_batch_end`是不会被调用的,那么我们其实直接可以直接设定就完事了,不需要再搞什么计数啥的,完美解决问题

```python
class T(k.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.cnt = 0
        self.change_flag = True

    def on_train_batch_end(self, batch, logs=None):
        if self.cnt == 3:
            print('change input scale')
            self.change_flag = True
            self.cnt = 0
        else:
            self.cnt += 1
        print(f'Now change is {self.change_flag}')

    def on_test_batch_end(self, batch, logs=None):
        print(f'Now change is {self.change_flag}')

    def on_test_begin(self, batch, logs=None):
        print('reverse change input scale')
        self.change_flag = False


def test_train_callback():
    train_x = np.random.randn(1000, 10).astype(np.float32)
    train_y = np.random.randn(1000, 1).astype(np.float32)

    test_x = np.random.randn(1000, 10).astype(np.float32)
    test_y = np.random.randn(1000, 1).astype(np.float32)
    train_ds = (tf.data.Dataset.from_tensor_slices((train_x, train_y)).
                shuffle(5000).
                repeat().
                batch(400, True).
                map(lambda x, y: ((x), y)))
    test_ds = (tf.data.Dataset.from_tensor_slices((test_x, test_y)).
               shuffle(5000).
               repeat().
               batch(400, True).
               map(lambda x, y: ((x), y)))

    model = k.Sequential([kl.Dense(1)])
    model.compile(k.optimizers.Adam(), 'mse')
    model.fit(train_ds, epochs=3, steps_per_epoch=10, validation_data=test_ds, validation_steps=3, callbacks=[T()], verbose=0)


test_train_callback()
```

```sh
Now change is True
Now change is True
Now change is True
change input scale
Now change is True
Now change is True
Now change is True
Now change is True
change input scale
Now change is True
Now change is True
Now change is True
reverse change input scale
Now change is False
Now change is False
Now change is False
Now change is False
change input scale
Now change is True
Now change is True
Now change is True
Now change is True
change input scale
Now change is True
Now change is True
Now change is True
Now change is True
change input scale
Now change is True
reverse change input scale
Now change is False
Now change is False
Now change is False
Now change is False
Now change is False
Now change is False
change input scale
Now change is True
Now change is True
Now change is True
Now change is True
change input scale
Now change is True
Now change is True
Now change is True
reverse change input scale
Now change is False
Now change is False
Now change is False
```