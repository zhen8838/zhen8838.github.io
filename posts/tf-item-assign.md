---
title: tf2.0数组索引与赋值
mathjax: true
toc: true
categories:
  - 深度学习
date: 2019-12-03 12:51:20
tags:
- Tensorflow
---

用了挺久的`tensorflow`，目前也尝试了一些别的框架，感觉最让我难受的一点就是没法很方便的按索引赋值。

原因主要有：
1.  `tensorflow`的索引方式与`numpy`不同，写起来别扭。
2.  ~~必须要变量类型才可以进行赋值。~~ 我写完这篇文章后找到对普通`tensor`的赋值方式了

这次就来说下一些数组操作在`tensorflow`里面的写法。

<!--more-->

# tensorflow与numpy索引的不同之处

## numpy中索引

`numpy`中是给定每一维下标，然后进行索引。比如一个4维的数组，我们需要索引到他的那个元素，那我们就需要给出一个元组，这个元组里面需要包含不大于维度个数组，每个数组的维度要保证可以适配`broadcast`机制。

```python
arr = np.random.randn(7, 10, 5, 2)
idx = (np.array([2]), np.array([3]), np.array([4]), np.array([1]))
arr[idx] # array([0.41087443])

idx = (np.array([2]), np.array([3]), np.array([4]), np.array([1, 0]))
arr[idx]  # array([ 0.41087443, -0.83541546])

idx = (np.array([2]), np.array([3, 2, 3]), np.array([4]), np.array([1, 0]))
arr[idx]  # error
```

## tf.gather索引

而`tensorflow`中索引，首先就有两个函数：`tf.gather`和`tf.gather_nd`。
先说`gather`，这个是只索引单一维度的，比较好理解，并且会自动将索引出来的值按索引的维度进行`stack`。
```python
arr = tf.random.normal((7, 10, 5, 2))
# shape = (3, 10, 5, 2)
tf.assert_equal(tf.gather(arr, [1, 2, 3]), tf.stack([arr[1], arr[2], arr[3]]))
# shape = (7, 10, 3, 2)
tf.assert_equal(tf.gather(arr, [1, 2, 3], axis=2),
                tf.stack([arr[:, :, 1, :],
                          arr[:, :, 2, :],
                          arr[:, :, 3, :]], axis=2))
```
上面看起来和`stack`差不多，但是`gather`可以通过给索引参数`indices`增加维度来增加最终的输出维度，相当于先添加维度再`stack`。

```python
# shape = (3, 1, 10, 5, 2)
tf.assert_equal(tf.gather(arr, [[1], [2], [3]]),
                tf.stack([arr[1][None, ...],
                          arr[2][None, ...],
                          arr[3][None, ...]]))
# shape = (7, 10, 3, 1, 2)
tf.assert_equal(tf.gather(arr, [[1], [2], [3]], axis=2),
                tf.stack([arr[:, :, 1, :][:, :, None, :],
                          arr[:, :, 2, :][:, :, None, :],
                          arr[:, :, 3, :][:, :, None, :]], axis=2))
```

## tf.gather_nd索引

`gather_nd`里面，是索引参数`indices`的最后一维所对应的值对应了数组的索引值，下面的例子可以发现：
```python
arr = tf.random.normal((7, 10, 5, 2))
# NOTE 单一维度索引
tf.gather(arr, [2]) # shape=(1, 10, 5, 2)
tf.gather_nd(arr, [2]) # shape=(10, 5, 2)
# NOTE 多维度索引
tf.gather(arr, [2, 2]) # shape=(2, 10, 5, 2)
tf.gather_nd(arr, [2, 2]) # shape=(5, 2)
arr[2, 2]  # shape=(5, 2)
```
但是当多个多维度索引时，`gather_nd`就不是很友好了，就相当于每个位置进行一次索引，再组合成新数组。

```python
tf.gather_nd(arr, [[2, 2]])  # shape=(1, 5, 2)
tf.convert_to_tensor([arr[2, 2]])  # shape=(1, 5, 2)
```

## 总结

本来我想说这个索引问题很头大，但是实际上按需求使用其实也还好。

# 按索引赋值

按索引赋值同样有两个方式，对单一维度，以及任意维度赋值。

## scatter_update赋值

与`gather`对应，`scatter_update`也是只索引某一个维度，并且我还没找到他如何指定维度，所以目前我就演示下在第一个维度下的批量赋值：

```python
arr = tf.Variable(tf.random.normal((7, 10, 5, 2)), False)
arr2 = tf.random.normal((10, 5, 2))
arr.scatter_update(tf.IndexedSlices(2, [0]))
arr[0]  # all is 2.
arr.scatter_update(tf.IndexedSlices(arr2, [1]))
tf.assert_equal(arr[1], arr2)  # true
```

## scatter_nd_update赋值

不过还好有这个`scatter_nd_update`，对应`gather_nd`，有两点要注意：
1.  `indices`的`shape[0]`必须等于`updates`的`shape[0]`
2.  `indices`的`shape[-1]`必须小于变量的`dim`
3.  `indices`的`dim`加`updates`的`dim`必须等于变量的`dim`

```python
arr = tf.Variable(tf.random.normal((2, 2, 2)), False)

arr.scatter_nd_update([[1, 1], [0, 1]], [[3, 2], [3, 3]])
arr[1, 1, :]  # [3., 2.]
arr[0, 1, :]  # [3., 3.]

arr.scatter_nd_update([[1, 1, 0], [0, 1, 1]], [6, 7])
arr[1, 1, 0]  # 6.
arr[0, 1, 1]  # 7.

arr = tf.Variable(tf.random.normal((7, 10, 5, 2)), False)
arr.scatter_nd_update([[1, 1, 1, 0], [1, 1, 1, 1]], [2, 3])
arr[1, 1, 1, 0]  # 2.0
arr[1, 1, 1, 1]  # 3.0
```

**注意!**


对于变量的`scatter_nd_update`是不支持使用`-1`进行更新的! 而`tensor_scatter_nd_update`是支持的.

```python
arr = tf.Variable(tf.random.normal((2, 2, 2)))

arr.scatter_nd_update([[1, -1], [0, -1]], [[3, 2], [3, 3]])
arr[1, 1, :]  # [ 0.6016006, -1.7807052]
arr[0, 1, :]  # [-2.4027731, -0.6694494]
arr.scatter_nd_update([[1, 1], [0, 1]], [[3, 2], [3, 3]])
arr[1, 1, :]  # [3., 2.]
arr[0, 1, :]  # [3., 3.]
```

## tensor_scatter_nd_update赋值

如果按位置赋值必须使用变量的话实在是太令人蛋疼了，因为变量是不能多次创建的，那我们必须要用很多全局变量，极大的破坏代码的结构。万幸`tensorflow`提供了`tensor_scatter_nd_update`可以对普通的`tensor`进行操作。其使用方式与`scatter_nd_update`相同，不过他并不能修改原始值。
```python
arr = tf.random.normal((2, 2, 2))

arr = tf.tensor_scatter_nd_update(arr, [[1, 1], [0, 1]], [[3, 2], [3, 3]])
arr[1, 1, :]  # [3., 2.]
arr[0, 1, :]  # [3., 3.]

arr = tf.tensor_scatter_nd_update(arr, [[1, 1, 0], [0, 1, 1]], [6, 7])
arr[1, 1, 0]  # 6.
arr[0, 1, 1]  # 7.

arr = tf.random.normal((7, 10, 5, 2))
arr = tf.tensor_scatter_nd_update(arr, [[1, 1, 1, 0], [1, 1, 1, 1]], [2, 3])
arr[1, 1, 1, 0]  # 2.0
arr[1, 1, 1, 1]  # 3.0
```


## 总结

这样总结一次相信对`tensorflow`中的索引与赋值更加熟悉，需要根据不同的场景选择合适的函数。当然如果需要更深入的了解，还是需要查看`api`文档。最后，根据我的测试，将原来的数据生成过程的标签制作、图像读取、图像归一化等过程全部利用`tensorflow`的函数重写之后，数据读取加快了**5倍**有余，已经好了很多，就是一些图像旋转等等的数据增强不好用`tensorflow`重写，不然速度还可以更快。`tensorflow`的使用过程虽然比较别扭，但是构建静态图之后有`xla`、图优化等方法使程序运行更高效。