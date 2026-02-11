---
title: Tensorflow加载pb文件继续训练
categories:
  - 深度学习
date: 2019-01-28 09:48:37
tags:
- Tensorflow
---

>   Tensorflow中模型即代码

上面这句话说的很对,当我们只有一个预训练好的`pb`文件,我们如何加载这个模型继续训练呢?今天就来解决这个问题.

<!--more-->

## 1.    加载pb文件

我们得到了一个`.pb`文件,不论是提取他的参数还是用他进行推理,都得加载这个文件.请看下面几行代码:
```python
def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.GFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
```

#### 讲解

这里是分两种情况,我们就看第一种:直接使用`tf.gfile.GFile`打开此文件,然后获得当前文件的图定义,读入数据,最后将此图导入.

## 2.   获得操作

一般我们加载了图之后,都是去获得他的占位符去进行输入,然后输出.为了得到所有的权重,我们使用`g.get_operations()`获得所有的操作节点.

```python
import tensorflow as tf
from mobilenetv1.base_func import *


if __name__ == "__main__":
    g = tf.get_default_graph()
    sess = tf.Session()
    load_model('pretrained/mobilenetv1_1.0.pb')
    g.get_operations()
```


**注意**:

导入`pb`文件之后使用`tf.global_variables()`等获取变量的方式都是无效的,获得的都是空值.如下所示:
```sh
In [6]:     tf.global_variables()
Out[6]: []

In [7]:     tf.trainable_variables()
Out[7]: []
```



## 3.   获得tensor

当我们有了操作列表之后如何进行读取变量呢?让我们先看看操作列表中的数据:
```sh
In [11]: optlist
Out[11]:
[<tf.Operation 'inputs' type=Placeholder>,
 <tf.Operation 'MobileNetV1/SpaceToBatchND/block_shape' type=Const>,
 <tf.Operation 'MobileNetV1/SpaceToBatchND/paddings' type=Const>,
 <tf.Operation 'MobileNetV1/SpaceToBatchND' type=SpaceToBatchND>,
 <tf.Operation 'MobileNetV1/Conv2d_0_3x3/weights' type=Const>,
 <tf.Operation 'MobileNetV1/Conv2d_0_3x3/weights/read' type=Identity>,
 <tf.Operation 'MobileNetV1/Conv2d_0_3x3/Conv2D' type=Conv2D>,
 <tf.Operation 'MobileNetV1/Conv2d_0_3x3/BatchNorm/beta' type=Const>,
 <tf.Operation 'MobileNetV1/Conv2d_0_3x3/BatchNorm/beta/read' type=Identity>
```

我们可以看到这里的这些不是变量,并且种类烦杂,不过我直接说明`xxxx/read`等操作就是读取预存的权重的操作.因此我们可以直接把这些操作过滤出来.

```python
def get_vars_from_optlist(optlist: list)->list:
    """ 从optlist获得所有的变量节点 """
    varlist = [node for node in optlist if '/read' in node.name]
    return varlist
```
现在我们有个对应的读取变量操作列表,但是要读取变量还是要进行转化,因为`varlist`只是一个操作,还没有变成可运行的`tensor`,所以我只要在操作名后面加上`:0`,同时`get_tensor_by_name()`即可得到对应的`tensor`

```Python
def convert_vars_to_tensor(g, varlist: list)->list:
    """ 把varlist中的操作转变为可运行的tensor """
    tensorlist = []
    for var in varlist:
        tensorlist.append(g.get_tensor_by_name(var.name+':0'))
    return tensorlist
```

## 4.   读取变量

有了`tensorlist`,我们可以来读取变量了.为了`restore`的方便,我将他保存成字典的形式,并且修改每一个`key`都与原图中的变量名相同,这样`restore`的时候直接判断名字是否相同即可.

```python
# 将所有变量存入字典
vardict = {}
for v in tensorlist:
    vardict[v.name.replace('/read', '')] = sess.run(v)
```

## 5.   保存字典

使用下面的这个函数保存我们的`vardict`
```python
def save_pkl(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
```


## 6.   恢复权重


**注意**:
要恢复权重,你必须得有原图的定义,否则你必须重新写一个.

1.  首先定义一个原图,接下来需要回复权重.
2.  搜集此图中所有可训练的变量(我这里用except_last控制是否加载最后一层权重)
3.  加载之前保存的字典文件
4.  使用`tf.assign()`,将`modelvarlist`与`pre_weight_dict`中名字相同的变量进行赋值,存入`optlist`
5.  使用`sess.run(optlist)`,进行赋值操作
6.  大功告成~

```python
def restore_form_pkl(sess: tf.Session(), pklpath: str, except_last=True):
    """ restore the pre-train weight form the .pkl file

    Parameters
    ----------
    sess : tf.Session
        sess
    pklpath : str
        .pkl file path
    except_last : bool, optional
        whether load the last layer weight, when you custom the net shouldn't load 
        the layer name scope is 'MobileNetV1/Bottleneck2'
        (the default is True, which not load the last layer weight)
    """
    # tf.global_variables() == tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # filter the last layer weight
    modelvarlist = [var for var in tf.trainable_variables(scope='MobileNetV1') if not (except_last and 'MobileNetV1/Bottleneck2' in var.name)]
    pre_weight_dict = load_pkl(pklpath)

    # make sure the number equal
    var_num = len(modelvarlist)

    # save the opt to list
    opt_list = []
    for newv in modelvarlist:
        for k, oldv in pre_weight_dict.items():
            if k == newv.name:
                opt_list.append(tf.assign(newv, oldv))

    # make sure the number equal
    assert len(opt_list) == var_num
    # run the assign
    sess.run(opt_list)
```