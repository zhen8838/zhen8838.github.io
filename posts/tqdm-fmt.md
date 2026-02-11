---
title: tqdm中后缀的添加
categories:
  - 深度学习
date: 2019-01-24 20:06:56
tags:
- Tensorflow
- Python
---

苦于`tensorflow`中没有好用的训练显示函数,所以我准备用`tqdm`库显示一下训练过程.既然要显示训练过程中的参数,那肯定要自己对他的默认进度条格式进行修改,所以这里就来说几个使用方式.

<!--more-->


# 默认进度条格式
首先分析`tqdm`的默认格式:
重点就在于`bar_format`函数

-   **bar_format** : str, optional
    
    Specify a custom bar string formatting. May impact performance. **[default: '{l_bar}{bar}{r_bar}']**, where **l_bar='{desc}: {percentage:3.0f}%|'** and **r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'** Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt, percentage, rate, rate_fmt, rate_noinv, rate_noinv_fmt, rate_inv, rate_inv_fmt, elapsed, remaining, desc, postfix. Note that a trailing ": " is automatically removed after {desc} if the latter is empty.

其中默认进度条有3块组成,分别是**l_bar,bar,r_bar**,下面分别描述一下.

## l_bar

>   100%|███████████████████████████████████████████| 10/10 [00:01<00:00,  9.96it/s]


进度条的前端,上面的**100%**即为`l_bar`,其中`l_bar='{desc}: {percentage:3.0f}%|'`.`desc`为描述前缀,可以由`desc`参数赋值,后面的`percentage`就是我们看到的`100%`.

下面给出一个修改前缀的实例:
```python
from tqdm import trange, tqdm
from random import random, randint
from time import sleep

def training(epoch: int, step_per_epoch: int):
    for i in range(epoch):
        with tqdm(total=step_per_epoch, desc='epoch {}'.format(i), ncols=80) as t:
            for j in range(step_per_epoch):
                sleep(.1)
                t.update()


training(epoch=3, step_per_epoch=10)
```

效果:

```sh
epoch 0: 100%|██████████████████████████████████| 10/10 [00:01<00:00,  9.96it/s]
epoch 1: 100%|██████████████████████████████████| 10/10 [00:01<00:00,  9.96it/s]
epoch 2: 100%|██████████████████████████████████| 10/10 [00:01<00:00,  9.95it/s]
```

## bar
这个就不介绍了,因为进度条的精髓就在这了.并且一般不需要去修改.

## r_bar

现在分析`r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'`

> 100%|████████████████████████████████████████| 10/10 [00:00<00:00, 89813.79it/s]

此处的`{n_fmt}/{total_fmt}`即为上面的**10/10**,这个对于我来说是需要的

此处的`{elapsed}<{remaining}`即为**00:00<00:00**,这个对于我来说是不要的

此处的`{rate_fmt}`就是**9.96it/s**,这个也需要的,并且这个还可以进一步的更改

最后的`{postfix}`上面没有,是需要自己添加参数来显示的,这里的他可以是任意的

其实我主要就是要修改他的`postfix`参数,来随时的显示我的`loss`,`learn rete`啥的.

### 修改后缀实例

#### 例子1

```python
from tqdm import tqdm
from random import random, randint
from time import sleep


def training(epoch: int, step_per_epoch: int):
    for i in range(epoch):
        with tqdm(total=step_per_epoch, ncols=80) as t:
            for j in range(step_per_epoch):
                t.set_postfix(loss='{:^7.3f}'.format(random()))
                sleep(0.1)
                t.update()


training(epoch=3, step_per_epoch=10)
```

#### 效果1

```sh
100%|███████████████████████████████| 10/10 [00:01<00:00,  9.94it/s, loss=0.279]
100%|███████████████████████████████| 10/10 [00:01<00:00,  9.94it/s, loss=0.736]
100%|███████████████████████████████| 10/10 [00:01<00:00,  9.94it/s, loss=0.022]
```

#### 分析1
这里使用了`t.set_postfix`函数,我进入这个函数中,找到了重要部分:
```python
self.postfix = ', '.join(key + '=' + postfix[key].strip()
                                 for key in postfix.keys())
```
这个函数会把输入的参数变成`key=xxx`的形式复制给自身的`self.postfix`,这样还是挺好的,并且可以自己随意添加`list`,`str`,`int`型的变量.


#### 例子2

```python
from tqdm import trange, tqdm
from random import random, randint
from time import sleep

def training(epoch: int, step_per_epoch: int):
    for i in range(epoch):
        with tqdm(total=step_per_epoch, bar_format='{l_bar}{bar}| {rate_fmt} {postfix[0]}{postfix[1][loss]:>6.3f}',
                  unit=' batch', postfix=['loss=', dict(loss=0)], ncols=80) as t:
            for j in range(step_per_epoch):
                t.postfix[1]["loss"] = random()
                t.update()

training(epoch=3, step_per_epoch=10)
```

#### 分析2
上面我首先改变了`bar_format`,不过我没有动前面的两部分,只修改了最后一部分.具体看这里:`{postfix[0]}{postfix[1][loss]:>6.3f}`.这样相当与这个`postfix`是一个列表.其中`t.postfix[1]["loss"] = random()`意思是向这个列表中的字典项`loss`复制,最终达到更新`loss`的效果.

#### 例子3

```python
from tqdm import trange, tqdm
from random import random, randint
from time import sleep


def training(epoch: int, step_per_epoch: int):
    for i in range(epoch):
        with tqdm(total=step_per_epoch, bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt} {postfix}', ncols=80) as t:
            for i in range(step_per_epoch):
                t.set_postfix_str('loss={:^7.3f}'.format(random()))
                sleep(.1)
                t.update()


training(epoch=3, step_per_epoch=10)
```

#### 效果3
```sh
10/10 |███████████████████████████████████████████████|  9.94it/s , loss= 0.093
10/10 |███████████████████████████████████████████████|  9.94it/s , loss= 0.889
10/10 |███████████████████████████████████████████████|  9.94it/s , loss= 0.090
```
#### 分析3

这个最好理解,直接对`postfix`赋值字符串,可以随便自己操作.我比较喜欢.