---
title: 使用argparse解析Bool型的坑
categories:
  - 编程语言
date: 2019-02-24 21:21:13
tags:
-   Python
-   踩坑经验
---

今天又碰到一个坑...使用argparse解析`bool`型参数返回值总是`true`.

<!--more-->

# 问题出现

首先我使用以下进行参数解析
```python
parser.add_argument('--train_classifier',           type=bool,  help='wether train the classsifier',    default=False)
```

在使用时指定

```sh
--train_classifier False
```

得到参数都是`True`

# 问题分析

这里是因为`argparse`库对于`bool`型参数是这样控制的:
```sh
--train_classifier
```

使用这个选项即打开.

# 问题解决

```python
parser.add_argument('--train_classifier',           type=str,   help='wether train the classsifier',    choices=['True', 'False'], default='False')
```

现在使用选项的方式来控制`bool`型