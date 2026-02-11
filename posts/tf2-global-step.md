---
title: Tensorflow 2.0中使用global steps
categories:
  - 深度学习
date: 2019-05-06 12:54:33
tags:
-   Tensorflow
---

用了一段时间的`tensorflow 2.0`,总的来说默认`eager`模式操作数据十分的方便,并且可以适当的转为`tf.function`加快速度.但是和`keras`的结合还是不够灵活,比如可以单独用`fit`可以执行,但是想用更加灵活的方式训练有时候就会出现莫名其妙的问题,让人抓狂.


<!--more-->

今天我想用以前的方式使用`global step`,在教程里面找了只能设置`step=optimizer.iterations`,这也太蠢了8,如果我要在训练过程中进行测试,`step`也必须要增加的.然后我摸索到了如下使用方式:

```python
writer = summary.create_file_writer(os.path.join('log', datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')))
steps = tf.train.create_global_step()
with writer.as_default():
    testiter = iter(dataset_test)
    for i in range(3):
        for x, y in dataset:
            loss, acc = train_one_step(model, optimizer, x, y)
            summary.scalar('train_loss', loss, step=steps)
            summary.scalar('train_acc', acc, step=steps)
            if steps.numpy() % 20 == 0:
                test_x, test_y = next(testiter)
                loss, acc = test_one_step(model, test_x, test_y)
                summary.scalar('test_loss', loss, step=steps)
                summary.scalar('test_acc', acc, step=steps)
            steps.assign_add(1)

            print('\rsteps:{}\t\tloss:{:.4f}\t\tacc:{:.4f}%'.format(steps.numpy(), loss, acc * 100), end='')
```
