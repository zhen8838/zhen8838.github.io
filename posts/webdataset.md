---
title: Pytorch Webdataset初体验
mathjax: true
toc: true
categories:
  - 深度学习
date: 2020-11-12 10:16:48
tags:
- Pytorch
---

最近都在用pytorch，虽然pytorch很多东西都比tensorflow舒服，但是在`data pipeline`方面还是tensorflow比较有优势，缺乏一个紧凑压缩的record的读取方法，虽然可以用DALI，但是之前用了一下还是不够灵活。最近在pytorch博客中发现了一个`Webdataset`，因此就尝试一下。

<!--more-->

# 介绍

他的方法是将所有的样本压缩到tar文件中，使用名字作为样本的`key`，比如样本`A`可以包含`A.jpg,A.json`等等，读取的时候根据`key`一次性将所有的样本元素全部读取到`dict`中，之后我们可以随意的`map`，灵活性还是比较大的。

# 结论

经过测试之后，速度较以前的确有所提升，并且读取的速度比较稳定。不过也有几个不太方便的地方：

1.  无法得知数据集的长度
    因为是`tar`文件，构建数据集时无法得知整体长度，所以需要显式的指定。

2.  不像`tfrecord`，无法对一个`tar`文件进行多线程读取。
    `pytorch`中的`dataloader`可以指定多个`worker`进行读取，但是如果`tar`文件没有进行分片的话，就不会起作用，必须要将`tar`文件先进行分片才行。不过就算不分片，速度也比原来的多线程读取要快、要稳定。

3.  无法进行`concat`等等操作
    这个没有办法，毕竟`tensorflow`的`dataset`也没有这个功能。

# 例子
## 制作分片的数据集

```python
from pathlib import Path
import shutil
import os
import webdataset as wds

if __name__ == "__main__":
  train = Path('/media/zqh/Documents/facelandmark_dataset/train')
  test = Path('/media/zqh/Documents/facelandmark_dataset/test')
  if not train.exists():
    train.mkdir()
  if not test.exists():
    test.mkdir()
  org1 = Path('/media/zqh/Documents/JOJO_face_crop_big')
  org2 = Path('/home/zqh/workspace/data512x512')

  test_ids = []
  train_ids = []

  for org in [org1, org2]:
    ids = list(set([p.stem for p in org.iterdir()]))
    n = len(ids)
    test_n = int(n * 0.1)
    for id in ids[:test_n]:
      test_ids.append(org / id)

    for id in ids[test_n:]:
      train_ids.append(org / id)

  for dst_root, ids in [(test, test_ids), (train, train_ids)]:
    total = len(ids)
    pattern = dst_root.as_posix() + f'-{str(total)}-%d.tar'
    with wds.ShardWriter(pattern, maxcount=5000, encoder=False) as f:
      for id in ids:
        with open(id.as_posix() + '.jpg', "rb") as stream:
          image = stream.read()
        with open(id.as_posix() + '.json', "rb") as stream:
          json = stream.read()
        key = id.name
        f.write({'__key__': key, 'jpg': image, 'json': json})
```


## 读取分片的数据集


```python
def get_pattern_and_total_num(root, stage='train'):
  root = Path(root)
  splits = []
  for s in list(root.glob(f'{stage}*')):
    name, total, split = s.stem.split('-')
    splits.append(split)
  if len(splits) > 1:
    patten_str = '{' + '..'.join([splits[0], splits[-1]]) + '}'
  else:
    patten_str = splits[0]
  patten = (root / ('-'.join([name, total, patten_str]) + '.tar')).as_posix()
  return patten, int(total)


def dev_load_shared_dataset():
  root = '/media/zqh/Documents/facelandmark_dataset'
  url, total = get_pattern_and_total_num(root, 'train')

  fn = lambda x, **kwarg: x
  idenity = A.Lambda(image=fn, mask=fn,
                     keypoint=fn, bbox=fn)

  train_transform = A.Compose([
      A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2),
      A.Resize(256, 256),
  ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

  def perar_fn(sample):
    keypoints = FaceLandMarkDataModule.parser_landmark(sample['json'])
    return train_transform(image=sample['jpg'], keypoints=keypoints)

  ds: wds.Dataset = wds.Dataset(url, length=total).shuffle(5000).decode(
      'rgb8').map(perar_fn).to_tuple('image', 'keypoints').batched(8)
  # Read ！
  for sampe in ds:
    pass

```
