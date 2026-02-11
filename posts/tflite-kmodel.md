---
title: 比较kmdoel和tflite推理输出
categories:
  - 边缘计算
date: 2019-07-06 18:16:57
tags:
-   Tensorflow
-   K210
---

我的yolo3模型在k210里面输出结果完全不对，所以我十分怀疑是量化出了问题，但是我又找不到问题。还好昨天case小姐姐帮忙更新了nncase，可以在pc上推理kmdoel.然后我推理了几个图像，这次就是记录一下这个脚本，免得下次要用找不到了。。。

<!--more-->

# 代码

```python
import numpy as np
from scipy.special import expit, softmax
from tensorflow import lite
from pathlib import Path
from skimage.io import imread
from termcolor import colored
np.set_printoptions(suppress=True)


""" 加载tflite """
interpreter = lite.Interpreter(model_path=str('mobile_yolo.tflite'))
interpreter.allocate_tensors()

input_index = [details["index"] for details in interpreter.get_input_details()]
output_index = [details["index"] for details in interpreter.get_output_details()]


""" 加载图片 """
img_paths = list(Path('test_logs').glob('*.jpg'))
img_paths.sort()
imgs = np.array([imread(str(path)) for path in img_paths])
imgs = imgs / 255


""" 推理 """


def infer(img: np.ndarray) -> [np.ndarray, np.ndarray]:
    inp = img[np.newaxis, ...].astype('float32')
    interpreter.set_tensor(input_index[0], inp)
    interpreter.invoke()
    predictions = [interpreter.get_tensor(idx)[0] for idx in output_index]
    return predictions


tf_res = [infer(img) for img in imgs]

""" 加载bin """
bin_paths = list(Path('output').glob('*.bin'))
bin_paths.sort()


def parser_bin(path: Path) -> [np.ndarray, np.ndarray]:
    content = path.open('rb').read()  # type:bytes
    assert len(content) / 4 == 7 * 10 * 75 + 14 * 20 * 75
    # out = np.fromstring(content, dtype='<f4')
    out = np.array(np.frombuffer(content, '<f4'))
    # out = [out[:7 * 10 * 75].reshape(7, 10, 75),
    #        out[7 * 10 * 75:].reshape(14, 20, 75)]
    out = [np.transpose(out[:7 * 10 * 75].reshape(75, 7, 10), (1, 2, 0)),
           np.transpose(out[7 * 10 * 75:].reshape(75, 14, 20), (1, 2, 0))]

    return out


kmd_res = [parser_bin(path) for path in bin_paths]


""" 解析输出 """
inshape = (224, 320)
anchors = np.array([[[81, 82], [135, 169], [344, 319]], [[10, 14], [23, 27], [37, 58]]])

for i in range(len(kmd_res)):
    for j in range(len(output_index)):
        grid_shape = tf_res[i][j].shape[0:2]
        a = tf_res[i][j].reshape(grid_shape + (3, 25))
        b = kmd_res[i][j].reshape(grid_shape + (3, 25))

        """ 解析xy """
        grid_y = np.tile(np.reshape(np.arange(0, grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
        grid_x = np.tile(np.reshape(np.arange(0, grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
        grid = np.concatenate([grid_x, grid_y], axis=-1)

        a[..., 0:2] = (expit(a[..., 0:2]) + grid) / grid_shape[::-1] * inshape[::-1]
        b[..., 0:2] = (expit(b[..., 0:2]) + grid) / grid_shape[::-1] * inshape[::-1]

        """ 解析wh """
        a[..., 2:4] = np.exp(a[..., 2:4]) * anchors[j]
        b[..., 2:4] = np.exp(b[..., 2:4]) * anchors[j]

        """ 解析confendice """
        a[..., 4:5] = expit(a[..., 4:5])
        b[..., 4:5] = expit(b[..., 4:5])

        """ 解析类别 """
        a[..., 5] = np.argmax(softmax(a[..., 5:], axis=-1), axis=-1)
        b[..., 5] = np.argmax(softmax(b[..., 5:], axis=-1), axis=-1)

        print(colored(f'img {i} layer {j} tflite :\n', 'blue'), a[np.where(a[..., 4] > .6)][:, :5])
        print(colored(f'img {i} layer {j} kmodel :\n', 'green'), b[np.where(b[..., 4] > .6)][:, :5])

```

# 结果

```sh
INFO: Initialized TensorFlow Lite runtime.
img 0 layer 0 tflite :
 []
img 0 layer 0 kmodel :
 []
img 0 layer 1 tflite :
 [[ 50.671265  184.66817    16.581987   14.750296    0.7672671]
 [135.05923   188.98267    14.629576    9.189983    0.7180456]
 [180.27417   193.83612    18.99109    10.469457    0.7107621]
 [258.86105   197.36963    41.65087    16.02682     0.8539901]
 [259.66782   197.74883    33.983326   23.298958    0.720852 ]]
img 0 layer 1 kmodel :
 [[ 51.26226    184.          13.131495    14.           0.69366026]]
img 1 layer 0 tflite :
 [[172.43805   147.88477   210.21072   149.22597     0.7295683]]
img 1 layer 0 kmodel :
 [[173.1715     147.51495    211.01398    141.3503       0.72762936]]
img 1 layer 1 tflite :
 []
img 1 layer 1 kmodel :
 [[107.401764   186.7845      63.801384    52.96505      0.73084366]
 [117.872925   187.09857     63.801384    52.96505      0.6740316 ]]
img 2 layer 0 tflite :
 [[181.06418    138.47464    224.90767    197.36978      0.9352028 ]
 [172.32611    137.85072    261.53143    188.90297      0.7145112 ]
 [195.62141    136.05687    213.14252    194.8197       0.85339344]]
img 2 layer 0 kmodel :
 [[180.8456     137.29376    230.73112    184.79173      0.87710017]
 [170.5144     137.29376    263.13138    186.64618      0.6714251 ]
 [195.9328     135.123      211.01398    184.79173      0.87710017]]
img 2 layer 1 tflite :
 []
img 2 layer 1 kmodel :
 []
img 3 layer 0 tflite :
 [[200.14052    100.00058     52.676952   117.090515     0.75216126]]
img 3 layer 0 kmodel :
 [[199.123     100.954346   56.663612  107.2014      0.7615662]]
img 3 layer 1 tflite :
 [[197.36374     98.02013     22.453339    80.50932      0.63156927]
 [244.84764     98.41268     23.07575     50.217842     0.63718444]]
img 3 layer 1 kmodel :
 [[131.26227   100.90143    19.180124   66.949135    0.6537735]
 [197.21548    98.61127    19.594486   83.40284     0.8488212]
 [245.87292    98.81582    23.         50.983635    0.7126105]
 [281.7856     98.23825    21.457172   91.33108     0.7650425]]
```

# 分析

现在看起来量化应该是没什么问题，差距不是很大。我得去找找c代码是不是出错了。。好累。。 🙄