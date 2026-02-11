---
title: mindspore vs tensorflow
mathjax: true
toc: true
categories:
  - 深度学习
date: 2020-09-21 18:51:05
tags:
- Tensorflow
- mindspore
---

尝试用了一下`mindspore`，这里给出一个`dcgan`的`demo`对比一下两个框架。
我使用`mindspore 0.7`，`tensorflow 2.2`，`megengine 0.6`，其他参数均相同。


<!--more-->

#### `mindspore`版


```python
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.train.parallel_utils import ParallelMode
from mindspore.parallel._utils import (_get_device_num, _get_mirror_mean, _get_parallel_mode)

import mindspore as ms
import mindspore.context as context
import mindspore.nn.wrap as mwp
import mindspore.nn.layer as ml
import mindspore.train.callback as callback
import mindspore.nn.loss as mls
import mindspore.nn.optim as moptim
from mindspore.nn import Cell
import mindspore.ops.functional as F
import mindspore.ops.operations as P
import mindspore.ops.composite as C
from mindspore.common import initializer as minit
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
from urllib.parse import urlparse
import gzip
import time


def unzipfile(gzip_path):
  """unzip dataset file
  Args:
      gzip_path: dataset file path
  """
  open_file = open(gzip_path.replace('.gz', ''), 'wb')
  gz_file = gzip.GzipFile(gzip_path)
  open_file.write(gz_file.read())
  gz_file.close()


def download_dataset():
  """Download the dataset from http://yann.lecun.com/exdb/mnist/."""
  train_path = "./MNIST_Data/train/"
  test_path = "./MNIST_Data/test/"
  train_path_check = os.path.exists(train_path)
  test_path_check = os.path.exists(test_path)
  if train_path_check == False and test_path_check == False:
    os.makedirs(train_path)
    os.makedirs(test_path)
  else:
    return
  print("******Downloading the MNIST dataset******")
  train_url = {"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
               "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"}
  test_url = {"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
              "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"}
  for url in train_url:
    url_parse = urlparse(url)
    # split the file name from url
    file_name = os.path.join(train_path, url_parse.path.split('/')[-1])
    if not os.path.exists(file_name.replace('.gz', '')):
      file = urllib.request.urlretrieve(url, file_name)
      unzipfile(file_name)
      os.remove(file_name)
  for url in test_url:
    url_parse = urlparse(url)
    # split the file name from url
    file_name = os.path.join(test_path, url_parse.path.split('/')[-1])
    if not os.path.exists(file_name.replace('.gz', '')):
      file = urllib.request.urlretrieve(url, file_name)
      unzipfile(file_name)
      os.remove(file_name)


def create_dataset(data_path, noise_dim, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
  """ create dataset for train or test
  Args:
      data_path: Data path
      batch_size: The number of data records in each group
      repeat_size: The number of replicated data records
      num_parallel_workers: The number of parallel workers
  """
  # define dataset
  mnist_ds = ds.MnistDataset(data_path)

  hwc2chw_op = transforms.vision.c_transforms.HWC2CHW()
  # apply map operations on images
  mnist_ds = (mnist_ds.map(operations=lambda x: ((x - 127.5) / 127.5).astype('float32'), input_columns="image",
                           num_parallel_workers=num_parallel_workers)
              .map(operations=hwc2chw_op, input_columns="image",
                   num_parallel_workers=num_parallel_workers)
              .map(operations=lambda x: (x, np.random.randn(noise_dim).astype('float32')),
                   input_columns="image",
                   output_columns=["image", "noise"],
                   columns_order=["image", "noise"],
                   num_parallel_workers=num_parallel_workers))
  # apply DatasetOps
  buffer_size = 60000
  mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
  mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
  mnist_ds: ds.MindDataset = mnist_ds.repeat(repeat_size)
  # print(mnist_ds.output_())
  # print(mnist_ds.output_types())
  return mnist_ds


class Reshape(Cell):
  def __init__(self, shape: list) -> None:
    super().__init__()
    self.shape = shape

  def construct(self, x):
    return F.reshape(x, self.shape)


def make_generator_model(noise_dim):
  model = ml.SequentialCell(
      ml.Dense(noise_dim, 7 * 7 * 256, has_bias=False),
      Reshape((-1, 256, 7, 7)),
      ml.BatchNorm2d(256),
      ml.LeakyReLU(),
      # assert model.output_shape == (None, 7, 7, 256)  # 注意：batch size 没有限制
      ml.Conv2dTranspose(256, 128, (5, 5), stride=(1, 1), pad_mode='same', has_bias=False),
      # assert model.output_shape == (None, 7, 7, 128)
      ml.BatchNorm2d(128),
      ml.LeakyReLU(),
      ml.Conv2dTranspose(128, 64, (5, 5), stride=(2, 2), pad_mode='same', has_bias=False),
      # assert model.output_shape == (None, 14, 14, 64)
      ml.BatchNorm2d(64),
      ml.LeakyReLU(),
      ml.Conv2dTranspose(64, 1, (5, 5), stride=(2, 2),
                         pad_mode='same', has_bias=False),
      ml.Tanh()
      # assert model.output_shape == (None, 28, 28, 1)
  )
  return model


def make_discriminator_model():
  model = ml.SequentialCell(
      ml.Conv2d(1, 64, (5, 5), stride=(2, 2), pad_mode='same'),
      ml.LeakyReLU(),
      ml.Dropout(0.3),
      ml.Conv2d(64, 128, (5, 5), stride=(2, 2), pad_mode='same'),
      ml.LeakyReLU(),
      ml.Dropout(0.3),
      ml.Flatten(),
      ml.Dense(128 * 7 * 7, 1),
      ml.Sigmoid()
  )

  return model


class GANBaseNet(Cell):
  def __init__(self, noise_dim) -> None:
    super().__init__(auto_prefix=True)
    self.generator = make_generator_model(noise_dim)
    self.discriminator = make_discriminator_model()

  def construct(self, images, noise):
    generated_images = self.generator(noise)
    real_output = self.discriminator(images)
    fake_output = self.discriminator(generated_images)
    return real_output, fake_output


class GANWithLoss(Cell):
  def __init__(self, base_net: GANBaseNet) -> None:
    super().__init__(auto_prefix=True)
    self.base_net = base_net
    self.cross_entropy = P.BinaryCrossEntropy()  # 是否需要sigmoid是个问题

  def discriminator_loss(self, real_output, fake_output, weight):
    real_loss = self.cross_entropy(real_output, F.ones_like(real_output), weight)
    fake_loss = self.cross_entropy(fake_output, F.zeros_like(fake_output), weight)
    total_loss = real_loss + fake_loss
    return total_loss

  def generator_loss(self, fake_output, weight):
    return self.cross_entropy(fake_output, F.ones_like(fake_output), weight)

  def construct(self, images, noise):
    real_output, fake_output = self.base_net(images, noise)
    weight = F.ones_like(real_output)
    gen_loss = self.generator_loss(fake_output, weight)
    disc_loss = self.discriminator_loss(real_output, fake_output, weight)
    return gen_loss, disc_loss


class IthOutputCell(Cell):
  """ 显式指定反向传播图 """

  def __init__(self, network, output_index):
    super(IthOutputCell, self).__init__()
    self.network = network
    self.output_index = output_index

  def construct(self, image, noise):
    predict = self.network(image, noise)[self.output_index]
    return predict


class TrainStepWrap(Cell):
  def __init__(self, network: GANWithLoss, g_optimizer: moptim.Optimizer, d_optimizer: moptim.Optimizer, sens=1.0):
    # NOTE 这里必须要设置auto_prefix，否则两个优化器的参数将冲突
    super(TrainStepWrap, self).__init__(auto_prefix=True)
    self.network = network
    self.network.set_grad()
    self.network.add_flags(defer_inline=True)
    self.g_weights = g_optimizer.parameters
    self.g_optimizer = g_optimizer
    self.d_weights = d_optimizer.parameters
    self.d_optimizer = d_optimizer
    self.g_grad = C.GradOperation('g_grad', get_by_list=True, sens_param=True)
    self.d_grad = C.GradOperation('d_grad', get_by_list=True, sens_param=True)

    self.g_loss_net = IthOutputCell(network, output_index=0)
    self.d_loss_net = IthOutputCell(network, output_index=1)

    self.sens = sens
    self.reducer_flag = False
    self.grad_reducer = None
    parallel_mode = _get_parallel_mode()
    if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
      self.reducer_flag = True
    if self.reducer_flag:
      mean = _get_mirror_mean()
      degree = _get_device_num()
      self.g_grad_reducer = DistributedGradReducer(g_optimizer.parameters, mean, degree)
      self.d_grad_reducer = DistributedGradReducer(d_optimizer.parameters, mean, degree)

  def update_model(self, image, noise, loss, loss_net, grad, optimizer, weights, grad_reducer):
    sens = F.fill(F.dtype(loss), F.shape(loss), self.sens)
    grads = grad(loss_net, weights)(image, noise, sens)
    if self.reducer_flag:
      # apply grad reducer on grads
      grads = grad_reducer(grads)
    return F.depend(loss, optimizer(grads))

  def construct(self, image, noise):
    g_loss, d_loss = self.network(image, noise)
    g_out = self.update_model(image, noise, g_loss, self.g_loss_net, self.g_grad,
                              self.g_optimizer, self.g_weights, self.g_grad_reducer)
    d_out = self.update_model(image, noise, d_loss, self.d_loss_net, self.d_grad,
                              self.d_optimizer, self.d_weights, self.d_grad_reducer)
    return g_out, d_out


class GANLossMonitor(callback.LossMonitor):
  def step_end(self, run_context):
    cb_params = run_context.original_args()
    g_loss, d_loss = cb_params.net_outputs
    g_loss: ms.Tensor

    g_loss = np.mean(g_loss.asnumpy())
    d_loss = np.mean(d_loss.asnumpy())

    cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

    if isinstance(g_loss, float) and (np.isnan(g_loss) or np.isinf(g_loss)):
      raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
          cb_params.cur_epoch_num, cur_step_in_epoch))
    if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
      print("epoch: %s step: %s, g_loss %s d_loss %s" %
            (cb_params.cur_epoch_num, cur_step_in_epoch, g_loss, d_loss), flush=True)


class GANImageSave(callback.Callback):
  def __init__(self, generator: Cell, noise_dim) -> None:
    super().__init__()
    self.generator = generator
    self.seed = ms.Tensor(np.random.randn(16, noise_dim), ms.float32)
    if not os.path.exists('./log'):
      os.mkdir('./log')

  def epoch_end(self, run_context):
    cb_params = run_context.original_args()
    # self.generator.set_train(False) NOTE 暂时不知道是否需要设置
    predictions: ms.Tensor = self.generator(self.seed)
    predictions = predictions.asnumpy()

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i + 1)
      plt.imshow(predictions[i, 0, :, :] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

    plt.savefig('./log/image_at_epoch_{:04d}.png'.format(cb_params.cur_epoch_num))


class Timer(callback.Callback):

  def epoch_begin(self, run_context):
    self.start = time.time()

  def epoch_end(self, run_context):
    cb_params = run_context.original_args()
    print('Time for epoch {} is {} sec'.format(cb_params.cur_epoch_num, time.time() - self.start))


if __name__ == "__main__":
  EPOCHS = 50
  NOISE_DIM = 100
  BATCH_SIZE = 256
  num_examples_to_generate = 16
  context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
  sink_mode = True

  """ set dataset ~ """
  download_dataset()
  mnist_path = "./MNIST_Data"
  ds_train = create_dataset(os.path.join(mnist_path, "train"), NOISE_DIM,
                            BATCH_SIZE, 1)

  """ define model ~ """
  net = GANBaseNet(NOISE_DIM)
  net_loss = GANWithLoss(net)
  generator_optimizer = moptim.Adam(net.generator.trainable_params(), 1e-4)
  discriminator_optimizer = moptim.Adam(net.discriminator.trainable_params(), 1e-4)
  net_train_step = TrainStepWrap(net_loss, generator_optimizer, discriminator_optimizer)
  model = ms.train.Model(net_train_step, amp_level='O2')

  """ trianing ~ """
  model.train(EPOCHS, ds_train,
              callbacks=[Timer(), GANImageSave(net.generator, NOISE_DIM)],
              dataset_sink_mode=sink_mode)
  """ make gif ~ """
  anim_file = 'dcgan.gif'
  import imageio
  import glob
  with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('./log/image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
      frame = 2 * (i**0.5)
      if round(frame) > round(last):
        last = frame
      else:
        continue
      image = imageio.imread(filename)
      writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
""" 14.3 sec/epoch , GPU mem 1544Mb """
```


输出：
```sh
Time for epoch 1 is 14.746528148651123 sec
Time for epoch 2 is 13.69857907295227 sec
Time for epoch 3 is 13.860252380371094 sec
Time for epoch 4 is 13.879372358322144 sec
Time for epoch 5 is 13.845653057098389 sec
Time for epoch 6 is 13.994170665740967 sec
Time for epoch 7 is 13.880078554153442 sec
```

#### `tensorflow`版


```python
import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # 将图片标准化到 [-1, 1] 区间内
BUFFER_SIZE = 60000
BATCH_SIZE = 256
# 批量化和打乱数据
train_dataset = tf.data.Dataset.from_tensor_slices(
    train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def make_generator_model():
  model = tf.keras.Sequential()
  model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Reshape((7, 7, 256)))
  assert model.output_shape == (None, 7, 7, 256)  # 注意：batch size 没有限制

  model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
  assert model.output_shape == (None, 7, 7, 128)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 14, 14, 64)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2),
                                   padding='same', use_bias=False, activation='tanh'))
  assert model.output_shape == (None, 28, 28, 1)

  return model


def make_discriminator_model():
  model = tf.keras.Sequential()
  model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                          input_shape=[28, 28, 1]))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Flatten())
  model.add(layers.Dense(1))

  return model


generator = make_generator_model()
discriminator = make_discriminator_model()
# 该方法返回计算交叉熵损失的辅助函数
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss


def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16


# 我们将重复使用该种子（因此在动画 GIF 中更容易可视化进度）
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# 注意 `tf.function` 的使用
# 该注解使函数被“编译”


@tf.function
def train_step(images):
  noise = tf.random.normal([BATCH_SIZE, noise_dim])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(
      zip(gradients_of_discriminator, discriminator.trainable_variables))


def generate_and_save_images(model, epoch, test_input):
  # 注意 training` 设定为 False
  # 因此，所有层都在推理模式下运行（batchnorm）。
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')

  plt.savefig('/tmp/image_at_epoch_{:04d}.png'.format(epoch))


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

  generate_and_save_images(generator,
                           epochs,
                           seed)


train(train_dataset, EPOCHS)
anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('/tmp/image_at_epoch_*.png')
  filenames = sorted(filenames)
  last = -1
  for i, filename in enumerate(filenames):
    frame = 2 * (i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)
""" 7.74 sec/epoch, total 183.87s, GPU mem 2655Mb """
```


输出：
```sh
Time for epoch 1 is 10.126373767852783 sec
Time for epoch 2 is 7.418195009231567 sec
Time for epoch 3 is 7.2069251537323 sec
Time for epoch 4 is 7.063368797302246 sec
Time for epoch 5 is 7.035956144332886 sec
```

#### `megengine`版


```python
import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np
from megengine.data import DataLoader
from megengine.data.dataset import MNIST
from megengine.data import SequentialSampler
from megengine.data.transform import Normalize, ToMode, Compose, VisionTransform
from megengine.jit import trace
from megengine import optimizer as optim
from megengine.core.tensor_factory import ones
import time


def ones_like(inp):
  return ones(inp.shapeof()).astype(inp.dtype)


batch_size = 256


class Noise(VisionTransform):
  def __init__(self, noise_dim=100, *, order=None) -> None:
    super().__init__(order)
    self.noise_dim = noise_dim

  def _apply_image(self, image):
    return image, np.random.randn(self.noise_dim).astype('float32')


mnist_train_dataset = MNIST(root="./MNIST", train=True, download=False)
sequential_sampler = SequentialSampler(dataset=mnist_train_dataset, batch_size=batch_size)
mnist_train_dataloader = DataLoader(dataset=mnist_train_dataset,
                                    sampler=sequential_sampler,
                                    transform=Compose([
                                        Normalize(127.5, 127.5),
                                        ToMode('CHW'),
                                        Noise()]))


class Reshape(M.Module):
  def __init__(self, shape: tuple) -> None:
    super().__init__()
    self.shape = shape

  def forward(self, inputs):
    return F.reshape(inputs, self.shape)


generator = M.Sequential(
    M.Linear(100, 7 * 7 * 256, bias=False),
    Reshape((-1, 256, 7, 7)),
    M.BatchNorm2d(256),
    M.LeakyReLU(),
    M.ConvTranspose2d(256, 128, 5, stride=1, padding=2, bias=False),
    M.BatchNorm2d(128),
    M.LeakyReLU(),
    M.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
    M.BatchNorm2d(64),
    M.LeakyReLU(),
    M.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=False),
    M.Elemwise('TANH')
)


discriminator = M.Sequential(
    M.Conv2d(1, 64, 5, stride=2, padding=1),
    M.LeakyReLU(),
    M.Dropout(0.3),
    M.Conv2d(64, 128, 5, stride=2, padding=1),
    M.LeakyReLU(),
    M.Dropout(0.3),
    Reshape((-1, 128 * 6 * 6)),
    M.Linear(128 * 6 * 6, 1),
    M.Sigmoid()
)

# iters = iter(mnist_train_dataloader)
# (img, noise), _ = next(iters)
# discriminator(img).shape
# # 4608
# 128*6*6


def discriminator_loss(real_output, fake_output):
  real_loss = F.binary_cross_entropy(real_output, ones_like(real_output))
  fake_loss = F.binary_cross_entropy(fake_output, F.zeros_like(fake_output))
  total_loss = real_loss + fake_loss
  return total_loss


def generator_loss(fake_output):
  return F.binary_cross_entropy(fake_output, ones_like(fake_output))


@trace(symbolic=True)
def train_step(images, noise, *, g_opt, d_opt, g_net, d_net):
  generator_optimizer.zero_grad()
  discriminator_optimizer.zero_grad()
  generated_images = g_net(noise)

  real_output = d_net(images)
  fake_output = d_net(generated_images)

  gen_loss = generator_loss(fake_output)
  disc_loss = discriminator_loss(real_output, fake_output)
  g_opt.backward(gen_loss)
  d_opt.backward(disc_loss)
  return gen_loss, disc_loss


generator_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)


trace.enabled = True  # 开启trace，使用静态图模式

EPOCHS = 50


generator.train()
discriminator.train()
for epoch in range(EPOCHS):
  start = time.time()
  for (imgs, noise), _ in mnist_train_dataloader:

    gen_loss, disc_loss = train_step(imgs, noise, g_opt=generator_optimizer,
                                     d_opt=discriminator_optimizer,
                                     g_net=generator, d_net=discriminator)
  # generate_and_save_images(generator,
  #                          epoch + 1,
  #                          seed)

  print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

""" 8.47 sec/epoch, GPU mem 1299Mb """
```


输出:
```sh
21 21:17:30 process the raw files of train set...
100%|██████████████████████████████████| 60000/60000 [00:02<00:00, 21099.52it/s]
100%|████████████████████████████████| 60000/60000 [00:00<00:00, 1832574.11it/s]
Time for epoch 1 is 8.129057168960571 sec
Time for epoch 2 is 7.941509008407593 sec
Time for epoch 3 is 7.948836326599121 sec
Time for epoch 4 is 7.97221827507019 sec
Time for epoch 5 is 7.969634771347046 sec
Time for epoch 6 is 7.968409776687622 sec
Time for epoch 7 is 7.994731187820435 sec
Time for epoch 8 is 8.000129699707031 sec
Time for epoch 9 is 8.010562181472778 sec
Time for epoch 10 is 8.037590265274048 sec
```

#### 总结

1.  `tensorflow`占显存为`2655Mb`，转静态图之后的速度真的还是非常快的。虽然我吐槽`tf`，但强还是强的啊。
2.  `mindspore`虽然占显存为`1544Mb`，但是明明是静态图运行，速度居然比`tf`慢一倍，这个真的让人难以接受，并且我还是开启了他的自动半精度，并没有什么用处。此外我还尝试了扩大数据载入的线程，然而4线程的时候一个周期反而需要`55s`，这是让我没想到的。
3.  `megengine`是我之前最不看好的，因为我感觉就像全抄`pytorch`的一样，肯定不会快很多。没想到结果还挺香，显存只需要`1299Mb`，速度还挺快.
