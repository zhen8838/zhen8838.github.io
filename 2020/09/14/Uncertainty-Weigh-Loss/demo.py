import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
from typing import List
import numpy as np
from datetime import datetime
import sys


def classifier(num_class: int):
  return k.Sequential([kl.Flatten(),
                       kl.Dense(32),
                       kl.ReLU(),
                       kl.Dense(num_class),
                       kl.Softmax()])


def build_model():
  inputs = k.Input((28, 28, 1))
  encoder = k.Sequential([kl.Conv2D(16, 3, 1, 'same'),
                          kl.ReLU(),
                          kl.Conv2D(16, 3, 1, 'same'),
                          kl.ReLU(),
                          kl.MaxPool2D(2, 2),
                          kl.Conv2D(16, 2, 1, 'same'),
                          kl.ReLU(),
                          kl.Conv2D(8, 2, 1, 'same'),
                          kl.ReLU(),
                          kl.MaxPool2D(2, 2),
                          kl.Conv2D(8, 2, 2, 'same'),
                          kl.ReLU(),
                          kl.MaxPool2D(2, 2)])
  classifier1 = classifier(3)
  classifier2 = classifier(10)
  reconstructor = k.Sequential([kl.Conv2DTranspose(8, 3, 3, 'valid'),
                                kl.ReLU(),
                                kl.Conv2DTranspose(4, 3, 2, 'same'),
                                kl.ReLU(),
                                kl.Conv2DTranspose(2, 3, 1, 'valid'),
                                kl.ReLU(),
                                kl.Conv2DTranspose(1, 2, 2, 'valid'),
                                kl.Activation('tanh'), ])
  x = encoder(inputs)
  x1 = classifier1(x)
  x2 = classifier2(x)
  x3 = reconstructor(x)
  model = k.Model(inputs, [x1, x2, x3])
  return model


def uncertaint_weight(loss, weight):
  return 0.5 * tf.exp(-weight) * loss + 0.5 * weight
  # return 1 / (2 * weight * weight) * loss + tf.math.log(weight)


if __name__ == "__main__":
  USE_LEARNABLE = True if sys.argv[1] == 'True' else False
  USE_RECONSTRUCT = True if sys.argv[2] == 'True' else False

  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

  model: k.Model = build_model()
  sub_dir = datetime.strftime(datetime.now(), r'%Y%m%d-%H%M%S')
  root_dir = 'tmp/'
  s1 = 'use_learnable' if USE_LEARNABLE else 'use_fixed'
  s2 = 'use_recon' if USE_RECONSTRUCT else 'no_recon'
  sub_dir = s1 + '_' + s2 + '/' + sub_dir

  writer = tf.summary.create_file_writer(root_dir + sub_dir)
  initer = k.initializers.RandomUniform(0.2, 1.0)
  weights: List[tf.Variable] = [tf.Variable(
      initer([], tf.float32), trainable=USE_LEARNABLE) for i in range(3)]
  optimizer = k.optimizers.Adam(0.0001)
  ce_fn1 = kls.CategoricalCrossentropy()
  ce_fn2 = kls.CategoricalCrossentropy()
  mse_fn = kls.MeanSquaredError()
  ceacc_fn1 = k.metrics.CategoricalAccuracy()
  ceacc_fn2 = k.metrics.CategoricalAccuracy()
  batch_size = 256
  epochs = 10

  @tf.function
  def step(x, y1, y2, y3):
    with tf.GradientTape() as tape:
      p1, p2, p3 = model(x, training=True)
      l1 = ce_fn1(y1, p1)
      ceacc_fn1.update_state(y1, p1)
      l1_w = uncertaint_weight(l1, weights[0])
      l2 = ce_fn2(y2, p2)
      ceacc_fn2.update_state(y2, p2)
      l2_w = uncertaint_weight(l2, weights[1])
      l3 = mse_fn(y3, p3)
      l3_w = uncertaint_weight(l3, weights[2])
      l = (l1_w + l2_w + l3_w) if USE_RECONSTRUCT else (l1_w + l2_w)
      if not USE_LEARNABLE:
        l = (l1 + l2 + l3) if USE_RECONSTRUCT else (l1 + l2)

    grads = tape.gradient(l, model.trainable_variables + weights)
    optimizer.apply_gradients(zip(grads, model.trainable_variables + weights))
    if USE_LEARNABLE:
      return l, l1_w, l2_w, l3_w
    else:
      return l, l1, l2, l3

  (x_train, y_train), (x_test, y_test) = k.datasets.mnist.load_data()

  x_train = ((x_train / 255.)[..., None] - 0.5) / 0.5
  x_test = ((x_test / 255.)[..., None] - 0.5) / 0.5

  # 将3与7设为2种，其他设为1种
  y_train_1 = np.ones_like(y_train) * 2
  y_train_1[y_train == 2] = 0
  y_train_1[y_train == 6] = 1
  y_train_1 = k.utils.to_categorical(y_train_1, 3)
  y_train_2 = k.utils.to_categorical(y_train, 10)
  y_train_3 = x_train.copy()

  y_test_1 = np.ones_like(y_test) * 2
  y_test_1[y_test == 2] = 0
  y_test_1[y_test == 6] = 1
  y_test_1 = k.utils.to_categorical(y_test_1, 3)
  y_test_2 = k.utils.to_categorical(y_test, 10)
  y_test_3 = x_test.copy()

  train_ds = (tf.data.Dataset.from_tensor_slices(
      ((x_train), (y_train_1, y_train_2, y_train_3))).
      shuffle(batch_size * 100).
      batch(batch_size, drop_remainder=False).
      prefetch(None))

  test_ds = (tf.data.Dataset.from_tensor_slices(
      ((x_test), (y_test_1, y_test_2, y_test_3))).
      shuffle(batch_size * 100).
      batch(batch_size, drop_remainder=False).
      prefetch(None))

  for ep in range(epochs):
    for i, (in_x, (in_y1, in_y2, in_y3)) in enumerate(train_ds):
      l, l1, l2, l3 = step(in_x, in_y1, in_y2, in_y3)
      cur_step = optimizer.iterations.numpy()
      with writer.as_default():
        tf.summary.scalar('train/loss', l.numpy(), step=cur_step)
        tf.summary.scalar('train/loss_classify_1', l1.numpy(), step=cur_step)
        tf.summary.scalar('train/loss_classify_2', l2.numpy(), step=cur_step)
        tf.summary.scalar('train/acc_classify_1', ceacc_fn1.result().numpy(), step=cur_step)
        tf.summary.scalar('train/acc_classify_2', ceacc_fn2.result().numpy(), step=cur_step)
        tf.summary.scalar('train/loss_mse', l3.numpy(), step=cur_step)
        for _ in range(3):
          tf.summary.scalar(f'train/weight_{_}', weights[_].numpy(), step=cur_step)

  """ eval """
  ceacc_fn1 = k.metrics.CategoricalAccuracy()
  ceacc_fn2 = k.metrics.CategoricalAccuracy()
  for i, (in_x, (in_y1, in_y2, in_y3)) in enumerate(train_ds):
    l, l1, l2, l3 = step(in_x, in_y1, in_y2, in_y3)
    p1, p2, p3 = model(in_x, training=False)
    ceacc_fn1.update_state(in_y1, p1)
    ceacc_fn2.update_state(in_y2, p2)
  print('learnable', USE_LEARNABLE, 'Reconstruct', USE_RECONSTRUCT)
  print('3 classify  :', ceacc_fn1.result().numpy())
  print('10 classify :', ceacc_fn2.result().numpy())


# python ./source/_posts/Uncertainty-Weigh-Loss/demo.py True True
# python ./source/_posts/Uncertainty-Weigh-Loss/demo.py False True
# python ./source/_posts/Uncertainty-Weigh-Loss/demo.py True False
# python ./source/_posts/Uncertainty-Weigh-Loss/demo.py False False
