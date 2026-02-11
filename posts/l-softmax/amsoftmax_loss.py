from tensorflow.python import keras
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.constraints import unit_norm

(train_x, train_y), (test_x, test_y) = keras.datasets.fashion_mnist.load_data()

train_x = K.reshape(train_x, (-1, 784))
train_y = keras.utils.to_categorical(train_y, 10)


model = keras.Sequential([Input(shape=(784,)),
                          Dense(512, keras.activations.relu),
                          Dense(256, keras.activations.relu),
                          Dense(128, keras.activations.relu),
                          Lambda(lambda x: K.l2_normalize(x, 1)),
                          Dense(10, use_bias=False, kernel_constraint=unit_norm())])


def am_softmax_loss(y_true, y_pred, scale=30, margin=0.35):
    # NOTE 预测出来的x就是归一化后的,并且W也是归一化后的,所以y_pred就是cos(𝜃)
    y_pred = (y_true * (y_pred - margin) + (1 - y_true) * y_pred) * scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


model.compile(loss=am_softmax_loss, optimizer=keras.optimizers.Adam(),
              metrics=[keras.metrics.CategoricalAccuracy()])

model.fit(x=train_x, y=train_y, batch_size=128, epochs=5)
