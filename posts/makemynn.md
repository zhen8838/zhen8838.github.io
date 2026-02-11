---
title: Python神经网络编程
categories:
  - 深度学习
mathjax: true
toc: true
date: 2018-11-21 20:27:56
tags:
- Python
---

这两天看了这本书,非常好.一下就可以把人讲懂.我试着写了他的例子

<!--more-->


# 程序


```python
import scipy.special
import numpy
import matplotlib.pyplot as plt
import pickle


class neuralNetwork:

    def __init__(self, inputnodes: int, hiddennodes: int, outputnodes: int, learningrate: int):
        # set the node number
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # set the learn rate
        self.lr = learningrate
        # generate the W_input_hidden and W_hidden_output matrix

        # hidden = w_ih * innodes  =>   w_ih is [hidden x input]
        self.w_ih = numpy.random.normal(scale=pow(self.hnodes, -0.5),
                                        size=(self.hnodes, self.inodes))
        # out = w_ho * outnodes  =>   w_ho is [out x hidden]
        self.w_ho = numpy.random.normal(scale=pow(self.onodes, -0.5),  # 节点数的 -1/2 次
                                        size=(self.onodes, self.hnodes))
        # set the active function
        self.active_fuc = lambda x: scipy.special.expit(x)
        self.re_active_fuc = lambda x: scipy.special.logit(x)
    # we need set the target to train

    def train(self, input_list, target_list):
        # convert list to 2d array
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.w_ih, inputs)
        hidden_outputs = self.active_fuc(hidden_inputs)
        # calculate signals of the output layer
        final_inputs = numpy.dot(self.w_ho, hidden_outputs)
        final_outputs = self.active_fuc(final_inputs)

        # hidden_error = output -target
        output_errors = targets-final_outputs
        hidden_errors = numpy.dot(self.w_ho.T, output_errors)

        # input_error = hidden_error - hidden_output
        # update the W_ho
        self.w_ho += self.lr * \
            numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),
                      numpy.transpose(hidden_outputs))

        self.w_ih += self.lr * \
            numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),
                      numpy.transpose(inputs))
        pass

    def query(self, input_list: numpy.ndarray)->numpy.ndarray:
        # convt  input_list to 2d array
        inputs = numpy.array(input_list, ndmin=2).T
        # X_hidden = W_ih · inputs
        hidden_input = numpy.dot(self.w_ih, inputs)
        hidden_output = self.active_fuc(hidden_input)

        final_input = numpy.dot(self.w_ho, hidden_output)
        final_output = self.active_fuc(final_input)

        return final_output

    def backquery(self, targets_list: list)->numpy.ndarray:
        # transpose the targets list to a vertical array
        final_outputs = numpy.array(targets_list, ndmin=2).T

        # calculate the signal into the final output layer
        final_inputs = self.re_active_fuc(final_outputs)

        # 权重矩阵可能无法求逆，所以直接乘转置
        hidden_outputs = numpy.dot(self.w_ho.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        # calculate the signal into the hidden layer
        hidden_inputs = self.re_active_fuc(hidden_outputs)

        # calculate the signal out of the input layer
        inputs = numpy.dot(self.w_ih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs


if __name__ == "__main__":
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10

    learn_rate = 0.3

    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learn_rate)

    # # read the data set
    # train_file = open("./mnist_train.csv")
    # train_data = train_file.readlines()  # type:list
    # train_file.close()

    # for record in train_data:
    #     all_value = record.split(',')
    #     # set inputs data to 0.01 ~ 0.99
    #     inputs = (numpy.asfarray(all_value[1:])/255.0*0.99)+0.01
    #     # set target data
    #     targets = numpy.zeros(output_nodes)+0.01  # set the false is 0.01
    #     targets[int(all_value[0])] = 0.99  # set the right is 0.99

    #     n.train(inputs, targets)

    # save the model !
    # savefile = open('nnmodel', 'wb')
    # pickle.dump(n.w_ih, savefile, 1)
    # pickle.dump(n.w_ho, savefile, 1)
    # savefile.close()
    # read the model !
    savefile = open('/home/zqh/Program/Python_study/NN/nnmodel', 'rb')
    n.w_ih = pickle.load(savefile)
    n.w_ho = pickle.load(savefile)
    savefile.close()
    # read the test data
    test_file = open("./mnist_test.csv")
    test_data = test_file.readlines()  # type:list
    test_file.close()
    scorecard = []
    for record in test_data:
        test_value = record.strip().split(',')
        inputs = (numpy.asfarray(test_value[1:]) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        # print("原始：{} 预测：{}".format(test_value[0], label))
        if int(test_value[0]) == label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    print("准确率 = {:.2f}%".format(
        sum(scorecard) / len(scorecard)*100.0))
    print('反向查询：')
    image_array = n.backquery(
        [0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    plt.imshow(image_array.reshape(28, 28), cmap='Greys')
    plt.show()

```




# 运行结果

```sh
➜  NN /usr/bin/python3 /home/zqh/Program/Python_study/NN/nn.py
准确率 = 94.68%
反向查询：
```

![数字2的图像](./makemynn/1.png)