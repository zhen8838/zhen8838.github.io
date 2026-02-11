import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['STZhongsong']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

A = np.array([
    [0.9, 0.05, 0.05],
    [0.1, 0.8, 0.1],
    [0.04, 0.01, 0.95]
])

x = np.array([300000, 400000, 100000])


xs = [x]
for i in range(80):
  xs.append(xs[i] @ A)

xs = np.array(xs)
plt.plot(xs[:, 0], label='A')
plt.plot(xs[:, 1], label='B')
plt.plot(xs[:, 2], label='C')
plt.legend()
plt.title('马尔科夫链转移示意图', fontsize=20)
plt.tight_layout(True)
plt.savefig('/home/zqh/Documents/gitio/source/_posts/statis-learn-cp16/markov_chain.png')
