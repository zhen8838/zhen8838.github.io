from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats._discrete_distns import binom
from scipy.stats._continuous_distns import beta, norm
from scipy.stats._continuous_distns import uniform

plt.rcParams['font.sans-serif'] = ['STZhongsong']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

x = np.linspace(0, 1)
q_x = uniform


def p_x(x):
  return x * np.cos(71 * x) + np.sin(13 * x) + 2


point = 0.75
c = np.max(p_x(x))
upper = c * q_x.pdf(point)
plt.plot(x, c * q_x.pdf(x), label='$cq(x)$')
plt.plot(x, p_x(x), label='$p(x)$')
plt.arrow(point, 0, 0, p_x(point), linewidth=1,
          head_width=0.03, head_length=0.01, fc='g', ec='g')
plt.arrow(point, upper, 0, -(upper - p_x(point)), linewidth=1,
          head_width=0.03, head_length=0.01, fc='r', ec='r')
plt.text(point + .05, 2., 'Reject', fontsize=16)
plt.text(point + .05, 0.75, 'Accept', fontsize=16)
plt.title('接受-拒绝抽样示意图', fontsize=20)
plt.legend()
plt.tight_layout(True)
plt.savefig('/home/zqh/Documents/gitio/source/_posts/statis-learn-cp16/mcmc_accept_1.png')
plt.show()

n = 500000
# 从均匀分布中采样x
sample_x = q_x.rvs(size=n)
u = uniform.rvs(size=n)  # 这里采样u作为比例进行计算
v = sample_x[u <= (p_x(sample_x) / (c * q_x.pdf(sample_x)))]  # v为接受的样本

hist, bin_edges = np.histogram(v, bins=100, normed=True)
factor = 2 # 这个参数本来应该是通过p(x)的cdf计算得到的,我这里偷懒了
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
plt.step(bin_centers, hist * factor, linewidth=2, label='sampling')
plt.plot(x, c * q_x.pdf(x), label='$cq(x)$')
plt.plot(x, p_x(x), label='$p(x)$')
plt.legend()
plt.title(f'接受-拒绝抽样 接受率{np.size(v)/n:.3f}', fontsize=20)
plt.tight_layout(True)
plt.savefig('/home/zqh/Documents/gitio/source/_posts/statis-learn-cp16/mcmc_accept_2.png')
plt.show()
