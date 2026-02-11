import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

m = 4
theta_range = np.arange(m + 1) * np.pi / m
psi = np.hstack([np.power(-1, k) * np.cos(m * np.linspace(theta_range[k], theta_range[k + 1])) - 2 * k for k in range(m)])
theta = np.hstack([np.linspace(theta_range[k], theta_range[k + 1]) for k in range(m)])

plt.figure(figsize=(10, 6))
plt.plot(theta, psi, label=r'A softmax m=4')

m = 0.35
s = 30
theta = np.linspace(0, np.pi, 50 * 5)
psi = (np.cos(theta) - m)
plt.plot(theta, psi, label=r'AM softmax m=0.35')

ax = plt.gca()


def pi_formatter(x, pos):
    """ 
    比较罗嗦地将数值转换为以pi/4为单位的刻度文本 
    """
    m = np.round(x / (np.pi / 4))
    n = 4
    if m % 2 == 0:
        m, n = m / 2, n / 2
    if m % 2 == 0:
        m, n = m / 2, n / 2
    if m == 0:
        return "0"
    if m == 1 and n == 1:
        return "$\pi$"
    if n == 1:
        return r"$%d \pi$" % m
    if m == 1:
        return r"$\frac{\pi}{%d}$" % n
    return r"$\frac{%d \pi}{%d}$" % (m, n)


# 设置两个坐标轴的范围
# plt.ylim(-1.5, 1.5)
# plt.xlim(0, np.max())

# 设置图的底边距
plt.subplots_adjust(bottom=0.15)

# plt.grid()  # 开启网格

# 主刻度为pi/4
ax.xaxis.set_major_locator(MultipleLocator(np.pi / 4))

# 主刻度文本用pi_formatter函数计算
ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))

# 副刻度为pi/20
ax.xaxis.set_minor_locator(MultipleLocator(np.pi / 20))

# 设置刻度文本的大小
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(16)

plt.legend()
plt.title(r'$\psi(\theta)$')
plt.savefig('source/_posts/l-softmax/psi2.png')
plt.show()
