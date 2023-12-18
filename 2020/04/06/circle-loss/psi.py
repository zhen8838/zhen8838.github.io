import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

# m = 4
# theta_range = np.arange(m + 1) * np.pi / m
# psi = np.hstack([
#     np.power(-1, k) * np.cos(m * np.linspace(theta_range[k], theta_range[k + 1]))
#     - 2 * k for k in range(m)
# ])
# theta = np.hstack(
#     [np.linspace(theta_range[k], theta_range[k + 1]) for k in range(m)])

# plt.figure(figsize=(10, 6))
# plt.plot(theta, psi, label=r'A softmax m=4')

# m = 0.35
# s = 30
# theta = np.linspace(0, np.pi, 50 * 5)
# psi = (np.cos(theta) - m)
# plt.plot(theta, psi, label=r'AM softmax m=0.35')


def relu(x):
  return (np.abs(x) + x) / 2


m = 0.25
gamma = 30
Op = 1 + m
On = -m
Dp = 1 - m
Dn = m
theta = np.linspace(0, np.pi, 50 * 5)

sp = np.cos(theta)
Ap = relu(Op - sp)
psi_sp = Ap * (sp - Dp)

sn = np.cos(theta)
An = relu(sn - On)
psi_sn = An * (sn - Dn)

plt.plot(theta, psi_sp, label=r'$\alpha_{p}^{i}\left(s_{p}^{i}-\Delta_{p}\right)$')
plt.plot(theta, psi_sn, label=r'$\alpha_{n}^{i}\left(s_{n}^{i}-\Delta_{n}\right)$')

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
plt.title(f'margin m={m}')
plt.savefig('source/_posts/circle-loss/circle-loss-4.png')
plt.show()


X, Y = np.meshgrid(theta, theta)

psi_sp_, psi_sn_ = np.meshgrid(psi_sp, psi_sn)

Z = psi_sn_ - psi_sp_

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig: plt.Figure = plt.figure()
ax: Axes3D = fig.gca(projection='3d')
fig.set_tight_layout(True)
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
cset = ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=cm.coolwarm)
ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
ax.yaxis.set_major_formatter(FuncFormatter(pi_formatter))
ax.set_xlabel(r'$\cos(\theta_p)$')
ax.set_ylabel(r'$\cos(\theta_n)$')
ax.set_zlabel(r'margin')
ax.set_zlim(-2, 5)
ax.view_init(elev=25., azim=120.)
ax.set_title(r'$\alpha_{n}\left(s_{n}-\Delta_{n}\right)-\alpha_{p}\left(s_{p}-\Delta_{p}\right)$')
plt.savefig('source/_posts/circle-loss/circle-loss-5.png')
plt.show()
