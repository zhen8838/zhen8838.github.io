import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, binom, beta, uniform

plt.rcParams['font.sans-serif'] = ['STZhongsong']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def target(like: binom, prior: beta, n, h, theta):
  if theta < 0 or theta > 1:
    return 0
  else:
    return like(n, theta).pmf(h) * prior.pdf(theta)


n = 100
h = 61
a = 10
b = 10
like = binom
prior = beta(a, b)
sigma = 0.3

naccept = 0

niters = 10000
thetas = np.zeros(niters + 1)
thetas[0] = 0.1  # 设定初始参数
for i in range(niters):
  # 以当前状态下,转移矩阵为N(0,sigma)
  theta_p = norm(thetas[i], sigma).rvs()
  #
  rho = min(1, target(like, prior, n, h, theta_p) /
            target(like, prior, n, h, thetas[i]))
  u = np.random.uniform()
  if u < rho:
    naccept += 1
    thetas[i + 1] = theta_p
  else:
    thetas[i + 1] = thetas[i]
nmcmc = len(thetas) // 2  # 为了尽量选取平稳状态的采样值


x = np.linspace(0, 1, 200)
true_posterior = prior.pdf(x) * like(n, x).pmf(h)  # 先验*似然
true_posterior /= (np.sum(true_posterior) / np.size(true_posterior))
plt.hist(thetas[nmcmc:], 80, density=True, label='posterior')
plt.hist(prior.rvs(nmcmc), 80, density=True, label='prior')
plt.plot(x, true_posterior, label='true posterior', c='red')
plt.legend()
plt.title(f"采样效率 = {naccept / niters:.3f}", fontsize=20)
plt.tight_layout(True)
plt.savefig('/home/zqh/Documents/gitio/source/_posts/statis-learn-cp16/metro_hastings_1.png')
plt.show()

""" 收敛状态评估 """


def mh_coin(niters, init_theta):
  thetas = np.zeros(niters + 1)
  thetas[0] = init_theta  # 设定初始参数
  for i in range(niters):
    # 以当前状态下,转移矩阵为N(0,sigma)
    theta_p = norm(thetas[i], sigma).rvs()
    #
    rho = min(1, target(like, prior, n, h, theta_p) /
              target(like, prior, n, h, thetas[i]))
    u = np.random.uniform()
    if u < rho:
      thetas[i + 1] = theta_p
    else:
      thetas[i + 1] = thetas[i]

  return thetas


thetass = [mh_coin(100, i) for i in np.arange(0.1, 1.1, 0.2)]

for thetas, init in zip(thetass, np.arange(0.1, 1.1, 0.2)):
  plt.plot(thetas, '-', label=f'init={init:.1f}')
plt.legend()
plt.title(f"马尔科夫链收敛状态评估", fontsize=20)
plt.tight_layout(True)
plt.savefig('/home/zqh/Documents/gitio/source/_posts/statis-learn-cp16/metro_hastings_2.png')
plt.show()
