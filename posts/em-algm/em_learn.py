from numpy.core.umath_tests import matrix_multiply as mm
from scipy.optimize import minimize
from scipy.stats import bernoulli, binom
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)


""" 做五次实验 """
n = 10  # 每次实验投掷10次
theta_A = 0.8
theta_B = 0.3
theta_0 = [theta_A, theta_B]
# 两个硬币对应两个不同thta的二项分布
coin_A = bernoulli(theta_A)
coin_B = bernoulli(theta_B)

zs = [0, 0, 1, 0, 1]  # 代表使用的硬币为哪个
# 得到实验结果
xs = np.array([np.sum(coin.rvs(n)) for coin in [coin_A, coin_A, coin_B, coin_A, coin_B]])
print(xs)  # [7 9 2 6 0]

""" 精确求解 """
# 计算所有的硬币A朝上的比例作为概率
ml_A = np.sum(xs[[0, 1, 3]]) / (3.0 * n)
# 计算所有的硬币B朝上的比例作为概率
ml_B = np.sum(xs[[2, 4]]) / (2.0 * n)

print(ml_A, ml_B)  # 0.7333333333333333 0.1


""" 数值估计 """


def neg_loglik(thetas, n, xs, zs):
    """ 对数似然计算函数
        这里的的二项分布是次数为n，概率可能为 theta_A 或 theta_B。
        将每个二项分布对应x的对数概率密度函数求和后取相反数，接下来就是最小化此函数即可。
    """

    logpmf = -np.sum([binom(n, thetas[z]).logpmf(x) for (x, z) in zip(xs, zs)])
    return logpmf


bnds = [(0, 1), (0, 1)]
# 使用优化策略进行最小化求解
res = minimize(neg_loglik, [0.5, 0.5], args=(n, xs, zs),
               bounds=bnds, method='tnc', options={'maxiter': 100})
print(res)
""" fun: 7.655267754139319
     jac: array([-7.31859018e-05, -7.58504370e-05])
 message: 'Converged (|f_n-f_(n-1)| ~= 0)'
    nfev: 17
     nit: 6
  status: 1
 success: True
       x: array([0.73333285, 0.09999965]) """


""" 下面是em算法的例子 """

""" 第一种方式 """
xs = np.array([(5, 5), (9, 1), (8, 2), (4, 6), (7, 3)])
thetas = np.array([[0.6, 0.4], [0.5, 0.5]])  # 初始化参数A B (向上概率，向下概率)

tol = 0.01  # 变化容忍度
max_iter = 100  # 迭代次数

ll_old = 0
for i in range(max_iter):
    exp_A = []
    exp_B = []

    ll_new = 0

    # ! E-step: 计算可能的概率分布
    for x in xs:
        # 求解当前theta下两个分布的对数似然
        ll_A = np.sum(x * np.log(thetas[0]))  # 多项式分布的对数似然函数 (忽略常数).
        ll_B = np.sum(x * np.log(thetas[1]))

        w_A = np.exp(ll_A) / (np.exp(ll_A) + np.exp(ll_B))  # 求出概率A的权重
        w_B = np.exp(ll_B) / (np.exp(ll_A) + np.exp(ll_B))  # 求出概率B的权重

        exp_A.append(w_A * x)  # 概率A权重乘上样本
        exp_B.append(w_B * x)  # 概率B权重乘上样本

        ll_new += w_A * ll_A + w_B * ll_B  # 计算当前的theta值对应的似然值

    # ! M-step: 为给定的分布更新当前参数
    thetas[0] = np.sum(exp_A, 0) / np.sum(exp_A)  # 利用更新之后的样本值计算出当前A的theta值
    thetas[1] = np.sum(exp_B, 0) / np.sum(exp_B)  # 利用更新之后的样本值计算出当前B的theta值

    # 输出每个x和当前参数估计z的分布
    print("Iteration: %d" % (i + 1))
    print("theta_A = %.2f, theta_B = %.2f, ll = %.2f" % (thetas[0, 0], thetas[1, 0], ll_new))

    if np.abs(ll_new - ll_old) < tol:
        break
    ll_old = ll_new


""" 我的方式 """
n = 10  # 实验次数
m_xs = np.array([5, 9, 8, 4, 7])  # 向上次数
theta_A = 0.6
theta_B = 0.5


tol = 0.01  # 变化容忍度
max_iter = 100  # 迭代次数
loglike_old = 0  # 初始对数似然值
for i in range(max_iter):
    cnt_A = []
    cnt_B = []
    loglike_new = 0  # 新的对数似然值
    # ! E-step
    for x in m_xs:
        pmf_A = binom(n, theta_A).pmf(x)  # 当前theta下 A的概率
        pmf_B = binom(n, theta_B).pmf(x)  # 当前theta下 B的概率

        logpmf_A = binom(n, theta_A).logpmf(x)  # 当前theta下 A的对数概率
        logpmf_B = binom(n, theta_B).logpmf(x)  # 当前theta下 B的对数概率

        weight_A = pmf_A / (pmf_A + pmf_B)  # 求得权重
        weight_B = pmf_B / (pmf_A + pmf_B)  # 求得权重

        cnt_A.append(weight_A * np.array([x, n - x]))  # 概率A权重乘上样本，得到新的硬币次数统计
        cnt_B.append(weight_B * np.array([x, n - x]))  # 概率B权重乘上样本，得到新的硬币次数统计

        loglike_new += weight_A * logpmf_A + weight_B * logpmf_B  # 计算当前的theta值对应的似然值
    # ! M-step
    theta_A = np.sum(cnt_A, 0)[0] / np.sum(cnt_A)  # 硬币A向上的次数除以总次数即为theta_A
    theta_B = np.sum(cnt_B, 0)[0] / np.sum(cnt_B)  # 硬币B向上的次数除以总次数即为theta_B

    # 输出每个x和当前参数估计z的分布
    print("Iteration: %d" % (i + 1))
    print("theta_A = %.2f, theta_B = %.2f, ll = %.2f" % (theta_A, theta_B, loglike_new))

    if np.abs(loglike_new - loglike_old) < tol:
        break
    loglike_old = loglike_new


""" k means """

from numpy.core.umath_tests import inner1d
import numpy as np
import seaborn as sns


def kmeans(xs, k, max_iter=10):
    """K-means 算法."""
    idx = np.random.choice(len(xs), k, replace=False)
    cs = xs[idx]
    for n in range(max_iter):
        ds = np.array([inner1d(xs - c, xs - c) for c in cs])
        zs = np.argmin(ds, axis=0)
        cs = np.array([xs[zs == i].mean(axis=0) for i in range(k)])
    return (cs, zs)


iris = sns.load_dataset('iris')
data = iris.iloc[:, :4].values
cs, zs = kmeans(data, 3)
iris['cluster'] = zs
sns.pairplot(iris, hue='cluster', diag_kind='kde', vars=iris.columns[:4])
plt.show()


""" 混合高斯模型 """
from scipy.stats import multivariate_normal


def normalize(xs, axis=None):
    """Return normalized marirx so that sum of row or column (default) entries = 1."""
    if axis is None:
        return xs / xs.sum()
    elif axis == 0:
        return xs / xs.sum(0)
    else:
        return xs / xs.sum(1)[:, None]


def mix_mvn_pdf(xs, pis, mus, sigmas):
    return np.array([pi * multivariate_normal(mu, sigma).pdf(xs) for (pi, mu, sigma) in zip(pis, mus, sigmas)])


def em_gmm_orig(xs, pis, mus, sigmas, tol=0.01, max_iter=100):

    n, p = xs.shape
    k = len(pis)

    ll_old = 0
    for i in range(max_iter):
        exp_A = []
        exp_B = []
        ll_new = 0

        # ！ E-step
        ws = np.zeros((k, n))
        for j in range(len(mus)):
            for i in range(n):
                # 遍历所有的 mu，sigma，pi 来计算概率密度
                ws[j, i] = pis[j] * multivariate_normal(mus[j], sigmas[j]).pdf(xs[i])
        ws /= ws.sum(0)  # 根据概率密度求权值

        # M-step
        # NOTE 下面的更新过程是根据公式来计算的！
        pis = np.zeros(k)
        for j in range(len(mus)):
            for i in range(n):
                pis[j] += ws[j, i]  # 使用权值更新pi
        pis /= n

        mus = np.zeros((k, p))
        for j in range(k):
            for i in range(n):
                mus[j] += ws[j, i] * xs[i]  # 使用权值更新mu
            mus[j] /= ws[j, :].sum()

        sigmas = np.zeros((k, p, p))
        for j in range(k):
            for i in range(n):
                ys = np.reshape(xs[i] - mus[j], (2, 1))
                sigmas[j] += ws[j, i] * np.dot(ys, ys.T)  # 使用权值更新sigma
            sigmas[j] /= ws[j, :].sum()

        # 更新对数似然函数
        ll_new = 0.0
        for i in range(n):
            s = 0
            for j in range(k):
                s += pis[j] * multivariate_normal(mus[j], sigmas[j]).pdf(xs[i])
            ll_new += np.log(s)

        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

    return ll_new, pis, mus, sigmas


# 构建数据集
n = 1000
_mus = np.array([[0, 4], [-2, 0]])
_sigmas = np.array([[[3, 0], [0, 0.5]], [[1, 0], [0, 2]]])
_pis = np.array([0.6, 0.4])
xs = np.concatenate([np.random.multivariate_normal(mu, sigma, int(pi * n))
                     for pi, mu, sigma in zip(_pis, _mus, _sigmas)])

# 初始化预测值
pis = normalize(np.random.random(2))  # pi 要经过归一化
mus = np.random.random((2, 2))
sigmas = np.array([np.eye(2)] * 2)

# 使用EM算法拟合
ll1, pis1, mus1, sigmas1 = em_gmm_orig(xs, pis, mus, sigmas)

# 绘图
intervals = 101
ys = np.linspace(-8, 8, intervals)
X, Y = np.meshgrid(ys, ys)
_ys = np.vstack([X.ravel(), Y.ravel()]).T

z = np.zeros(len(_ys))
for pi, mu, sigma in zip(pis1, mus1, sigmas1):
    z += pi * multivariate_normal(mu, sigma).pdf(_ys)
z = z.reshape((intervals, intervals))

ax = plt.subplot(111)
plt.scatter(xs[:, 0], xs[:, 1], alpha=0.2)
plt.contour(X, Y, z, N=10)
plt.axis([-8, 6, -6, 8])
ax.axes.set_aspect('equal')
plt.tight_layout()
plt.show()
