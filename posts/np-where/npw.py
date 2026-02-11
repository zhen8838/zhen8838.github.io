import numpy as np


if __name__ == "__main__":
    # 构造数组
    a = np.zeros((7, 10, 5, 25))
    # 首先自己设置几个值
    dim1 = np.random.randint(0, high=7, size=3)
    dim2 = np.random.randint(0, high=10, size=3)
    dim3 = np.random.randint(0, high=5, size=3)
    a[dim1, dim2, dim3, 4] = 1
    # 我需要在最后一维的第5列找到大于0.7的元素的索引
    idex1, idex2, idex3 = np.where(a[..., 4] > .7)
    print('dim1:',dim1, idex1)
    print('dim2:',dim2, idex2)
    print('dim3:',dim3, idex3)
