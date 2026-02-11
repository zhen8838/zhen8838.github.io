#! python3


import random as rnd
import math
# 利用随机数生成的长度100 主元为3的列表
rdlist = [3, 3, 3, 3, 3, 3, 6, 3, 8, 3, 3, 11, 3, 3, 3, 3, 16, 3, 18, 3, 20, 3, 22, 23, 3, 3, 26, 27, 3, 29, 3, 3, 32, 33, 3, 3, 3, 3, 3, 39, 3, 41, 3, 43, 3, 3, 3, 3, 3, 49, 3,
          3, 3, 3, 54, 3, 3, 57, 3, 3, 60, 61, 3, 3, 64, 3, 66, 3, 3, 3, 3, 71, 72, 3, 3, 3, 3, 3, 78, 3, 3, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
rdlist.count


def Majority(lst, n):
    k = lst.count(lst[rnd.randint(0, n-1)])
    return k > n/2


def Majority2(lst, n):
    if Majority(lst, n):
        return True
    else:
        return Majority(lst, n)


def MajorityMC(lst, n, e):
    for i in range(0, math.ceil(math.log(1/e)/math.log(2))):
        if Majority(lst, n):
            return True
    return False


# 统计执行Majority 10万次得到正确解的概率
truecnt = 0
for i in range(0, 100000):
    # 直接写长度减少开销
    if Majority(rdlist, 100):
        truecnt += 1
print("执行Majority 10万次正确概率为{}%".format(truecnt/100000.0))


truecnt = 0
for i in range(0, 100000):
    # 直接写长度减少开销
    if Majority2(rdlist, 100):
        truecnt += 1
print("执行Majority2 10万次正确概率为{}%".format(truecnt/100000.0))

truecnt = 0
for i in range(0, 100000):
    # 直接写长度减少开销
    if MajorityMC(rdlist, 100, 0.1):
        truecnt += 1
print("执行MajorityMC 10万次正确概率为{}%".format(truecnt/100000.0))
