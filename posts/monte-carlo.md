---
title: 蒙特卡洛法
categories:
- 机器学习
mathjax: true
toc: true
date: 2018-11-15 20:12:27
tags:
-   概率论
---

在实际过程中,有的算法不能保证每次都能的到正确的解.蒙特卡洛算法则是在一般情况下可以保证对问题的所有实例都以高概率给出正确解,但是通常无法判定一个具体解是否正确.

<!--more-->

# 算法基本思想

设$p$是实数,且$\frac{1}{2} < p < 1$.如果该算法对于问题的任一实例得到正确解的概率不小于$p$,则称该蒙特卡洛算法是$p$正确的,且称$p-\frac{1}{2}$是该算法的优势.

在一般情况下,设$\varepsilon$和$\delta$是两个正实数,且$\varepsilon+\delta<\frac{1}{2}$. 设$MC(x)$是$\frac{1}{2}+\varepsilon$正确的.且$C_\varepsilon=-\frac{2}{log(1-4\varepsilon^2)}$.

如果调用算法至少$C_\varepsilon log(\frac{1}{\delta})$次,并返回各次调用出现频次最高的解,就可以得到一个$1-\delta$正确的蒙特卡洛算法.

由此可见,不论算法$MC(x)$的优势$\varepsilon>0$多么小,都可以通过反复调用来放大算法的优势,最终得到的算法具有可以接受的错误概率.

证明如下,设$n>C_\varepsilon log(\frac{1}{\delta})$是重复调用$\frac{1}{2}+\varepsilon$正确的蒙特卡洛算法次数,且$p=\frac{1}{2}+\varepsilon$,$q=1-p=\frac{1}{2}-\varepsilon,m=\lfloor \frac{n}{2} \rfloor+1$.即经过$n$次调用算法,问题的正确解至少应出现$m$次,其出现错误的概率最多为:
$$\begin{aligned}
  & \sum^{m-1}_{i=0}Prob{\ n次调用出现i次正确解} \\
  & \leq \sum^{m-1}_{i=0} \lgroup \begin{aligned} n \\ i \end{aligned} \rgroup  
  p^i=q^{n-i}=(pq)^{\frac{n}{2}} \sum^{m-1}_{i=0} \lgroup \begin{aligned} n \\ i \end{aligned} \rgroup   (\frac{q}{p})^{\frac{n}{2}-i}    \\
  & \leq (pq)^{\frac{n}{2}} \sum^{m-1}_{i=0} \lgroup \begin{aligned} n \\ i \end{aligned} \rgroup  \ \ \ \ \ \ \ \ \ (\frac{q}{p}<1,\frac{n}{2}-i\geq0) \\
  & \leq (pq)^{\frac{n}{2}} \sum^{m-1}_{i=0} \lgroup \begin{aligned} n \\ i \end{aligned} \rgroup = (pq)^{\frac{n}{2}}2^n=(4pq)^{\frac{n}{2}}=(1-4\varepsilon^2)^{\frac{n}{2}} \\
  & \leq (1-4\varepsilon^2)^{\frac{c_\varepsilon}{2}}\log(1/\delta)\ \ \ \ \ \ \ \ \ (0<(1-4\varepsilon^2)<1) \\
  & = 2^{-\log(\frac{1}{\delta})} \\
  & = \delta       \ \ \ \ \ \ \ \ \  (当\ x>0 \Rightarrow x^{\frac{1}{\log x}}=2)
\end{aligned} $$


# 实例

## 主元素问题

设$T[1:n]$是一个含有$n$个元素的数组,当$|\ \{i\ |\ T[i]=x\}\ |>\frac{n}{2}$时,称元素$x$是数组$T$的主元素.

编写一个程序如下:

```cpp
RandomNumber rnd;
template<class Type> bool Majority(Type *T, int n) {
    // 判断主元素的蒙卡罗特方法
    int i= rnd.Random(n) + 1;
    Type x= T[i];
    int k= 0;
    // 统计主元个数
    for (int j= 1; j <= n; ++j) {
        if (T[j] == x) { k++; }
    }
    return (k > n / 2); //返回是否为主元
}
```



上述算法是一个偏真的$\frac{1}{2}$算法,因为当此数组里面没有主元时,必然返回`False`,否则将以大于$\frac{1}{2} $的概率返回`True`.


##  执行两次

执行两次上述程序:

```cpp
template<class Type> bool Majority2(Type *T, int n) {
    // 调用两次Majority算法
    if (Majority(T, n)) {
        return true;
    } else {
        return Majority(T, n);
    }
}
```


如果$T$不含主元素
- 算法`Majority2`肯定返回`False`.

如果$T$含有主元素:

- 第一次算法`Majority`返回`True`的概率大于$\frac{1}{2}  $,此时算法`Majority2`必然返回`True`
- 第一次算法`Majority`返回`False`的概率为$1-p$,第二次算法`Majority`返回`True`的概率为`p`

因此:
$$ 
  p+(1-p)p=1-(1-p)^2>\frac{3}{4}  
 $$



## 执行$\log(\frac{1}{\varepsilon})次$

此时的算法错误概率小于$\varepsilon$

```cpp
template<class Type> bool MajorityMC(Type *T, int n, double e) {
    // 重复log(1/ε)次调用算法
    int k= ceil(log(1 / e) / log(2));
    for (int i= 1; i <= k; ++i) {
        if (Majority(T, n)) { return true; }
    }
    return false;
}
```


 算法的时间复杂度为:$O(n\log(\frac{1}{\varepsilon}))$


#  实例运行


```python
#! python3


import random as rnd
import math
# 利用随机数生成的长度100 主元为3的列表
rdlist = [3, 3, 3, 3, 3, 3, 6, 3, 8, 3, 3, 11, 3, 3, 3, 3, 16, 3, 18, 3, 20, 3, 22, 23, 3, 3, 26, 27, 3, 29, 3, 3, 32, 33, 3, 3, 3, 3, 3, 39, 3, 41, 3, 43, 3, 3, 3, 3, 3, 49, 3,
          3, 3, 3, 54, 3, 3, 57, 3, 3, 60, 61, 3, 3, 64, 3, 66, 3, 3, 3, 3, 71, 72, 3, 3, 3, 3, 3, 78, 3, 3, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]



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

```




```sh
➜  gitio /usr/bin/python3 /media/zqh/文档1/Blog/gitio/source/_posts/monte-carlo/monte.py
执行Majority 10万次正确概率为0.54821%
执行Majority2 10万次正确概率为0.7967%
执行MajorityMC 10万次正确概率为0.95879%
```