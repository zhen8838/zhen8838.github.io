---
title: aggressicecows
date: 2018-09-27 23:35:41
mathjax: true
tags:
-   LeetCode
categories:
-   数据结构
---


Farmer John has built a new long barn, with N (2 <= N <= 100,000) stalls. The stalls are located along a straight line at positions x1,...,xN (0 <= xi <= 1,000,000,000).

His C (2 <= C <= N) cows don't like this barn layout and become aggressive towards each other once put into a stall. To prevent the cows from hurting each other, FJ want to assign the cows to the stalls, such that the minimum distance between any two of them is as large as possible. What is the largest minimum distance?

<!--more-->


## 输入

* Line 1: Two space-separated integers: N and C
* Lines 2..N+1: Line i+1 contains an integer stall location, xi

## 输出

* Line 1: One integer: the largest minimum distance

## 样例输入

`5 3
1
2
8
4
9`


## 样例输出

`3`


# 代码

```cpp
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;
#define Dcout(x) cout << #x << ": " << (x) << endl

bool putcows(int cows, vector<int> &a, int dis) {
    /* 首先在第一个位置放牛 */
    auto last= a.begin();
    for (auto it= a.begin() + 1; it != a.end(); ++it) {
        /* 距离大于mid则放牛 */
        if ((*it - *last) >= dis) {
            cows--;
            last= it;
        }
    }
    if (cows <= 0) {
        return true;
    } else {
        return false;
    }
}

int main(int argc, char const *argv[]) {
    int n, c;
    vector<int> postion;
    cin >> n >> c;
    while (n--) {
        int tmp;
        cin >> tmp;
        postion.push_back(tmp);
    }
    /* 第一步：排序 */
    sort(postion.begin(), postion.end());
    int left= 0;
    /* 第二步：设最大距离为s "0<s<=(xn-x1)/(c-1)" */
    int right= (postion.back() - postion[0]) / (c - 1);
    while (left <= right) {
        /* 第三步：尝试中间解 */
        int mid= (left + right) / 2;
        /* 第四步：判断当前间隔是否能放下牛 */
        if (putcows(c - 1, postion, mid)) {
            /* 可以放下，范围向右缩小 */
            left= mid + 1;
        } else {
            /* 不可以放下，范围向左缩小 */
            right= mid - 1;
        }
    }
    cout << left - 1;
    return 0;
}
```
