---
title: 字典序
date: 2018-09-10 21:39:21
tags:
-   C
categories:
-   数据结构
---


直接上程序：


<!--more-->



```cpp
#include <algorithm>
#include <iostream>
#include <string>
using namespace std;

void swap(string::iterator A, string::iterator B) {
    char tmp;
    tmp= *A;
    *A= *B;
    *B= tmp;
}

int main(int argc, char const *argv[]) {
    string ss;
    string::iterator Pj, Pk;
    cin >> ss;
    while (1) {
        /* 从右端起，找出第一个比右边小的位置j（j从左端开始计算） */
        for (Pj= ss.end() - 2; Pj != ss.begin(); Pj--) {
            if (*Pj < *(Pj + 1)) {
                break;
            }
        }
        if (*Pj >= *(Pj + 1)) { /* 没有该位置则结束 */
            return 0;
        } else {
            /* 在Pj的右边找出所有比Pj大的数中最小的数字Pk */
            for (Pk= ss.end() - 1; Pk > Pj; Pk--) {
                if (*Pk > *Pj) {
                    break;
                }
            }
            /* 对换Pj，Pk */
            swap(Pj, Pk);
            /* 再将Pj之后的元素重新排列 */
            reverse(Pj + 1, ss.end());
            cout << ss << endl;
        }
    }
    return 0;
}

```



