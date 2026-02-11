---
title: 二分查找-统一框架
mathjax: true
toc: true
categories:
  - 数据结构
date: 2021-04-05 10:18:47
tags:
- 二分法
---

前天面试的时候又考到二分查找了，但是没有写出来，之前看了labuladong的鬼模板，以为自己懂了，发现其实并不懂，这几天重新学习之后，写下了这篇文章。


<!--more-->


# 「二分」的本质是两段性


「二分」的本质是两段性，并非单调性。只要一段满足某个性质，另外一段不满足某个性质，就可以用「二分」。---[宫水三叶](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/solution/shua-chuan-lc-yan-ge-ologn100yi-qi-kan-q-xifo/)


后序我会通过一系列的题目来表明两段性如何运用。

# 统一模板--查找下界与查找上界

举个例子，比如我有一个有序数组：
$$
[1,2,4,6,8,9]
$$

我想查找$target=4$,那么就有两种方法：


| 查找方式      | 查找下界                                                                                     | 查找上界                                                                                                     |
| ------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 区间范围      | $x\geq4$                                                                                     | $x\leq4$                                                                                                     |
| 示意图        | ![](bin-search-template/binsearch-template-1.png)                                            | ![](bin-search-template/binsearch-template-2.png)                                                            |
| mid在区间中   | ![](bin-search-template/inrange-template-1.png)<br> 找下界，r必然缩小，极端情况时等于mid     | ![](bin-search-template/inrange-template-2.png)<br>找上界，l必然增大，极端情况时等于mid                      |
| mid不在区间中 | ![](bin-search-template/notinrange-template-1.png)<br>找下界，l必然增大，极端情况时等于mid+1 | ![](bin-search-template/notinrange-template-2.png)<br>找上界，r必然缩小，极端情况时等于mid-1                 |
| 代码模板      | ![](bin-search-template/code-template-1.png)                                                 | ![](bin-search-template/code-template-2.png)                                                                 |
| 说明          | `mid                                                `是`l+r`的向下取整。                     | `mid                                                `是`l+r`的向上取整(避免[1,2]中找小于等于1的死循环情况)。 |


**注意：**
  
  - 示意图中红色部分是我们的当前要寻找的区间，对应的是函数中`inrange`函数。
  
  - 此模板`l，r`都是能取到的有效值。
  
  - 此模板不需要中途退出，不需要考虑`l`是小于还是小于等于`r`，并且永远以`l`作为返回值，大大降低心智负担。


# 刷题时间

## [x 的平方根](https://leetcode-cn.com/problems/sqrtx/)

#### 分析

返回类型是只保留整数部分，那么不就是要我们找到$mid<=\sqrt{x}$的上界吗？那么我们**模板二走起**～

#### 代码

```cpp
class Solution {
 public:
  int mySqrt(int x) {
    long l = 0, r = x, mid;  //避免溢出
    while (l < r) {
      mid = (l + r + 1) >> 1;
      if (mid <= (x / mid)) l = mid;
      else r = mid - 1;
    }
    return l;
  }
};
```


## [猜数字大小](https://leetcode-cn.com/problems/guess-number-higher-or-lower/)

#### 分析

调用`guess`函数得到不同的情况，明显的二段性：
```
0---------------pick--------------N
      mid       mid       mid
       1         0        -1
```

这题可以找上界，也可以找下界，这里我们试试找下界，**模板一走起～**

#### 代码

```cpp
class Solution {
 public:
  int guessNumber(int n) {
    long l = 0, r = n, mid;
    while (l < r) {
      mid = l + r >> 1;
      if (guess(mid) <= 0) r = mid;
      else l = mid + 1;
    }
    return l;
  }
};
```

## [第一个错误的版本](https://leetcode-cn.com/problems/first-bad-version/)

#### 分析

典型的二段性：
```
0---------------first--------------N
      mid       mid       mid
     false      true      true
```

也就是右区域找下界，**模板一走起～**

#### 代码

```cpp
class Solution {
 public:
  int firstBadVersion(int n) {
    long l = 0, r = n, mid;
    while (l < r) {
      mid = l + r >> 1;
      if (isBadVersion(mid)) r = mid;
      else l = mid + 1;
    }
    return l;
  }
};
```

## [在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

#### 分析

这题比较好，可以让我们使用两个模板，首先我们可以找到右区域的下界（模板一），然后再找到左区域的上界（模板二），让我们开始吧！

#### 代码

```cpp
class Solution {
 public:
  vector<int> searchRange(vector<int>& nums, int target) {
    if (nums.empty()) return {-1, -1};
    /* 我们首先找到大于等于target的下界（模板1） */
    int l = 0, r = nums.size() - 1, mid;
    while (l < r) {
      mid = l + r >> 1;
      if (nums[mid] >= target) r = mid;
      else l = mid + 1;
    }
    if (nums[l] != target) return {-1, -1}; // 如果没有直接退出
    /*  在新的区域内找到小于等于target的上界（模板二） */
    int start = l;
    r = nums.size() - 1;
    while (l < r) {
      mid = l + r + 1 >> 1;
      if (nums[mid] <= target) l = mid;
      else r = mid - 1;
    }
    return {start, l};
  }
};
```

## [寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

#### 分析
![](bin-search-template/寻找旋转排序数组中的最小值.png)

上图看出来我们需要找下界，接下来我们只需要对比最右侧的元素确定我们当前位于右区域还是左区域即可，然后**模板一走起～**

#### 代码

```cpp
class Solution {
 public:
  int findMin(vector<int>& nums) {
    int l = 0, r = nums.size() - 1, mid;
    while (l < r) {
      mid = l + r >> 1;
      if (nums[mid] <= nums[r]) r = mid;
      else l = mid + 1;
    }
    return nums[l];
  }
};
```

## [寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/description/)

#### 分析

这题多了一个点，就是数据会出现重复，也就是可能会出现如下情况：
```
        /--
       /
L ----/          --- R
                /
               /
            --/
```

但是其实我们贯彻一个思路，就是找小于等于`nums[r]`区域的下界，那么就是如果`nums[l]==nums[r]`的时候，我们都向上收缩就好了，那么接下来的事情就交给模板一去做就完事了！

#### 代码

```cpp
class Solution {
 public:
  int findMin(vector<int>& nums) {
    int l = 0, r = nums.size() - 1, mid;
    while (l < r) {
      if (nums[l] == nums[r]) { // 跳过相同元素
        l++;
        continue;
      }
      mid = l + r >> 1;
      if (nums[mid] <= nums[r]) r = mid;
      else l = mid + 1;
    }
    return nums[l];
  }
};
```


## [搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

#### 分析
首先找到旋转排序数组中的最小值（模板一），然后根据情况重新分配区域，最后再搜索一次搜索排序数组（模板一），就搞定收工。

#### 代码

```cpp
class Solution {
 public:
  int search(vector<int>& nums, int target) {
    int n = nums.size(), l = 0, r = n - 1, mid;
    /* 找到旋转排序数组中的最小值（模板一） */
    while (l < r) {
      mid = l + r >> 1;
      if (nums[mid] <= nums[r]) r = mid;
      else l = mid + 1;
    }
    /* 根据情况重新分配区域 */
    if (target <= nums[n - 1]) r = n - 1;
    else r = l - 1, l = 0;
    /* 搜索排序数组（模板一） */
    while (l < r) {
      mid = l + r >> 1;
      if (nums[mid] >= target) r = mid;
      else l = mid + 1;
    }
    return nums[l] == target ? l : -1;
  }
};
```

## [寻找峰值](https://leetcode-cn.com/problems/find-peak-element/description/)

#### 分析

```    
        /\
   /\  /  \
\ /  \/    \
            \
```
这题看着复杂，其实挺简单，他的两段性在数据中，我们可以找：
1.  上升区域的上界，上升区域必然有`nums[mid]>nums[mid-1]`
2.  下降区域的下界，下降区域必然有`nums[mid]>nums[mid+1]`

当然还需要考虑`mid`在两侧的情况，这里我们选第一种做法吧，**模板二走起～**

#### 代码

```cpp
class Solution {
 public:
  int findPeakElement(vector<int>& nums) {
    int n = nums.size(), l = 0, r = n - 1, mid;
    while (l < r) {
      mid = (l + r + 1) >> 1;
      if ((mid > 0 and nums[mid] > nums[mid - 1]) or
          (mid == 0 and nums[mid] > nums[mid + 1]))
        l = mid;
      else
        r = mid - 1;
    }
    return l;
  };
};
```

## [找到 K 个最接近的元素](https://leetcode-cn.com/problems/find-k-closest-elements/description/)

#### 分析

这题其实就是找到距离`target`最接近的元素，找大于等于`target`的下界与找上界差别不大，然后需要需要注意当前点和另外一个点到`target`的距离，来确定起始点，最后双指针。

#### 代码

```cpp
class Solution {
 public:
  vector<int> findClosestElements(vector<int>& arr, int k, int x) {
    // 找到小于等于x的上界，用模板二
    int l = 0, r = arr.size() - 1, mid;
    while (l < r) {
      mid = (l + r + 1) >> 1;
      if (arr[mid] <= x) l = mid;
      else r = mid - 1;
    }
    // 找到分界点之后要判断一下左边还是右边
    if ((l < arr.size() - 1) and (abs(arr[l] - x) > abs(arr[l + 1] - x))) {
      l = r = (l + 1);
    }
    // 双指针收尾
    k--;
    while (k--) {
      if (l == 0) {
        r++;
      } else if (r == arr.size() - 1) {
        l--;
      } else {
        int ldiff = abs(arr[l - 1] - x), rdiff = abs(arr[r + 1] - x);
        if (ldiff <= rdiff) {
          l--;
        } else if (ldiff > rdiff) {
          r++;
        }
      }
    }
    return vector<int>(arr.begin() + l, arr.begin() + r + 1);
  }
};
```


## [搜索长度未知的有序数组](https://leetcode-cn.com/problems/valid-perfect-square/description/)

#### 分析

这题和`x 的平方根`类似，那题我们用模板二找的上界，这题我们用模板一找下界。

#### 代码

```cpp
class Solution {
 public:
  bool isPerfectSquare(int num) {
    long l = 1, r = num, mid;
    while (l < r) {
      mid = l + r >> 1;
      if (mid >= num / mid) r = mid;
      else l = mid + 1;
    }
    return (l * l == num) ? true : false;
  }
};
```

## [寻找比目标字母大的最小字母](https://leetcode-cn.com/problems/find-smallest-letter-greater-than-target/description/)

#### 分析

典型的找下界题目，并且这里的范围是必须比`target`大，模板一用起来！

#### 代码

```cpp
class Solution {
 public:
  char nextGreatestLetter(vector<char>& letters, char target) {
    int l = 0, r = letters.size() - 1, mid;
    while (l < r) {
      mid = l + r >> 1;
      if (letters[mid] > target) r = mid;
      else l = mid + 1;
    }
    return letters[l] > target ? letters[l] : letters[(l + 1) % letters.size()];
  }
};
```


## [两数之和 II - 输入有序数组](https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/description/)

#### 分析

写多了，都麻了，这题简单循环加二分找下界即可。

#### 代码

```cpp
class Solution {
 public:
  vector<int> twoSum(vector<int>& numbers, int target) {
    int l = 0, r = numbers.size() - 1, mid;
    int start, end;
    for (int i = 0; i < numbers.size() - 1; i++) {
      start = i, l = i + 1, r = numbers.size() - 1;
      while (l < r) {
        mid = l + r >> 1;
        if (numbers[mid] >= (target - numbers[i])) r = mid;
        else l = mid + 1;
      }
      end = l;
      if (numbers[end] == (target - numbers[start])) { break; }
    }
    return {start + 1, end + 1};
  }
};
```


## [找出第 k 小的距离对](https://leetcode-cn.com/problems/find-k-th-smallest-pair-distance/description/)

#### 分析

第k的最小距离，讲道理得用堆做，不过这题用二分也是可以做的，其实第k小的距离对，表明了当前数组中比如存在着$n-k\geq 0$个距离对，他们的距离大于我们需要找的那一对。那么这题我们就是在$n-k\geq 0$这个区间中找下界了，**用模板一**。

当然难的二分题难点都不是二分，这题的困难在于如何统计有多少对距离对大于`mid`，并且我们找下界的方式可以提前退出。

#### 代码

```cpp
class Solution {
 public:
  int smallestDistancePair(vector<int>& nums, int k) {
    sort(nums.begin(), nums.end());
    int l = 0, r = ((*nums.rbegin()) - (*nums.begin())), mid;
    while (l < r) {
      mid = l + r >> 1;
      if (has_K_pair(nums, mid, k)) r = mid;
      else l = mid + 1;
    }
    return l;
  }
  bool has_K_pair(vector<int>& nums, int mid, int k) {
    int cnt = 0, l = 0;
    // 用双指针的方式检测当前有多少个pair的差值大于mid
    for (int r = 0; r < nums.size(); r++) {
      while (nums[r] - nums[l] > mid) { l++; }
      cnt += (r - l);
      if (cnt >= k) return true; 
    }
    return false;
  }
};
```


## 总结

1.  感谢y总的[视频](https://www.bilibili.com/video/BV1Ft41157zW),以及宫水三叶小姐姐的一些题解～
2.  本文中的题目都选自`二分查找`的[leetbook](https://leetcode-cn.com/leetbook/detail/binary-search/)。
3.  大部分简单的二分题用模板就能轻松解决。
4.  困难的二分题大多都很难看出二段性，需要加强锻炼。