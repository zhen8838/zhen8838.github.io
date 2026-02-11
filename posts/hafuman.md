---
title: 哈夫曼树
date: 2018-08-23 11:41:53
mathjax: true
tags:
-   树
categories:
-   数据结构
---


其实早就应该写完这个哈夫曼树，只不过最近有点没有心情学习。对于哈夫曼树的构造，我总结了以下几步：

<!--more-->


1.  取出元素构建子树
2.  子树入队
3.  构建新子树
4.  将队列中的子树插入新子树，并入队
5.  循环至最后一个子树
6.  将队列中所有元素插入最后子树


# 程序

## main.cpp

```cpp
#include "hafuman.cpp"
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <queue>
using namespace std;

/* 运算符重载  */
struct mycmp {
    bool operator()(int a, int b) { //通过传入不同类型来定义不同类型优先级
        return a > b;               //最小值优先
    }
};

int TestNum[] = {5, 4, 3, 2, 1};

int main(int argc, char const *argv[]) {
    /* 最小堆用于存放数据 */
    priority_queue<int, vector<int>, mycmp> MyHeap;
    /* 队列于存放临时节点 */
    queue<HAFUNode_t<int>> TempQueue;
    /* 哈夫曼树根节点 */
    HAFUNode_t<int> HaFuTree = nullptr;
    /* 创建临时变量 */
    int tright, tleft;
    /* 赋值 */
    for (int i = 0; i < 5; i++) {
        MyHeap.push(TestNum[i]);
    }

    printf("测试哈夫曼\n");
    while (!MyHeap.empty()) {               /* 优先队列非空 */
        tleft = MyHeap.top(), MyHeap.pop(); /* 输出左值并释放 */
        if (MyHeap.empty()) {               /* 判断是否为空 */
            break;
        }
        tright = MyHeap.top(), MyHeap.pop(); /* 输出右值并释放 */
        /* 构造子树 */
        HaFuTree = CreatSubTree(tleft, tright, tleft + tright);
        if (MyHeap.empty()) { /* 若为最后一个节点 */
            break;
        } else {
            AutoJoin(HaFuTree, TempQueue);
        }
        /* 再加入元素 */
        MyHeap.push(tleft + tright);
    }
    /* 将队列中所有的子树合并成一颗 */
    while (!TempQueue.empty()) {
        FinalJoin(HaFuTree, TempQueue);
    }

    DrawTree(HaFuTree);
    return 0;
}
```



## hafuman.cpp

```cpp
#include "hafuman.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

/**
 * @brief  构造一个子树
 * @param[in]  类型T：左节点 右节点
 * @param[out]
 * @return
 **/
template <typename T> HAFUNode_t<T> CreatSubTree(T Left, T Right, T Root) {
    /* set the root  */
    HAFUTree<T> *pH = (HAFUNode_t<T>)malloc(sizeof(struct HAFUTree<T>));
    pH->data = Root;

    /* set the left child */
    pH->left = (HAFUNode_t<T>)malloc(sizeof(struct HAFUTree<T>));
    pH->left->data = Left;
    pH->left->left = nullptr;
    pH->left->right = nullptr;

    /* set the right child */
    pH->right = (HAFUNode_t<T>)malloc(sizeof(struct HAFUTree<T>));
    pH->right->data = Right;
    pH->right->left = nullptr;
    pH->right->right = nullptr;
    return pH;
}

/**
 * @brief 将一个根挂载在另一根的左侧
 **/
template <typename T>
static HAFUNode_t<T> LeftJoin(HAFUNode_t<T> LEFT, HAFUNode_t<T> ROOT) {
    free(ROOT->left);
    ROOT->left = LEFT;
    return ROOT;
}
/**
 * @brief 将一个根挂载在另一根的右侧
 **/
template <typename T>
static HAFUNode_t<T> RightJoin(HAFUNode_t<T> LEFT, HAFUNode_t<T> ROOT) {
    free(ROOT->right);
    ROOT->right = LEFT;
    return ROOT;
}

/**
 * @brief 将队中的子树插入新的子树，成功后保存至队列
 **/
template <typename T>
void AutoJoin(HAFUNode_t<T> NEW, std::queue<HAFUNode_t<T>> &Q) {
    HAFUNode_t<T> temp = nullptr;
    if (Q.empty()) {
        Q.push(NEW);
    } else { /* Queue not empty then start insert */
        // printf("NEW->data:%d Q.front()->left->data:%d\n", NEW->data,
        //        Q.front()->left->data);
        if (Q.front()->data == NEW->left->data) { /* left join */
            temp = LeftJoin(Q.front(), NEW);
            Q.pop();      /* pop the old  */
            Q.push(temp); /* push the new root */
        } else if (Q.front()->data == NEW->right->data) {
            temp = RightJoin(Q.front(), NEW);
            Q.pop();
            Q.push(temp);
        } else { /* can't insert */
            Q.push(NEW);
        }
    }
}
/**
 * @brief 最终插入，将队列中的元素插入到根中
 **/
template <typename T>
void FinalJoin(HAFUNode_t<T> ROOT, std::queue<HAFUNode_t<T>> &Q) {
    //        Q.front()->left->data);
    if (Q.front()->data == ROOT->left->data) { /* left join */
        LeftJoin(Q.front(), ROOT);
        Q.pop(); /* pop the old  */
    } else if (Q.front()->data == ROOT->right->data) {
        RightJoin(Q.front(), ROOT);
        Q.pop();
    } else { /* can't insert */
        printf("insert error!\r\n");
    }
}

/**
 * description  输出二叉树的高度
 * @param[in]   BinTree_t
 * @retval      int
 **/
template <typename T> static int FindTreeHeight(HAFUNode_t<T> BT) {
    int rightlen, leftlen, maxlen;
    if (BT) {
        rightlen = FindTreeHeight(BT->right);
        leftlen = FindTreeHeight(BT->left);
        maxlen = leftlen > rightlen ? leftlen : rightlen;
        return maxlen + 1;
    } else {
        return 0;
    }
}

/**
 * @brief 绘出树
 **/
template <typename T> void DrawTree(HAFUNode_t<T> BT) {
    int height = FindTreeHeight(BT);
    int cnt = 0;

    /* 开始层序遍历二叉树并且绘制图形 */
    std::queue<HAFUNode_t<T>> CurLevelqueue;  /*记录当前层的元素  */
    std::queue<HAFUNode_t<T>> NextLevelqueue; /*记录下一层的元素  */
    std::queue<HAFUNode_t<T>> CurTempqueue;   /*当前队列副本  */
    std::queue<HAFUNode_t<T>> NextTempqueue;  /*下一层队列副本  */
    HAFUNode_t<T> temp, lasttemp = NULL;
    int width = 0;
    CurLevelqueue.push(BT); //将头节点入队s
    /*现在修改了入队函数,空指针也可以入队
    所以在出队的时候就需要进行判断  */
    for (int i = 0; i < height; i++) {
        cnt = (int)pow(2, i);           //当前行的个数21
        width = pow(2, height - i + 1); //设置宽度为2^(height-i+1)
        CurTempqueue = CurLevelqueue;   /* 临时记录 */
        while (cnt--) {
            temp = CurLevelqueue.front();
            if (temp != NULL) {
                printf("%*d%*c", width, temp->data, width, ' ');
                NextLevelqueue.push(temp->left);
                NextLevelqueue.push(temp->right); //将下一层子节点入队
            } else {
                printf("%*c%*c", width, ' ', width, ' ');
                NextLevelqueue.push(nullptr);
                NextLevelqueue.push(nullptr); //如果此层是空，那么再入两个空指针
            }
            CurLevelqueue.pop(); //将上一层节点出队
        }
        printf("\n");
        NextTempqueue = NextLevelqueue; /* 记录下一层元素 */
        /* 接下来打印层间符号 */
        if (i != height - 1) { /* 若不是最后一行 */
            /* 先记录下一行元素宽度间隔 */
            int nextwidth = (int)pow(2, height - i);
            /* 并且以他下一层元素的个数打印'-' */
            for (int k = 0; k < (int)pow(2, i + 1); k++) {
                /* 每隔一位去打印width个'-' */
                if ((k % 2) == 0) { /* 偶数位打印 空格+`-` */
                    if (NextTempqueue.front() != nullptr) {
                        printf("%*c", nextwidth, '-');
                        for (uint8_t z = 0; z < nextwidth; z++) {
                            printf("-");
                        }
                    } else {
                        printf("%*c", nextwidth, ' ');
                        for (uint8_t z = 0; z < nextwidth; z++) {
                            printf(" ");
                        }
                    }
                    NextTempqueue.pop();
                } else { /* 奇数位打印 空格+`-` */
                    if (NextTempqueue.front() != nullptr) {
                        for (uint8_t z = 0; z < nextwidth; z++) {
                            printf("-");
                        }
                        printf("%*c", nextwidth, ' ');
                    } else {
                        for (uint8_t z = 0; z < nextwidth; z++) {
                            printf(" ");
                        }
                        printf("%*c", nextwidth, ' ');
                    }
                    NextTempqueue.pop();
                }
            }
            printf("\n");
        }
        printf("\r");
        swap(CurLevelqueue, NextLevelqueue); /* 交换队列 */
    }
}
```



## hafuman.h

```cpp
#ifndef _HAFUMAN_H
#define _HAFUMAN_H

#include <queue>
/**
 * @brief the defination
 **/
template <typename T> struct HAFUTree;
/**
 * @brief 别名
 **/
template <typename T> using HAFUNode_t = struct HAFUTree<T> *;
/**
 * @brief 结构体模板类
 **/
template <typename T> struct HAFUTree {
    T data;
    HAFUNode_t<T> left;
    HAFUNode_t<T> right;
};


#endif
```



## makefile
```makefile
CC = clang++
CFLAGS = -g # debug
CLIBS = #-lpthread
 
INCLUDE = $(wildcard ./*.h) # INCLUDE = a.h b.h ... can't be defined like "INCLUDE = ./*.h"
SOURCES = $(wildcard ./*.cpp)
 
TARGET = hafutree
OBJECTS = $(patsubst %.cpp,%.o,$(SOURCES))
 
$(TARGET) : $(OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)

$(OBJECTS) : %.o : %.cpp
	$(CC) -c $(CFLAGS) $< -o $@
 
.PHONY : clean
clean:
	rm -rf *.o $(TARGET)  

```


# 编译运行
```sh
➜  hafuman make && ./hafutree
clang++ -c -g  hafuman.cpp -o hafuman.o
clang++ -c -g  main.cpp -o main.o
clang++ -g  hafuman.o main.o -o hafutree
测试哈夫曼
                              15
               ---------------------------------
               6                               9
       -----------------               -----------------
       3               3               4               5
   ---------
   1       2
```