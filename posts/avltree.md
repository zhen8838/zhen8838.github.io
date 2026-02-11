---
title: 平衡二叉树   
date: 2018-07-31 20:57:45
mathjax: true
tags:
-   树
categories:
-   数据结构
---



最后好久没写数据结构了，今天我把之前写的函数都写成C++的了。舒服的用一波C++中的queue和stack。
废话少说直接上代码（这次又调整了打印二叉树的程序 美滋滋）

<!--more-->


# main.cpp


```c
#include "bantree.h"
#include <iostream>

using namespace std;

void UseageHelp(void) {
    cout << "\n \
    AVL树的操作\n \
    (1).创建新的AVL树;\n \
    (2).插入元素;\n \
    (3).删除元素;\n \
    (4).寻找最大元素;\n \
    (5).寻找最小元素;\n \
    (6).绘制二叉树;\n \
    (h).帮助;\n \
    (q).quit;\n \
    " << endl;
}

int main(int argc, char const *argv[]) {
    ElementType i = 0;
    AvlTree pTree = NULL;
    AvlTree tmp = NULL;
    UseageHelp();
    while (1) {
        switch (getchar()) {
        case '1':
            pTree = MakeEmpty(pTree);
            cout << "创建成功 Tree:" << pTree << endl;
            break;
        case '2':
            cout << "请输入插入元素:";
            cin >> i;
            pTree = Insert(i, pTree);
            cout << endl;
            break;
        case '3':
            cout << "请输入删除元素:";
            cin >> i;
            Delete(i, pTree);
            cout << endl;
            break;
        case '4':
            tmp = FindMax(pTree);
            if (tmp != nullptr) {
                cout << "最大元素：" << tmp->Element << endl;
            }
            cout << endl;
            break;
        case '5':
            tmp = FindMin(pTree);
            if (tmp != nullptr) {
                cout << "最小元素：" << tmp->Element << endl;
            }
            cout << endl;
            break;
        case '6':
            DrawTree(pTree);
            cout << endl;
            break;
        case 'h':
            UseageHelp();
            break;
        case 'q':
            exit(0);
            break;
        default:
            break;
        }
    }
    return 0;
}
```



# bantree.h


```c
#ifndef _BANTREE_H
#define _BANTREE_H

#define ElementType int

struct AvlNode;
typedef struct AvlNode *Position;
typedef struct AvlNode *AvlTree;

AvlTree MakeEmpty(AvlTree T);
void DrawTree(AvlTree BT);
Position Find(ElementType X, AvlTree T);
Position FindMin(AvlTree T);
Position FindMax(AvlTree T);
AvlTree Insert(ElementType X, AvlTree T);
AvlTree Delete(ElementType X, AvlTree T);
ElementType Retrieve(Position P);
Position SingleRotateWithLeft(Position K2);
Position SingleRotateWithRight(Position K2);
Position DoubleRotateWithLeft(Position K3);
Position DoubleRotateWithRight(Position K3);


struct AvlNode {
    ElementType Element;
    AvlTree Left;
    AvlTree Right;
    int Height;
};

#endif
```



# bantree.cpp


```c
#include "bantree.h"
#include <cmath>
#include <iostream>
#include <queue>
#include <stack>
#include <string>
using namespace std;

AvlTree MakeEmpty(AvlTree T) {

    if (T != nullptr) {
        MakeEmpty(T->Left);
        MakeEmpty(T->Right);
        delete T;
    }
    return NULL;
}

static int Height(Position P) {
    if (P == nullptr) {
        return -1;
    } else {
        return P->Height;
    }
}

/**
 * description  输出二叉树的高度
 * @param[in]   BinTree_t
 * @retval      int
 **/
static int FindTreeHeight(AvlTree BT) {
    int rightlen, leftlen, maxlen;
    if (BT) {
        rightlen = FindTreeHeight(BT->Right);
        leftlen = FindTreeHeight(BT->Left);
        maxlen = leftlen > rightlen ? leftlen : rightlen;
        return maxlen + 1;
    } else {
        return 0;
    }
}

/**
 * @brief  用于清空队列元素
 * @param[in]
 * @param[out]
 * @return
 **/
static void ClearQueue(queue<AvlTree> &q) {
    queue<AvlTree> empty;
    swap(empty, q);
}

/**
 * @brief  简单的绘制二叉树图像
 * @param[in]
 * @param[out]
 * @return
 **/
void DrawTree(AvlTree BT) {
    int height = FindTreeHeight(BT);
    int cnt = 0;

    /* 开始层序遍历二叉树并且绘制图形 */
    queue<AvlTree> CurLevelqueue;  /*记录当前层的元素  */
    queue<AvlTree> NextLevelqueue; /*记录下一层的元素  */
    queue<AvlTree> CurTempqueue;   /*当前队列副本  */
    queue<AvlTree> NextTempqueue;  /*下一层队列副本  */
    AvlTree temp, lasttemp = NULL;
    int width = 0;
    CurLevelqueue.push(BT); //将头节点入队

    /*现在修改了入队函数,空指针也可以入队
    所以在出队的时候就需要进行判断  */
    for (int i = 0; i < height; i++) {
        cnt = (int)pow(2, i);           //当前行的个数21
        width = pow(2, height - i + 1); //设置宽度为2^(height-i+1)
        CurTempqueue = CurLevelqueue;   /* 临时记录 */
        while (cnt--) {
            temp = CurLevelqueue.front();
            if (temp != NULL) {
                printf("%*d%*c", width, temp->Element, width, ' ');
                NextLevelqueue.push(temp->Left);
                NextLevelqueue.push(temp->Right); //将下一层子节点入队
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
            // /* 按当前层元素位置打印'+' */
            // for (int J = 0; J < (int)pow(2, i); J++) {
            //     /* 如果不是最后一行且为null指针 不打印 */
            //     if (CurTempqueue.front() != nullptr) {
            //         printf("%*c%*c", width, '+', width, ' ');
            //     } else {
            //         printf("%*c%*c", width, ' ', width, ' ');
            //     }
            //     CurTempqueue.pop();
            // }
            // printf("\n");
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

Position Find(ElementType X, AvlTree T) {

    if (T == nullptr) {
        return nullptr;
    }

    if (X < T->Element) {
        return Find(X, T->Left);
    } else if (X > T->Element) {
        return Find(X, T->Right);
    } else {
        return T;
    }
}
Position FindMin(AvlTree T) {
    if (T == nullptr) {
        return nullptr;
    }
    if (T->Left != nullptr) {
        return FindMin(T->Left);
    } else {
        return T;
    }
}
Position FindMax(AvlTree T) {
    if (T == nullptr) {
        return nullptr;
    }
    if (T->Right != nullptr) {
        return FindMin(T->Right);
    } else {
        return T;
    }
}
/**
 * @brief  平衡树的插入，需要记录每个节点的高度,并实现旋转
 * @param[in]
 * @param[out]
 * @return
 **/
AvlTree Insert(ElementType X, AvlTree T) {

    if (T == nullptr) {
        T = new AvlNode;
        if (T == nullptr) {
            std::cout << "no mem!" << std::endl;
        } else {
            T->Element = X, T->Height = 0;
            T->Left = T->Right = nullptr;
        }
    } else if (X < T->Element) {
        T->Left = Insert(X, T->Left);
        if ((Height(T->Left) - Height(T->Right)) == 2) {
            if (X < T->Left->Element) {
                /* new node in left's left */
                T = SingleRotateWithLeft(T);
            } else {
                /* new node in left's right */
                T = DoubleRotateWithLeft(T);
            }
        }
    } else if (X > T->Element) {
        T->Right = Insert(X, T->Right);
        if ((Height(T->Right) - Height(T->Left)) == 2) {
            if (X > T->Right->Element) {
                /* new node in right's right */
                T = SingleRotateWithRight(T);
            } else {
                /* new node in right's left */
                T = DoubleRotateWithRight(T);
            }
        }
    }
    T->Height = max(Height(T->Left), Height(T->Right)) + 1;
    return T;
}

AvlTree Delete(ElementType X, AvlTree T) {
    Position tmpCell;
    if (T == nullptr) {
        std::cout << "error" << std::endl;
        return nullptr;
    }

    else if (X < T->Element) {
        T->Left = Delete(X, T->Left);
    } else if (X > T->Element) {
        T->Right = Delete(X, T->Right);
    }

    else if (T->Left && T->Right) {
        tmpCell = FindMin(T->Right);
        T->Element = tmpCell->Element;
        T->Right = Delete(tmpCell->Element, T->Right);
    }

    else {
        tmpCell = T;
        if (T->Left == nullptr) {
            T = T->Right;
        } else if (T->Right == nullptr) {
            T = T->Right;
        }
        delete tmpCell;
    }
}
ElementType Retrieve(Position P) {}

/**
* @brief  单左旋转
    当“麻烦节点”在“被发现者”的左子树的左边时，被称为LL插入
    需要对 '被发现者' RL旋转（左单旋）
**/
Position SingleRotateWithLeft(Position K2) {
    Position K1;
    K1 = K2->Left;
    K2->Left = K1->Right;
    K1->Right = K2;
    K2->Height = max(Height(K2->Left), Height(K2->Right)) + 1;
    K1->Height = max(Height(K1->Left), K2->Height) + 1;
    return K1; /* 新的根 */
}
/**
* @brief  单右旋转
    当“麻烦节点”在“被发现者”的右子树的右边时，被称为RR插入
    需要对 '被发现者' RR旋转（右单旋）
**/
Position SingleRotateWithRight(Position K2) {
    Position K1;
    K1 = K2->Right;
    K2->Right = K1->Left;
    K1->Left = K2;
    K2->Height = max(Height(K2->Left), Height(K2->Right)) + 1;
    K1->Height = max(Height(K1->Left), K2->Height) + 1;
    return K1; /* 新的根 */
}

Position DoubleRotateWithLeft(Position K3) {
    /* first rotate the k3'left */
    K3->Left = SingleRotateWithRight(K3->Left);
    return SingleRotateWithLeft(K3);
}

Position DoubleRotateWithRight(Position K3) {
    /* first rotate the k3'right */
    K3->Right = SingleRotateWithLeft(K3->Right);
    return SingleRotateWithRight(K3);
}
```



# Makefile

```makefile
CC = g++
CFLAGS = #-g 
CLIBS = #-lpthread
 
INCLUDE = $(wildcard ./*.h) # INCLUDE = a.h b.h ... can't be defined like "INCLUDE = ./*.h"
SOURCES = $(wildcard ./*.cpp)
 
TARGET = avltree
OBJECTS = $(patsubst %.cpp,%.o,$(SOURCES))
 
$(TARGET) : $(OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)
	rm -rf *.o

$(OBJECTS) : %.o : %.cpp
	$(CC) -c $(CFLAGS) $< -o $@
 
.PHONY : clean
clean:
	rm -rf *.o $(TARGET)  
```

# 测试命令

```sh
➜  balanceTree cat test.txt
1
2
32
2
33
2
31
2
36
2
38
2
20
2
24
2
94
2
58
2
12
6
q
➜  balanceTree make && cat test.txt | ./avltree
g++ -c  main.cpp -o main.o
g++ -c  bantree.cpp -o bantree.o
g++  main.o bantree.o -o avltree
rm -rf *.o

     AVL树的操作
     (1).创建新的AVL树;
     (2).插入元素;
     (3).删除元素;
     (4).寻找最大元素;
     (5).寻找最小元素;
     (6).绘制二叉树;
     (h).帮助;
     (q).quit;

创建成功 Tree:0
请输入插入元素:
请输入插入元素:
请输入插入元素:
请输入插入元素:
请输入插入元素:
请输入插入元素:
请输入插入元素:
请输入插入元素:
请输入插入元素:
请输入插入元素:
                              32
               ---------------------------------
              24                              36
       -----------------               -----------------
      20              31              33              58
   -----                                           ---------
  12                                              38      94

```
