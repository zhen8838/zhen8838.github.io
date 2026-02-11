---
title: 堆
date: 2018-08-14 21:20:57
mathjax: true
tags:
-   堆栈
categories:
-   数据结构
---

堆其实是也是一种二叉树，不过他的排序方式更加舒服，对于需要升序或者降序排列的数据非常有用。

<!--more-->


# 代码

## binheap.cpp

```c
#include "binheap.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
/**
 * @brief 优先队列-堆的数组实现
 **/
PriorityQueue Initialize(int MaxElements) {
    PriorityQueue H = nullptr;
    if (MaxElements < 0) {
        return nullptr;
    }
    /* 为队列结构体申请内存 */
    H = (PriorityQueue)malloc(sizeof(struct HeapStruct));

    /* 为队列数组申请内存 */
    H->Elements =
        (ElementType *)malloc((MaxElements + 1) * sizeof(ElementType));

    H->Capacity = MaxElements;
    H->Size = 0;
    H->Elements[0] = 0; /* scout element */
    return H;
}
void Destroy(PriorityQueue H) {}
void MakeEmpty(PriorityQueue H) {}
/**
 * @brief 插入的元素都会塞满数组
 **/
void Insert(ElementType X, PriorityQueue H) {
    int i;
    if (IsFull(H)) {
        return;
    }
    /*--->1.指向数组的下一个元素
    |   | 2.判断当前元素是否小于其母节点
    |   | 3.小于:将母节点元素移动到此处 大于:X赋值到此处
    ----v 4.指向母节点的位置 */
    for (i = ++H->Size; H->Elements[i / 2] > X; i /= 2) {
        H->Elements[i] = H->Elements[i / 2];
    }
    H->Elements[i] = X;
}
/**
 * @brief 只需将较小的元素上移即可
 **/
ElementType DeleteMin(PriorityQueue H) {
    int i, Child;
    ElementType MinElement, LastElement;
    if (IsEmpty(H)) {
        return H->Elements[0];
    }
    /* 最小元素 */
    MinElement = H->Elements[1];
    /* 倒数第二个元素 */
    LastElement = H->Elements[H->Size--];

    for (i = 1; i * 2 < H->Size; i = Child) {
        /* 子节点位置为当前节点的2倍 */
        Child = i * 2;
        /* 选择左右子节点中小的那个 */
        if (Child != H->Size && (H->Elements[Child] > H->Elements[Child + 1])) {
            Child++;
        }
        /* 没有到倒数第二个元素的右边 */
        if (LastElement > H->Elements[Child]) {
            /* 上浮 */
            H->Elements[i] = H->Elements[Child];
        } else {
            break;
        }
    }
    /* 交换最后一个元素 */
    H->Elements[i] = LastElement;
    /* 返回删去的元素 */
    return MinElement;
}
ElementType FindMin(PriorityQueue H) {
    if (IsEmpty(H)) {
        return 0;
    } else {
        return H->Elements[1];
    }
}
int IsEmpty(PriorityQueue H) {

    if (H->Size == 0) {
        return true;
    } else {
        return false;
    }
}
int IsFull(PriorityQueue H) {
    if (H->Size == H->Capacity) {
        return true;
    } else {
        return false;
    }
}

void Traversal(PriorityQueue H) {

    if (IsEmpty(H)) {
        printf("Empty\n");
    } else {
        for (int i = 1; i <= H->Size; i++) {
            printf("%d ", H->Elements[i]);
        }
        printf("\n");
    }
}
```



## binheap.h


```c
#ifndef _BinHeap_H
#define _BinHeap_H

#define ElementType int

struct HeapStruct;
typedef struct HeapStruct *PriorityQueue;

PriorityQueue Initialize(int MaxElements);
void Destroy(PriorityQueue H);
void MakeEmpty(PriorityQueue H);
void Insert(ElementType X, PriorityQueue H);
ElementType DeleteMin(PriorityQueue H);
ElementType FindMin(PriorityQueue H);
int IsEmpty(PriorityQueue H);
int IsFull(PriorityQueue H);
void Traversal(PriorityQueue H);
struct HeapStruct {
    int Capacity;
    int Size;
    ElementType *Elements;
};

#endif
```



## main.cpp

```c
#include "binheap.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

void printUseage(void) {
    printf("\
    二叉堆操作：\n\
    (1):初始化\n\
    (2):插入元素\n\
    (3):删除最小值\n\
    (4):遍历元素\n\
    (q):退出\n\
    ");
}

int main(int argc, char const *argv[]) {
    printUseage();
    PriorityQueue heap;
    ElementType tempint;
    while (true) {
        switch (getchar()) {
        case '1':
            printf("初始化大小为10的堆");
            heap = Initialize(10);
            printf("\n");
            break;
        case '2':
            printf("请输入插入元素:");
            scanf("%d", &tempint);
            Insert(tempint, heap);
            printf("\n");
            break;
        case '3':
            printf("删除:%d", DeleteMin(heap));
            printf("\n");
            break;
        case '4':
            printf("遍历堆:");
            Traversal(heap);
            printf("\n");
            break;
        case 'q':
            exit(EXIT_SUCCESS);
            break;
        default:
            break;
        }
    }

    return 0;
}
```



## Makefile
```makefile
CC = g++
CFLAGS = #-g 
CLIBS = #-lpthread
 
INCLUDE = $(wildcard ./*.h) # INCLUDE = a.h b.h ... can't be defined like "INCLUDE = ./*.h"
SOURCES = $(wildcard ./*.cpp)
 
TARGET = BinHeap
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

## test

```sh
1
2
10
2
8
2
7
2
5
2
3
2
11
2
13
4
3
4
3
4
3
4
q
```


# 编译运行

```sh
➜  heap make && cat test | ./BinHeap
g++ -c  binheap.cpp -o binheap.o
g++ -c  main.cpp -o main.o
g++  binheap.o main.o -o BinHeap
rm -rf *.o
    二叉堆操作：
    (1):初始化
    (2):插入元素
    (3):删除最小值
    (4):遍历元素
    (q):退出
    初始化大小为10的堆
请输入插入元素:
请输入插入元素:
请输入插入元素:
请输入插入元素:
请输入插入元素:
请输入插入元素:
请输入插入元素:
遍历堆:3 5 8 10 7 11 13

删除:3
遍历堆:5 7 8 10 13 11

删除:5
遍历堆:7 10 8 11 13

删除:7
遍历堆:8 10 13 11

```
