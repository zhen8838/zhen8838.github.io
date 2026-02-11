---
title: 二叉树
date: 2018-07-10 16:56:15
mathjax: true
tags:
-   树
categories:
-   数据结构
---


今天写的一波二叉树的操作。。感觉自己还得多多练习啊！
依旧直接上代码了，8点了，得回去休息了T_T。

<!--more-->


# 程序


```c
/*
 * @Author: Zheng Qihang
 * @Date: 2018-07-10 09:38:39
 * @Last Modified by: Zheng Qihang
 * @Last Modified time: 2018-11-08 16:36:09
 */
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
        data
       /    \
      V      V
    Left    Right
*/
typedef int ElementType;
typedef struct TreeNode {
    ElementType Data;
    struct TreeNode *Left;
    struct TreeNode *Right;
} * BinTree;

/************************* Queue define! **************************/
#define ElementQueueType BinTree
typedef struct QueueNode {
    ElementQueueType data;
    struct QueueNode *next;
} * QNode;

typedef struct QueueHeader {
    QNode front;
    QNode rear;
} * Queue;

/**
 * description  Create the Queue!
 * @param[in]   void
 * @retval      no
 **/
Queue CreateQueue(void) {
    Queue Q = (Queue)malloc(sizeof(struct QueueHeader));
    Q->front = NULL;
    Q->rear = NULL;
    return Q;
}

/**
 * description  Add Queue Node !
 * @param[in]   Queue header and a item
 * @retval      void
 **/
void AddQ(Queue Q, ElementQueueType item) {
    if (!Q) {
        return;
    }
    if (!item) {
        return;
    }
    QNode temp = (QNode)malloc(sizeof(struct QueueNode));
    temp->data = item;
    temp->next = NULL;
    // when the Queue is null
    if (!Q->front && !Q->rear) {
        Q->front = temp;
        Q->rear = temp;
    } else {
        Q->rear->next = temp;
        Q->rear = temp;
    }
}

/**
 * description  check the queue is empty
 * @param[in]   queue
 * @retval      int
 **/
int IsEmptyQ(Queue Q) { return !Q->front; }

/**
 * description  Delete the Queue Node
 * @param[in]   Queue header
 * @retval      data
 **/
ElementQueueType DeleteQ(Queue Q) {
    if (!Q) {
        return NULL;
    }
    if (IsEmptyQ(Q)) {
        return NULL;
    }
    QNode temp = Q->front;
    ElementQueueType tempdata;
    if (Q->front == Q->rear) {
        Q->front = NULL;
        Q->rear = NULL;
    } else {
        Q->front = Q->front->next;
    }
    tempdata = temp->data;
    free(temp);
    return tempdata;
}

void PrintQueue(Queue Q) {

    if (IsEmptyQ(Q)) {
        return;
    }
    printf("打印队列数据元素：\n");
    QNode temp = Q->front;

    for (; temp; temp = temp->next) {
        printf("0x%3lX   ", (unsigned long int)temp->data & 0xFFF);
    }
    printf("\n");
}

/************************* Queue define! **************************/

/************************* Stack define! **************************/

#define ElementStackType BinTree
typedef struct StackNode // node defination
{
    ElementStackType data;
    struct StackNode *next;
} * Stack; // Node is a pointer to _Node

int IsEmpty(Stack S) { return !S->next; }

Stack CreateStack(void) {
    Stack S = (Stack)malloc(sizeof(struct StackNode));
    S->next = NULL;
    return S;
}

ElementStackType Pop(Stack S) {
    ElementStackType temp;
    Stack tmepNode = S->next;
    if (tmepNode != NULL) {
        temp = tmepNode->data;
        S->next = tmepNode->next;
        free(tmepNode);
        return temp;
    } else {
        return NULL;
    }
}

void MakeEmpty(Stack S) {
    while (S->next != NULL) {
        Pop(S);
    }
}
void DisposeStack(Stack S) {
    MakeEmpty(S);
    free(S);
}

void Push(Stack S, ElementStackType X) {
    Stack tempNode = S->next;
    Stack newNode = (Stack)malloc(sizeof(struct StackNode));
    newNode->data = X;
    // printf("S->next %p\n",S->next);
    if (tempNode != NULL) {
        newNode->next = tempNode;
        // printf("%p = %p\n",newNode->next,tempNode);
    } else {
        newNode->next = NULL;
    }
    S->next = newNode;
    // printf("S->next %p\n",S->next);
}

ElementStackType Top(Stack S) {
    if (S->next != NULL) {
        return S->next->data;
    } else {
        return NULL;
    }
}

void PrintStack(Stack S) {
    Stack tep = S->next;
    while (tep != NULL) {
        printf("addr=0x%3lX  dataadd=0x%3lX  nextaddr=0x%3lX\n",
               (unsigned long int)tep & 0xFFF,
               (unsigned long int)tep->data & 0xFFF,
               (unsigned long int)tep->next & 0xFFF);
        tep = tep->next;
    }
}

/************************* Stack define! **************************/

/**
* description  first root then left
                child tree then right child tree
* @param[in]   BinTree
* @retval      void
**/
void PreOrderTraversal(BinTree BT) {
    if (BT) {
        printf("%d   ", BT->Data);
        PreOrderTraversal(BT->Left); // Recursion traversal
        PreOrderTraversal(BT->Right);
    }
}
/**
 * description  traversal the leaves
 * @param[in]   BinTree
 * @retval      void
 **/
void PreOrderPrintLeaves(BinTree BT) {
    if (BT) {

        if (!BT->Left && !BT->Left) {
            printf("%d   ", BT->Data);
        }
        PreOrderPrintLeaves(BT->Left); // Recursion traversal
        PreOrderPrintLeaves(BT->Right);
    }
}

/**
 * description  find the Binary tree alone length
 *              Tag==0 find the right
 *              Tag!=0 find the left
 * @param[in]   bintree
 * @retval      height
 **/
int FindTreeLength(BinTree BT, int tag) {
    if (BT) {

        if (tag) {
            return 1 + FindTreeLength(BT->Left, tag);
        } else {
            return 1 + FindTreeLength(BT->Right, tag);
        }
    } else {
        return 0;
    }
}
/*

// /**
//  * description  find the Binary tree height
//  *              Tag==0 find the right
//  *              Tag!=0 find the left
//  * @param[in]   bintree
//  * @retval      height
//  **/
// int FindTreeHeight(BinTree BT) {
//     int rightlen = FindTreeLength(BT, 0);
//     int leftlen = FindTreeLength(BT, 1);
//     return rightlen > leftlen ? rightlen : leftlen;
// }
/* the better implement */
int FindTreeHeight(BinTree BT) {
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
* description  first left child tree then
            root then  then right child tree
* @param[in]   BinTree
* @retval      void
**/
void InOrderTraversal(BinTree BT) {
    if (BT) {
        PreOrderTraversal(BT->Left); // Recursion traversal
        printf("%d   ", BT->Data);
        PreOrderTraversal(BT->Right);
    }
}
/**
* description  first left child tree then
             right child tree then root
* @param[in]   BinTree
* @retval      void
**/
void PostOrderTraversal(BinTree BT) {
    if (BT) {
        PreOrderTraversal(BT->Left); // Recursion traversal
        PreOrderTraversal(BT->Right);
        printf("%d   ", BT->Data);
    }
}

/*
* description first root then left
                child tree then right child tree
* @param[in]   BinTree
* @retval      void
**/
void PreStackTraversal(BinTree BT) {
    BinTree T = BT;
    Stack S = (Stack)malloc(sizeof(struct StackNode));
    // T!=null or stack is not empty
    while (T || !IsEmpty(S)) {
        while (T) { // push the left child tree in stack
            Push(S, T);
            printf("%d   ", T->Data);
            T = T->Left;
        }
        if (!IsEmpty(S)) {
            T = Pop(S);
            T = T->Right; // to the
        }
    }
    DisposeStack(S);
}
/*
* description  first left child tree then
            root then  then right child tree
* @param[in]   BinTree
* @retval      void
**/
void InStackTraversal(BinTree BT) {
    BinTree T = BT;
    Stack S = (Stack)malloc(sizeof(struct StackNode));
    // T!=null or stack is not empty
    while (T || !IsEmpty(S)) {
        while (T) { // push the left child tree in stack
            Push(S, T);
            T = T->Left;
        }
        if (!IsEmpty(S)) {
            T = Pop(S);
            printf("%d   ", T->Data);
            T = T->Right;
        }
    }
    DisposeStack(S);
}
/*
* description  first left child tree then
             right child tree then  root
* @param[in]   BinTree
* @retval      void
**/
void PostStackTraversal(BinTree BT) {
    BinTree T = BT;
    Stack S = (Stack)malloc(sizeof(struct StackNode));
    // T!=null or stack is not empty
    while (T || !IsEmpty(S)) {
        while (T) { // push the left child tree in S
            Push(S, T);
            T = T->Left;
        }
        if (!IsEmpty(S)) {
            T = Pop(S);
            if (T != BT) { // root don't print
                printf("%d   ", T->Data);
            }
            T = T->Right;
        }
    }
    printf("%d   ", BT->Data); // print root
    DisposeStack(S);
}

/* sequence traversal
    Use queue
    ----------------
            A           =>
    ----------------
    ----------------
        B       C       =>
    ----------------
    ----------------
    D     F   G    I    =>
    ----------------
    ----------------
        E       H       =>
    ----------------
*/
void LevelOrderTraversal(BinTree BT) {
    Queue myqueue = CreateQueue();
    BinTree T = BT;
    BinTree temp = NULL;
    AddQ(myqueue, T); // header add in queue
    while (!IsEmptyQ(myqueue)) {
        temp = DeleteQ(myqueue);
        printf("%d   ", temp->Data);
        AddQ(myqueue, temp->Left);
        AddQ(myqueue, temp->Right);
    }
}

/**
 * description  create the binary tree node
 * @param[in]   data
 * @retval      bintree
 **/
BinTree CreateTreeNode(int dat) {
    BinTree new = (BinTree)malloc(sizeof(struct TreeNode));
    new->Data = dat;
    new->Left = NULL;
    new->Right = NULL;
    return new;
}
void ConnectNode(BinTree root, BinTree left, BinTree right) {

    if (!root) {
        printf("error in null\n");
        return;
    }
    root->Left = left;
    root->Right = right;
}

/* create bin tree like this:
                  1                           A
            /          \                /          \
           2            3              B            C
         /   \        /   \          /   \        /   \
       4      5      6     7       D      E      F     G
      /        \    / \           /        \    / \
     8          9  10  11        H          I  J   K
 */
BinTree CreateBinTree(void) {
    printf("Create Binary Tree like this:\n \
                 1                           A\n \
            /          \\                /          \\ \n \
           2            3              B            C \n  \
        /   \\        /   \\          /   \\        /   \\ \n \
       4      5      6     7       D      E      F     G \n \
      /        \\    / \\           /        \\    / \\ \n \
     8          9  10  11        H          I  J   K \n \
    ");
    BinTree A = CreateTreeNode(1);
    BinTree B = CreateTreeNode(2);
    BinTree C = CreateTreeNode(3);
    BinTree D = CreateTreeNode(4);
    BinTree E = CreateTreeNode(5);
    BinTree F = CreateTreeNode(6);
    BinTree G = CreateTreeNode(7);
    BinTree H = CreateTreeNode(8);
    BinTree I = CreateTreeNode(9);
    BinTree J = CreateTreeNode(10);
    BinTree K = CreateTreeNode(11);
    ConnectNode(A, B, C);
    ConnectNode(B, D, E);
    ConnectNode(D, H, NULL);
    ConnectNode(E, NULL, I);
    ConnectNode(C, F, G);
    ConnectNode(F, J, K);
    return A;
}

BinTree BinaryTree = NULL;
int main(int argc, char const *argv[]) {
    printf("\n \
    Binary Tree Options\n \
    (1).make new linked Binary Tree;\n \
    (2).递归先序遍历;\n \
    (3).递归中序遍历;\n \
    (4).递归后序遍历;\n \
    (5).循环先序遍历;\n \
    (6).循环中序遍历;\n \
    (7).循环后序遍历;\n \
    (8).打印叶节点;\n \
    (9).输出树高度;\n \
    (a).层序遍历;\n \
    (q).quit;\n \
    ");
    while (1) {
        switch (getchar()) {
        case '1':
            BinaryTree = CreateBinTree();
            printf("create success!\n");
            break;
        case '2':
            printf("递归先序遍历:");
            PreOrderTraversal(BinaryTree);
            printf("\n");
            break;
        case '3':
            printf("递归中序遍历:");
            InOrderTraversal(BinaryTree);
            printf("\n");
            break;
        case '4':
            printf("递归后序遍历:");
            PostOrderTraversal(BinaryTree);
            printf("\n");
            break;
        case '5':
            printf("循环先序遍历:");
            PreStackTraversal(BinaryTree);
            printf("\n\n");
            break;
        case '6':
            printf("循环中序遍历:");
            InStackTraversal(BinaryTree);
            printf("\n\n");
            break;
        case '7':
            printf("循环后序遍历:");
            PostStackTraversal(BinaryTree);
            printf("\n\n");
            break;
        case '8':
            printf("打印叶节点:");
            PreOrderPrintLeaves(BinaryTree);
            printf("\n\n");
            break;
        case '9':
            printf("打印height:%d", FindTreeHeight(BinaryTree));
            printf("\n\n");
            break;
        case 'a':
            printf("层序遍历:");
            LevelOrderTraversal(BinaryTree);
            printf("\n");
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



# 执行结果

```sh
☁  binarytree  cat 1.txt | ./a.out

     Binary Tree Options
     (1).make new linked Binary Tree;
     (2).递归先序遍历;
     (3).递归中序遍历;
     (4).递归后序遍历;
     (5).循环先序遍历;
     (6).循环中序遍历;
     (7).循环后序遍历;
     (8).打印叶节点;
     (9).输出树高度;
     (a).层序遍历;
     (q).quit;
     Create Binary Tree like this:
                  1                           A
             /          \                /          \
            2            3              B            C
          /   \        /   \          /   \        /   \
        4      5      6     7       D      E      F     G
       /        \    / \           /        \    / \
      8          9  10  11        H          I  J   K
     create success!
递归先序遍历:1   2   4   8   5   9   3   6   10   11   7
循环先序遍历:1   2   4   8   5   9   3   6   10   11   7

递归中序遍历:2   4   8   5   9   1   3   6   10   11   7
循环中序遍历:8   4   2   5   9   1   10   6   11   3   7

递归后序遍历:2   4   8   5   9   3   6   10   11   7   1
循环后序遍历:8   4   2   5   9   10   6   11   3   7   1

打印叶节点:8   5   9   10   11   7

打印height:4

层序遍历:1   2   3   4   5   6   7   8   9   10   11
```