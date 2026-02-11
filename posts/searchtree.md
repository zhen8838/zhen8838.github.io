---
title: 二叉搜索树
date: 2018-07-14 22:18:20
categories:
-   数据结构
tags:
-   树
---


这几天搬寝室烦的一批，都没时间写代码，很烦。明天出去玩了，今天赶紧把这个写完。


<!--more-->


# 程序


```c
/*
 * @Author: Zheng Qihang
 * @Date: 2018-07-14 21:29:18
 * @Last Modified by: Zheng Qihang
 * @Last Modified time: 2018-11-08 16:38:15
 */
/*
 * @Author: Zheng Qihang
 * @Date: 2018-07-10 09:38:39
 * @Last Modified by: Zheng Qihang
 * @Last Modified time: 2018-07-13 14:38:44
 */
#include <limits.h>
#include <math.h>
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
} * BinTree_t;

/************************* Queue define! **************************/
#define ElementQueueType BinTree_t
typedef struct QueueNode {
    ElementQueueType data;
    struct QueueNode *next;
} * QNode_t;

typedef struct QueueHeader {
    QNode_t front;
    QNode_t rear;
} * Queue_t;

/**
 * description  Create the Queue!
 * @param[in]   void
 * @retval      no
 **/
Queue_t CreateQueue(void) {
    Queue_t Q = (Queue_t)malloc(sizeof(struct QueueHeader));
    Q->front = NULL;
    Q->rear = NULL;
    return Q;
}

/**
 * description  Add Queue Node !
 * @param[in]   Queue header and a item
 * @retval      void
 **/
void AddQ(Queue_t Q, ElementQueueType item) {
    if (!Q) {
        return;
    }
    /* 取消此处为了便于绘图
    if (!item) {
        return;
    }
    */
    QNode_t temp = (QNode_t)malloc(sizeof(struct QueueNode));

    if (!item) { //添加此处为了便于绘图
        temp->data = NULL;
    } else {
        temp->data = item;
    }
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
int IsEmptyQ(Queue_t Q) { return !Q->front; }

/**
 * description  Delete the Queue Node
 * @param[in]   Queue header
 * @retval      data
 **/
ElementQueueType DeleteQ(Queue_t Q) {
    if (!Q) {
        return NULL;
    }
    if (IsEmptyQ(Q)) {
        return NULL;
    }
    QNode_t temp = Q->front;
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

/**
 * description  Delete the queue
 * @param[in]   Queue_t
 * @retval      void
 **/
void DisposeQueue(Queue_t Q) {
    while (!IsEmptyQ(Q)) {
        DeleteQ(Q);
    }
    free(Q);
}

void PrintQueue(Queue_t Q) {

    if (IsEmptyQ(Q)) {
        return;
    }
    printf("打印队列数据元素：\n");
    QNode_t temp = Q->front;

    for (; temp; temp = temp->next) {
        printf("0x%3lX   ", (unsigned long int)temp->data & 0xFFF);
    }
    printf("\n");
}

/************************* Queue define! **************************/

/************************* Stack define! **************************/

#define ElementStackType BinTree_t
typedef struct StackNode // node defination
{
    ElementStackType data;
    struct StackNode *next;
} * Stack_t; // Node is a pointer to _Node

int IsEmpty(Stack_t S) { return !S->next; }

Stack_t CreateStack(void) {
    Stack_t S = (Stack_t)malloc(sizeof(struct StackNode));
    S->next = NULL;
    return S;
}

ElementStackType Pop(Stack_t S) {
    ElementStackType temp;
    Stack_t tmepNode = S->next;
    if (tmepNode != NULL) {
        temp = tmepNode->data;
        S->next = tmepNode->next;
        free(tmepNode);
        return temp;
    } else {
        return NULL;
    }
}

void MakeEmpty(Stack_t S) {
    while (S->next != NULL) {
        Pop(S);
    }
}
void DisposeStack(Stack_t S) {
    MakeEmpty(S);
    free(S);
}

void Push(Stack_t S, ElementStackType X) {
    Stack_t tempNode = S->next;
    Stack_t newNode = (Stack_t)malloc(sizeof(struct StackNode));
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

ElementStackType Top(Stack_t S) {
    if (S->next != NULL) {
        return S->next->data;
    } else {
        return NULL;
    }
}

void PrintStack(Stack_t S) {
    Stack_t tep = S->next;
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
 * description  create the binary tree node
 * @param[in]   data
 * @retval      bintree_t
 **/
BinTree_t CreateTreeNode(int dat) {
    BinTree_t new = (BinTree_t)malloc(sizeof(struct TreeNode));
    new->Data = dat;
    new->Left = NULL;
    new->Right = NULL;
    return new;
}

void ConnectNode(BinTree_t root, BinTree_t left, BinTree_t right) {

    if (!root) {
        printf("error in null\n");
        return;
    }
    root->Left = left;
    root->Right = right;
}

/**
 * description  层序遍历二叉树
 * @param[in]   BinTree_t
 * @retval      void
 **/
void LevelOrderTraversal(BinTree_t BT) {
    Queue_t myqueue = CreateQueue();
    BinTree_t temp = NULL;
    AddQ(myqueue, BT); //将头节点入队
    while (!IsEmptyQ(myqueue)) {
        temp = DeleteQ(myqueue); //将上一层节点出队
        if (temp != NULL) {
            printf("%d   ", temp->Data);
            AddQ(myqueue, temp->Left); //再将上层节点左右子节点入队
            AddQ(myqueue, temp->Right);
        }
    }
    DisposeQueue(myqueue);
}

/**
 * description  输出二叉树的高度
 * @param[in]   BinTree_t
 * @retval      int
 **/
int FindTreeHeight(BinTree_t BT) {
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
 * description  绘制二叉树图像
 * @param[in]   BinTree_t
 * @retval      void
 **/
void DrawTree(BinTree_t BT) {
    int height = FindTreeHeight(BT);
    int cnt = 0;

    /* 开始层序遍历二叉树并且绘制图形 */
    Queue_t myqueue = CreateQueue();
    BinTree_t temp = NULL;
    int width = 0;
    AddQ(myqueue, BT); //将头节点入队
    /*现在修改了入队函数,空指针也可以入队
    所以在出队的时候就需要进行判断  */
    for (int i = 0; i < height; i++) {
        cnt = (int)pow(2, i);           //当前行的个数21
        width = pow(2, height - i + 1); //设置宽度为2^(height-i+1)
        // printf("width:%d\n",width);
        while (cnt--) {
            temp = DeleteQ(myqueue); //将上一层节点出队
            if (temp != NULL) {
                printf("%*d%*c", width, temp->Data, width, ' '); //输出数据
                AddQ(myqueue, temp->Left); //再将上层节点左右子节点入队
                AddQ(myqueue, temp->Right);
            } else {
                printf("%p   ", temp);
            }
        }
        printf("\n");

        if (i != height - 1) {
            /* 先记录下一行元素个数 */
            int nextwidth = (int)pow(2, height - i);
            /* 如果不是最后一行那么按位置打印'+' */
            for (int J = 0; J < (int)pow(2, i); J++) {
                printf("%*c%*c", width, '+', width, ' ');
            }
            printf("\n");
            /* 并且以他下一层元素的个数打印'-' */
            for (int k = 0; k < (int)pow(2, i + 1); k++) {
                /* 每隔一位去打印width个'-' */
                if ((k % 2) == 0) {
                    printf("%*c", nextwidth, '-');
                    for (uint8_t z = 0; z < nextwidth; z++) {
                        printf("-");
                    }
                } else {
                    for (uint8_t z = 0; z < nextwidth; z++) {
                        printf("-");
                    }
                    printf("%*c", nextwidth, ' ');
                }
            }
            printf("\n");
            /* 再继续向下按位置打印'|' */
            // for (int J = 0; J < (int)pow(2, i + 1); J++) {
            //     printf("%*c%*c", nextwidth, '+', nextwidth, ' ');
            // }
            // printf("\n");
        }
        printf("\r");
    }
    DisposeQueue(myqueue);
}

/**
 * description  在二叉搜索树中寻找元素
 * @param[in]   BinTree_t DATA
 * @retval      BinTree_t
 **/
BinTree_t FindElement(BinTree_t BST, ElementType Dat) {

    BinTree_t temp = BST;
    if (BST == NULL) {
        return NULL;
    }
    while (temp) {
        if (Dat > temp->Data) {
            temp = temp->Right;
        } else if (Dat < temp->Data) {
            temp = temp->Left;
        } else {
            return temp;
        }
    }
    return NULL;
}

/**
 * description  寻找最大的元素
 * @param[in]   BinTree_t
 * @retval      BinTree_t
 **/
BinTree_t FindMax(BinTree_t BST) {

    if (BST->Right == NULL) {
        return BST;
    }
    return FindMax(BST->Right);
}

/**
 * description  寻找最小的元素
 * @param[in]   BinTree_t
 * @retval      BinTeww
 **/
BinTree_t FindMin(BinTree_t BST) {
    BinTree_t tmp = BST;
    while (tmp->Left) {
        tmp = tmp->Left;
    }
    return tmp;
}

/**
 * @brief  用于删除二叉树中的某个元素
 * @param[in]   BST Dat
 * @param[out]  void
 * @return      void
 **/
BinTree_t DeleteElementBST(BinTree_t BST, ElementType Dat) {
    BinTree_t tmpNode;
    if (BST == NULL) {
        return NULL;
    } else if (Dat < BST->Data) {
        BST->Left = DeleteElementBST(BST->Left, Dat);
    } else if (Dat > BST->Data) {
        BST->Right = DeleteElementBST(BST->Right, Dat);
    } else if (BST->Left && BST->Right) {
        tmpNode = FindMin(BST->Right);
        BST->Data = tmpNode->Data;
        BST->Right = DeleteElementBST(BST->Right, BST->Data);
    } else {
        tmpNode = BST;

        if (BST->Left == NULL) {
            BST = BST->Right;
        } else if (BST->Right == NULL) {
            BST = BST->Left;
        }
        free(tmpNode);
    }
    return BST;
}

// BinTree_t DeleteElementBST(BinTree_t BST, ElementType Dat) {

//     BinTree_t tempNode = BST;
//     BinTree_t lastNode = BST;
//     if (BST == NULL) {
//         return NULL;
//     }
//     while (tempNode) {
//         if (Dat > tempNode->Data) {
//             lastNode = tempNode; //记录值
//             tempNode = tempNode->Right;
//         } else if (Dat < tempNode->Data) {
//             lastNode = tempNode; //记录值
//             tempNode = tempNode->Left;
//         } else if (Dat == tempNode->Data) { //寻找到了该节点
//             /* 有两个子节点 替换右子树中最小值并且删除那个节点 */
//             if (tempNode->Right && tempNode->Left) {
//                 BinTree_t minNode = FindMin(tempNode->Right);
//                 tempNode->Data = minNode->Data;
//                 tempNode->Right =
//                     DeleteElementBST(tempNode->Right, minNode->Data);
//             } else { /* 有一个或没有子节点 */

//                 /* 如果左边为空 则指向右子节点 */
//                 if (tempNode->Left == NULL) {
//                     lastNode->Right = tempNode->Right;
//                 } else if (tempNode->Right == NULL) {
//                     lastNode->Left = tempNode->Left;
//                 }

//                 if (lastNode == tempNode) {
//                     /* 防止只有一个节点并递归时发生错误 */
//                     free(tempNode);
//                     lastNode = tempNode = NULL;
//                 } else {
//                     free(tempNode);
//                 }

//                 printf("lastNode=%p\n", lastNode);
//                 printf("tempNode=%p\n", tempNode);
//                 return lastNode;
//             }
//         }
//     }
//     return NULL;
// }

/**
 * description  在二叉搜索树中插入一个节点
 * @param[in]   数据
 * @retval      void
 **/
void InsertNode(ElementType X, BinTree_t BST) {
    if (BST == NULL) {
        return;
    }
    BinTree_t newNode = (BinTree_t)malloc(sizeof(struct TreeNode));
    BinTree_t temp = BST;
    BinTree_t last = BST;
    newNode->Data = X;
    newNode->Left = newNode->Right = NULL;
    while (temp) {
        last = temp; //记录上一次的位置
        if (X > temp->Data) {
            temp = temp->Right;
        } else if (X < temp->Data) {
            temp = temp->Left;
        } else if (X == temp->Data) {
            printf("已经存在相同元素");
            return;
        }
    }

    if (X > last->Data) { //判断插入左右节点
        last->Right = newNode;
    } else {
        last->Left = newNode;
    }
}

/**
 * @brief  打印使用说明
 * @param[in]   void
 * @param[out]  void
 * @return      void
 **/
void UseageHelp(void) {
    printf("\n \
    搜索树的操作\n \
    (1).创建新的搜索树;\n \
    (2).输出树高度;\n \
    (3).层序遍历;\n \
    (4).绘制二叉树;\n \
    (5).寻找某元素:\n \
    (6).递归寻找最大值:\n \
    (7).循环寻找最小值:\n \
    (8).插入一个元素:\n \
    (9).删除一个元素:\n \
    (h).帮助;\n \
    (q).quit;\n \
    ");
}

/**
 * description  创建一个我想要的树
 * @param[in]   void
 * @retval      BinTree_t
 **/
BinTree_t CreateBinTree(void) {
    printf("Create Binary Tree like this:        \n\
                     90                          \n\
                     |                           \n\
         ------------------------                \n\
         |                      |                \n\
         50                     150              \n\
         |                      |                \n\
   -------------          -------------          \n\
   |           |          |           |          \n\
   20          75         95          175        \n\
   |           |          |           |          \n\
-------     -------    -------     -------       \n\
|     |     |     |    |     |     |     |       \n\
5     nil   nil   80   92    111   nil   nil     \n\
    ");
    BinTree_t A = CreateTreeNode(90);
    BinTree_t B = CreateTreeNode(50);
    BinTree_t C = CreateTreeNode(150);
    BinTree_t D = CreateTreeNode(20);
    BinTree_t E = CreateTreeNode(75);
    BinTree_t F = CreateTreeNode(95);
    BinTree_t G = CreateTreeNode(175);
    BinTree_t H = CreateTreeNode(5);
    BinTree_t I = CreateTreeNode(25);
    BinTree_t J = CreateTreeNode(66);
    BinTree_t K = CreateTreeNode(80);
    BinTree_t L = CreateTreeNode(92);
    BinTree_t M = CreateTreeNode(111);
    ConnectNode(A, B, C);
    ConnectNode(B, D, E);
    ConnectNode(D, H, NULL);
    ConnectNode(E, NULL, K);
    ConnectNode(C, F, G);
    ConnectNode(F, L, M);
    ConnectNode(G, NULL, NULL);
    return A;
}

BinTree_t BinaryTree = NULL;
int main(int argc, char const *argv[]) {
    BinTree_t tmp = NULL;
    int empdat = 0;
    UseageHelp();
    while (1) {
        switch (getchar()) {
        case '1':
            BinaryTree = CreateBinTree();
            printf("创建成功!\n");
            break;
        case '2':
            printf("输出树高度:%d", FindTreeHeight(BinaryTree));
            printf("\n");
            break;
        case '3':
            printf("层序遍历:");
            LevelOrderTraversal(BinaryTree);
            printf("\n");
            break;
        case '4':
            printf("绘制二叉树:\n");
            DrawTree(BinaryTree);
            printf("\n");
            break;
        case '5':
            printf("寻找某元素:\n");
            printf("请输入元素值:");
            scanf("%d", &empdat);
            tmp = FindElement(BinaryTree, empdat);
            if (!tmp) {
                printf("not find the node");
            } else {
                printf("find the node");
            }

            printf("\n");
            break;
        case '6':
            printf("递归寻找最大值:\n");
            tmp = FindMax(BinaryTree);
            printf("%d", tmp->Data);
            printf("\n");
            break;
        case '7':
            printf("递归寻找小最值:\n");
            tmp = FindMin(BinaryTree);
            printf("%d", tmp->Data);
            printf("\n");
            break;
        case '8':
            printf("插入一个元素:\n");
            printf("请输入需要插入的值:");
            scanf("%d", &empdat);
            InsertNode(empdat, BinaryTree);
            printf("\n");
            break;
        case '9':
            printf("删除一个元素:\n");
            printf("请输入需要删除的值:");
            scanf("%d", &empdat);
            DeleteElementBST(BinaryTree, empdat);
            printf("\n");
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




# 运行结果


我使用这个文本作为输入

```sh
☁  serachtree  cat test
1
2
3
4
5
90
6
7
8
30
4
9
90
4
q
☁  serachtree  gcc searchtree.c -lm &&  cat test | ./a.out

     搜索树的操作
     (1).创建新的搜索树;
     (2).输出树高度;
     (3).层序遍历;
     (4).绘制二叉树;
     (5).寻找某元素:
     (6).递归寻找最大值:
     (7).循环寻找最小值:
     (8).插入一个元素:
     (9).删除一个元素:
     (h).帮助;
     (q).quit;
     Create Binary Tree like this:
                     90
                     |
         ------------------------
         |                      |
         50                     150
         |                      |
   -------------          -------------
   |           |          |           |
   20          75         95          175
   |           |          |           |
-------     -------    -------     -------
|     |     |     |    |     |     |     |
5     nil   nil   80   92    111   nil   nil
    创建成功!
输出树高度:4
层序遍历:90   50   150   20   75   95   175   5   80   92   111
绘制二叉树:
                              90
                               +
               ---------------------------------
              50                             150
               +                               +
       -----------------               -----------------
      20              75              95             175
       +               +               +               +
   ---------       ---------       ---------       ---------
   5    (nil)   (nil)     80      92     111    (nil)   (nil)

寻找某元素:
请输入元素值:find the node
递归寻找最大值:
175
递归寻找小最值:
5
插入一个元素:
请输入需要插入的值:
绘制二叉树:
                              90
                               +
               ---------------------------------
              50                             150
               +                               +
       -----------------               -----------------
      20              75              95             175
       +               +               +               +
   ---------       ---------       ---------       ---------
   5      30    (nil)     80      92     111    (nil)   (nil)

删除一个元素:
请输入需要删除的值:
绘制二叉树:
                              92
                               +
               ---------------------------------
              50                             150
               +                               +
       -----------------               -----------------
      20              75              95             175
       +               +               +               +
   ---------       ---------       ---------       ---------
   5      30    (nil)     80    (nil)    111    (nil)   (nil)

```



