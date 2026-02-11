---
title: 栈
date: 2018-07-05 21:18:04
categories:
-   数据结构
tags:
-   堆栈
---


栈的粗略实现~~~

<!--more-->


# 代码


```c
/*
 * @Author: Zheng Qihang
 * @Date: 2018-07-05 20:14:48
 * @Last Modified by: Zheng Qihang
 * @Last Modified time: 2018-11-08 16:38:26
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef int ElementType; // data type
#define Node ptrNode     // Node defination
#define Stack ptrNode    // list defination
typedef struct _Node     // node defination
{
    ElementType data;
    struct _Node *next;
} * ptrNode; // Node is a pointer to _Node

int IsEmpty(Stack S) { return S->next == NULL; }

Stack CreateStack(void) {
    Stack S = (Stack)malloc(sizeof(struct _Node));
    S->next = NULL;
    return S;
}

ElementType Pop(Stack S) {
    int temp;
    Node tmepNode = S->next;
    if (tmepNode != NULL) {
        temp = tmepNode->data;
        S->next = tmepNode->next;
        free(tmepNode);
        return temp;
    } else {
        return -1;
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

void Push(ElementType X, Stack S) {
    Node tempNode = S->next;
    Node newNode = (Node)malloc(sizeof(struct _Node));
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

ElementType Top(Stack S) {
    if (S->next != NULL) {
        return S->next->data;
    } else {
        return -1;
    }
}

void PrintStack(Stack S) {
    Node tep = S->next;
    while (tep != NULL) {
        printf("addr=0x%3lX  data=%d  nextaddr=0x%3lX\n",
               (unsigned long int)tep & 0xFFF, tep->data,
               (unsigned long int)tep->next & 0xFFF);
        tep = tep->next;
    }
}

int main(int argc, char const *argv[]) {
    int tempint;
    Stack MyStack = NULL;
    printf("\n \
   基本操作：\n \
   (a).make new Stack;\n \
   (b).push data;\n \
   (c).pop  data;\n \
   (d).find Stack's top;\n \
   (e).empty the Stack;\n \
   (f).dispose the  Stack;\n \
   (p).print the stack:\n \
   (q).quit;\n \
   ");
    while (1) {
        switch (getchar()) {
        case 'a':
            printf("create a new Stack\n");
            MyStack = CreateStack();
            if (MyStack != NULL) {
                printf("create stack success!\n");
            }
            break;
        case 'b':
            printf("enter and Push data in Stack:\n");
            scanf("%d", &tempint);
            Push(tempint, MyStack);
            break;
        case 'c':
            printf("pop data is:%d\n", Pop(MyStack));
            break;
        case 'd':
            printf("stack top is:%d\n", Top(MyStack));
            break;
        case 'e':
            MakeEmpty(MyStack);
            break;
        case 'f':
            DisposeStack(MyStack);
            break;
        case 'p':
            PrintStack(MyStack);
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
``

`