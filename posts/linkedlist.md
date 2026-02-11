---
title: 单链表
mathjax: true
toc: true
date: 2018-07-02 21:55:52
tags: 
-   链表
categories: 
-   数据结构
---


最近开始准备白天画画板子，写写程序，晚上给自己打点数据结构的基础。首先先实现一下最简单的的单链表。

<!--more-->


# main.c


```c
/*
 * @Author: Zheng Qihang 
 * @Date: 2018-07-02 21:17:17 
 * @Last Modified by: Zheng Qihang
 * @Last Modified time: 2018-07-02 21:55:11
 */
#include <stdio.h>
#include <stdlib.h>
#include "mylist.h"

int main(int argc, char const *argv[])
{
    Node *List = NULL;
    List = InitList(9);//init the list
    Node * pos =NULL;
    PrintList(List);

    /* printf("Let us test FindPos!\n");
    FindPos(3, List); */

   /*  printf("Let us test AddNode!\n");
    AddNode(11,List);
    PrintList(List); */

/*     printf("Let us test DeleteNode!\n");
    DeleteNode(11,List);
    PrintList(List);
    DeleteNode(9,List);
    PrintList(List);
    DeleteNode(5,List);
    PrintList(List); */
    
/*     printf("Let us test ListLength!\n");
    printf("List length %d\n",ListLength(List)); */
    
/*     printf("Let us test FindValue!\n");
    if((pos=FindValue(12,List))!=NULL)
    {
    printf("Node number %d\n",pos->number);
    }
    else
    {
        printf("no value\n");
    } */
    
    return 0;
}

```



# mylist.c


```c
/*
 * @Author: Zheng Qihang 
 * @Date: 2018-07-02 21:14:47 
 * @Last Modified by: Zheng Qihang
 * @Last Modified time: 2018-07-02 21:49:15
 */

#include <stdio.h>
#include <stdlib.h>
#include "mylist.h"
#include <time.h>
#include <stdint.h>

/**
* description  创建长度为n的链表
* @param[in]   长度
* @retval      链表头节点
**/
Node *InitList(uint8_t n)
{
    Node *Head = NULL; //首先为链表设定一个头节点
    Node *p = NULL;    //中间变量
    Node *Last = NULL; //链表的最后一个节点
    uint8_t cnt = 0;
    srand(time(0)); //随机种子

    p = (Node *)malloc(sizeof(Node)); //为链表申请内存
    Head = p;                         //第一个为头
    Last = Head;
    Head->value = rand() % 50;
    Head->number = cnt;
    while (n--)
    {
        cnt++;
        Last = (Node *)malloc(sizeof(Node)); //每次申请的都指向尾节点
        Last->value = rand() % 50;
        Last->number = cnt;
        p->next = Last;
        p = Last;
    }
    Last->next = NULL;
    return Head;
}

/**
* description  find postion in all list
* @param[in]   pos list
* @retval      node address pointer
**/
Node *FindPos(uint8_t pos, Node *list)
{
    Node *pnode;
    pnode = list->next;
    /* old is  while ((pnode->next != NULL) && (pos != 0)) 
    I find the x-- to 0 have a x+1 times
    */
    while ((pnode->next != NULL) && (pos != 1))
    {
        pos--;
        pnode = pnode->next;
        // printf("pos=%2d\tpos->number=%d\n", pos, pnode->number);
    }

    if (pos == 1)
    {
        return pnode;
    }
    else
    {
        return NULL;
    }
}
/**
* description  find the value first time int List
* @param[in]   value List
* @retval      node *
**/
Node *FindValue(uint8_t value, Node *list)
{

    while (list->next != NULL)
    {
        
        if (list->value==value) {
            return list;
        }
        list=list->next;   
    }
    return NULL;
}
/**
* description  在指定位置增加节点
* @param[in]   pos  insert address
* @retval      void
**/
void AddNode(uint8_t pos, Node *list)
{
    srand(time(0));
    Node *pnew = NULL;
    Node *ppos = NULL;
    Node *pnext = NULL;
    pnew = (Node *)malloc(sizeof(Node));
    pnew->value = rand() % 50;
    pnew->next = NULL;
    ppos = FindPos(pos, list);

    if (ppos != NULL)
    {
        if ((pnext = ppos->next) != NULL)
        {
            ppos->next = pnew;
            pnew->next = pnext;
        }
        else
        {
            ppos->next = pnew;
        }
    }
    else
    {
        printf("Add Node failed\n");
    }
}
/**
* description  ret the linked list length
* @param[in]   list
* @retval      uint8_t
**/
uint8_t ListLength(Node *list)
{
    uint8_t cnt = 0;

    while (list->next != NULL)
    {
        cnt++;
        list = list->next;
    }
    return cnt;
}
/**
* description  delete node in postion
* @param[in]   postion list header
* @retval      void
**/
void DeleteNode(uint8_t pos, Node *list)
{
    Node *ppre = FindPos(pos - 1, list); //find a previous node
    Node *ppos = NULL;                   //will be delete node
    if ((ppre != NULL) && ((ppos = ppre->next) != NULL))
    {
        ppre->next = ppos->next;
        free(ppos);
    }
    else
    {
        printf("delete node error\n");
    }
}

/**
* description  打印整个列表
* @param[in]   列表头节点
* @retval      
**/
void PrintList(Node *list)
{
    Node *p = list;
    int cnt = 0;
    while (p)
    {
        printf("addr:%p\tvlaue:%2d\tnext:%14p\tnumber=%d\n", p, p->value, p->next, p->number);
        p = p->next;
    }
    printf("\n");
}
/**
* description  判断链表是否为空
* @param[in]   链表头地址
* @retval      真假
**/
int IsEmpty(Node *list)
{
    return (list->next) == NULL;
}

/**
* description  判断位置是否为链表尾部
* @param[in]   位置 链表头地址
* @retval      真假
**/

int IsLast(char pos, Node *list)
{
    Node *p = list;

    while (p->next != NULL)
    {
        pos--;
        p = p->next;
    }
    if (pos != 0)
    {
        // printf("not last\n");
        return -1;
    }
    else
    {
        // printf("is last\n");
        return 0;
    }
}
```



# mylist.h


```c
/*
 * @Author: Zheng Qihang 
 * @Date: 2018-07-02 21:56:01 
 * @Last Modified by:   Zheng Qihang 
 * @Last Modified time: 2018-07-02 21:56:01 
 */
#ifndef __MYLIST_H
#define __MYLIST_H
#include <stdint.h>
typedef struct _Node
{
    int value;//链表中的数据项
    int number;//计数项
    struct _Node * next;//这里还没有定义Node,要加Struct
} Node;


Node *InitList(uint8_t n);
Node *FindPos(uint8_t pos, Node *list);
void AddNode(uint8_t pos, Node *list);
void PrintList(Node *list);
int IsEmpty(Node *list);
int IsLast(char pos,Node *list);
void DeleteNode(uint8_t pos, Node *list);
uint8_t ListLength(Node * list);
Node *FindValue(uint8_t value, Node *list);
#endif 



``

`