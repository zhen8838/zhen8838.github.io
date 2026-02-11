---
title: 中缀表达式转后缀表达式
date: 2018-07-06 20:59:47
mathjax: true
tags:
-   堆栈
categories:
-   数据结构
---

我用c语言写了个中缀表达式转后缀表达式代码。。。网上教程多的不要不要的，不过我觉得还是看数据结构-c语言实现是最舒服的。

<!--more-->


#   程序
这个程序只支持**个位数**的数字运算！(我偷懒不想写太多~~)

```c
/*
 * @Author: Zheng Qihang
 * @Date: 2018-07-05 20:14:48
 * @Last Modified by: Zheng Qihang
 * @Last Modified time: 2018-11-08 16:37:02
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef enum {
    ADD = 0,
    SUBT = 0,
    MULT = 1,
    DIVD = 1,
    LEFTB = -1,
    RIGHTB =-1,
    NLL = -2,
} Operator;

typedef char ElementType; // data type
#define Node ptrNode      // Node defination
#define Stack ptrNode     // list defination
typedef struct _Node      // node defination
{
    ElementType data;
    struct _Node *next;
} * ptrNode; // Node is a pointer to _Node

void CheckStack(Stack S) {
    if (S == NULL) {
        printf("stack is invaild!\n");
    }
}

int IsEmpty(Stack S) {
    CheckStack(S);
    return S->next == NULL;
}

Stack CreateStack(void) {
    Stack S = (Stack)malloc(sizeof(struct _Node));
    CheckStack(S);
    S->next = NULL;
    return S;
}

ElementType Pop(Stack S) {
    int temp;
    CheckStack(S);
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
    CheckStack(S);
    while (S->next != NULL) {
        Pop(S);
    }
}
void DisposeStack(Stack S) {
    CheckStack(S);
    MakeEmpty(S);
    free(S);
}

void Push(ElementType X, Stack S) {
    CheckStack(S);
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
    CheckStack(S);
    if (S->next != NULL) {
        return S->next->data;
    } else {
        return -1;
    }
}

void PrintOutPut(Stack S) {
    Node tep = S->next;
    char *str=(char*)malloc(30*sizeof(char));
    int cnt=0;
    CheckStack(S);
    while (tep != NULL) {
        *(str+(cnt++))=tep->data;//add char to string;
        tep = tep->next;
    }
    while(cnt--){
        printf("%c",*(str+cnt));//Reverse the string and print
    }
    free(str);//avoid memory leaks!!
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

Operator ChToOpt(char C) {
    Operator temp;
    switch (C) {
    case '+':
        temp = ADD;
        break;
    case '-':
        temp = SUBT;
        break;
    case '*':
        temp = MULT;
        break;
    case '/':
        temp = DIVD;
        break;
    case '(':
        temp = LEFTB;
        break;
    case ')':
        temp = RIGHTB;
        break;
    default:
        temp = NLL;
        break;
    }
    return temp;
}

void printfCovt(Stack stack, Stack out, int i) {
    printf("-------STEP  %d---------\n", i);
    printf("stack :  ");
    PrintOutPut(stack);
    printf("   <--\n");
    printf("output:  ");
    PrintOutPut(out);
    printf("   <--\n");
    printf("-----------------------\n\n");
}

int main(int argc, char const *argv[]) {
    int tempint;
    Stack MyStack, output = NULL;
    uint8_t i = 0;
    char *func = "1+2*3+(4*5+6)*7";
    Operator curOp;
    Operator TopOp;
    MyStack = CreateStack(); // make new stack;
    output = CreateStack();
    for (i = 0; i < strlen(func); i++) {

        switch (*(func + i)) {
        case '+':
        case '-':
        case '*':
        case '/':
            do {
                curOp = ChToOpt(*(func + i));
                TopOp = ChToOpt(Top(MyStack));
                // printf("topop:%d curop :%d\n", TopOp, curOp);
                if (TopOp >= curOp) {
                    Push(Pop(MyStack), output);    // move stack to output
                    TopOp = ChToOpt(Top(MyStack)); // refresh TopOperater
                }
            } while (TopOp >= curOp);
            Push(*(func + i), MyStack); // add Current Operater in stack
            break;
        case '(':
            Push(*(func + i), MyStack); // add left brackets in stack
            break;
        case ')':
            while (Top(MyStack) != '(') {
                // move stack to output utill top is '('
                Push(Pop(MyStack), output); 
            }
            Pop(MyStack); // pop the '('
            break;
        default: // when func[i]=0~9
            Push(*(func + i), output);
            break;
        }
        printfCovt(MyStack, output, i);
    }

    while (!IsEmpty(MyStack)) {
        Push(Pop(MyStack), output); // move stack to output
    }
    printfCovt(MyStack, output, i++);

    return 0;
}
```


# 运行结果
```sh
-------STEP  0---------
stack :     <--
output:  1   <--
-----------------------

-------STEP  1---------
stack :  +   <--
output:  1   <--
-----------------------

-------STEP  2---------
stack :  +   <--
output:  12   <--
-----------------------

-------STEP  3---------
stack :  +*   <--
output:  12   <--
-----------------------

-------STEP  4---------
stack :  +*   <--
output:  123   <--
-----------------------

-------STEP  5---------
stack :  +   <--
output:  123*+   <--
-----------------------

-------STEP  6---------
stack :  +(   <--
output:  123*+   <--
-----------------------

-------STEP  7---------
stack :  +(   <--
output:  123*+4   <--
-----------------------

-------STEP  8---------
stack :  +(*   <--
output:  123*+4   <--
-----------------------

-------STEP  9---------
stack :  +(*   <--
output:  123*+45   <--
-----------------------

-------STEP  10---------
stack :  +(+   <--
output:  123*+45*   <--
-----------------------

-------STEP  11---------
stack :  +(+   <--
output:  123*+45*6   <--
-----------------------

-------STEP  12---------
stack :  +   <--
output:  123*+45*6+   <--
-----------------------

-------STEP  13---------
stack :  +*   <--
output:  123*+45*6+   <--
-----------------------

-------STEP  14---------
stack :  +*   <--
output:  123*+45*6+7   <--
-----------------------

-------STEP  15---------
stack :     <--
output:  123*+45*6+7*+   <--
-----------------------

```