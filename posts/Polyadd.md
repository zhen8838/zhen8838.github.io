---
title: 多项式相加
date: 2018-07-09 20:10:52
categories:
-   数据结构
tags:
-   链表
---

废话不想多说，直接上程序 o(╥﹏╥)o最近腰疼的难受。

<!--more-->


# 程序

```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct PolyNode {
    int coef; // coeffciant
    int expon;
    struct PolyNode *next;
} * Polynomial;

void CreateP(Polynomial P1, Polynomial P2) {
    // P1=3*x^5+4*x^4-1*x^3+2*x^1-1
    Polynomial temp = (Polynomial)malloc(sizeof(struct PolyNode));
    temp->coef = 3;
    temp->expon = 5;
    P1->next = temp;
    temp->next = (Polynomial)malloc(sizeof(struct PolyNode));
    temp = temp->next;
    temp->coef = 4;
    temp->expon = 4;
    temp->next = (Polynomial)malloc(sizeof(struct PolyNode));
    temp = temp->next;
    temp->coef = -1;
    temp->expon = 3;
    temp->next = (Polynomial)malloc(sizeof(struct PolyNode));
    temp = temp->next;
    temp->coef = 2;
    temp->expon = 1;
    temp->next = (Polynomial)malloc(sizeof(struct PolyNode));
    temp = temp->next;
    temp->coef = -1;
    temp->expon = 0;
    temp->next = NULL;
    // P2=2*x^4+1*x^3-7*x^2+1*x^1
    temp = (Polynomial)malloc(sizeof(struct PolyNode));
    temp->coef = 2;
    temp->expon = 4;
    P2->next = temp;
    temp->next = (Polynomial)malloc(sizeof(struct PolyNode));
    temp = temp->next;
    temp->coef = 1;
    temp->expon = 3;
    temp->next = (Polynomial)malloc(sizeof(struct PolyNode));
    temp = temp->next;
    temp->coef = -7;
    temp->expon = 2;
    temp->next = (Polynomial)malloc(sizeof(struct PolyNode));
    temp = temp->next;
    temp->coef = 1;
    temp->expon = 1;
    temp->next = NULL;
}

int Compare(Polynomial P1, Polynomial P2) {

    if (P1->expon > P2->expon) {
        return 1;
    }

    if (P1->expon < P2->expon) {
        return -1;
    }

    if (P1->expon == P2->expon) {
        return 0;
    }
}

void Attach(int Coef, int Expon, Polynomial tail) {
    Polynomial temp = (Polynomial)malloc(sizeof(struct PolyNode));
    temp->coef = Coef;
    temp->expon = Expon;
    temp->next = NULL;
    tail->next = temp;
    tail = temp;
}

Polynomial PolyAdd(Polynomial P1, Polynomial P2) {
    Polynomial front, rear, temp;
    int sum;
    rear = (Polynomial)malloc(sizeof(struct PolyNode));
    front = rear;
    while (P1->next && P2->next) {
        switch (Compare(P1->next, P2->next)) {
        case 1:
            Attach(P1->coef, P1->expon, rear);
            P1 = P1->next;
            break;
        case -1:
            Attach(P2->coef, P2->expon, rear);
            P2 = P2->next;
            break;
        case 0:
            sum = P1->coef + P2->coef;
            if (sum) {
                Attach(sum, P1->expon, rear);
            }
            P1 = P1->next;
            P2 = P2->next;
            break;
        }
    }

    for (; P1; P1 = P1->next) {
        Attach(P1->coef, P1->expon, rear);
    }
    for (; P2; P2 = P2->next) {
        Attach(P2->coef, P2->expon, rear);
    }
    rear->next=NULL;
    temp=front;
    front=front->next;
    free(temp);
    return front;
}

int main(int argc, char const *argv[]) {
    /* create the polynomial */
    Polynomial MP1 = NULL, MP2 = NULL,MPA=NULL;
    MP1 = (Polynomial)malloc(sizeof(struct PolyNode));
    MP2 = (Polynomial)malloc(sizeof(struct PolyNode));
    MPA = (Polynomial)malloc(sizeof(struct PolyNode));
    CreateP(MP1, MP2);
    MPA=PolyAdd(MP1,MP2);
    return 0;
}
```