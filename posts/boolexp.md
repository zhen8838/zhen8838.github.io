---
title: Boolean Expressions
date: 2018-09-25 20:55:57
mathjax: true
tags:
-   LeetCode
categories:
-   数据结构
---


## 描述

The objective of the program you are going to produce is to evaluate boolean expressions as the one shown next: 
    <center>Expression: ( V | V ) & F & ( F | V )</center>

where V is for True, and F is for False. The expressions may include the following operators: ! for not , & for and, | for or , the use of parenthesis for operations grouping is also allowed. 

To perform the evaluation of an expression, it will be considered the priority of the operators, the not having the highest, and the or the lowest. The program must yield `V` or `F` , as the result for each expression in the input file. 

<!--more-->
## 输入

The expressions are of a variable length, although will never exceed 100 symbols. Symbols may be separated by any number of spaces or no spaces at all, therefore, the total length of an expression, as a number of characters, is unknown. 

The number of expressions in the input file is variable and will never be greater than 20. Each expression is presented in a new line, as shown below. 


## 输出

For each test expression, print "`Expression` " followed by its sequence number, ": ", and the resulting value of the corresponding test expression. Separate the output for consecutive test expressions with a new line. 

Use the same format as that shown in the sample output shown below. 


## 样例输入
```sh
( V | V ) & F & ( F| V)
!V | V & V & !F & (F | V ) & (!F | F | !V & V)
(F&F|V|!V&!F&!(F|F&V))
```
## 样例输出

```sh
Expression 1: F
Expression 2: V
Expression 3: V
```


# AC代码

```cpp
#include <algorithm>
#include <iostream>
#include <sstream>
using namespace std;
#define Dcout(x) cout << #x << ": " << (x) << endl

char s[101];
string str;
string::iterator index;

bool term_value();
bool factor_value() {
    bool result;
    char c= *index; /* 获取当前元素 */
    if (c == '(') { /* 括号中代表另一个表达式 */
        ++index;
        result= term_value();
        ++index;           /* 除去右括号 */
    } else if (c == '!') { /* ！后面是另一个因子 */
        ++index;
        result= !factor_value(); /* 值取反 */
    } else {
        if (c == 'V') { /* 非V即F */
            result= true;
        } else if (c == 'F') {
            result= false;
        }
        ++index;
    }
    return result;
}
bool term_value() {              /* 求第一项的值 */
    bool result= factor_value(); /* 求第一个因子的值 */
    bool more= true;
    while (more) {
        char op= *index;               /* 获取当前元素 */
        if (op == '&' || op == '|') {  /* 两个因子之间& | */
            ++index;                   /* 取出一个元素 */
            int value= factor_value(); /* 下一个因子的值 */
            if (op == '&') {
                result&= value;
            } else {
                result|= value;
            }
        } else {
            more= false;
        }
    }
    return result;
}
int main(int argc, char const *argv[]) {
    int casecnt= 1;
    while (cin.getline(s, 101)) {
        str= s;
        if (str.size() < 1) { /* 长度小于1退出 */
            break;
        } else {
            /* 去除空格 */
            string::iterator i= find(str.begin(), str.end(), ' ');
            while (i != str.end()) {
                str.erase(i);
                i= find(str.begin(), str.end(), ' ');
            }
        }
        index= str.begin();
        cout << "Expression " << casecnt++ << ": "
             << ((term_value() == true) ? 'V' : 'F') << endl;
    }
    return 0;
}
```
