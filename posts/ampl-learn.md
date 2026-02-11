---
title: Ampl学习
mathjax: true
toc: true
categories:
  - 运筹学
date: 2024-06-14 14:46:02
tags:
- Ampl
---

熟悉一下ampl的语法.

<!--more-->

# 表达式系统

## Indexing expressions

索引表达式用于构造一个用于索引的多维集合, 因此他是通过`{}`包裹, 中间使用`,`分割的多个set expressions.

```yaml
indexing: 
  { sexpr-list } 
  { sexpr-list : lexpr } 
sexpr-list:
  sexpr 
  dummy-member in sexpr
  sexpr-list, sexpr
```

比如:
```python
{A, B}
{1..3, B}
{i in A, B} # i可以给后续使用.
{i in A, j in B}
```

## Common expressions

通用表达式主要表示数学运算等内容

```yaml
expr: 
  number 
  variable 
  expr arith-op expr # arith-op is + - less * / mod div ˆ ** 
  unary-op expr # unary-op is `+`, `-` 
  built-in( exprlist ) # built-in is `sin` `cos` ...
  if lexpr then expr [ else expr ] 
  reduction-op indexing expr # reduction-op is sum prod max min 
  ( expr )
```

例子reduction op + indexing expr + (expr binary-op expr):
```python
sum {i in Prod} cost[i] * Make[i]
```

## Logical expressions

逻辑表达式是需要返回true或false的表达式

```yaml
lexpr: 
  expr 
  expr compare-op expr # compare-op is < <= = == != <> > >= 
  lexpr logic-op lexpr # logic-op is or || and && 
  not lexpr 
  member in sexpr 
  member not in sexpr 
  sexpr within sexpr 
  sexpr not within sexpr 
  opname indexing lexpr # opname is `exists` or `forall` 
  ( lexpr )
```

## Set expressions

set表达式专门用于构造set
```yaml
sexpr: 
  { [ member [ , member . . . ] ] }
  sexpr set-op sexpr # set-op is `union`, `diff`, `symdiff`, `inter`, `cross` 
  opname indexing sexpr # opname is `union` or `inter` 
  expr .. expr [ by expr ] # .. 表示线性增长, 可以添加by来指定stride
  setof indexing member  #  member是指set元素构造, setof也就是利用indexing来构造包含有member的set.
  if lexpr then sexpr else sexpr 
  ( sexpr ) 
  interval 
  infinite-set
  indexing # indexing表达式也属于set expr的一种
interval: 
  interval [ a , b ] #  {x: a ≤ x ≤ b}
  interval ( a , b ] #  {x: a < x ≤ b}
  interval [ a , b ) #  {x: a ≤ x < b}
  interval ( a , b ) #  {x: a < x < b}
  integer [ a , b ] #  {x: a ≤ x ≤ b and x ∈ I}
  integer ( a , b ] #  {x: a < x ≤ b and x ∈ I}
  integer [ a , b ) #  {x: a ≤ x < b and x ∈ I}
  integer ( a , b ) #  {x: a < x < b and x ∈ I}
```


例子:
```python
ampl: set y = setof {i in 1..5} (i,iˆ2);
ampl: display y; 
set y := (1,1) (2,4) (3,9) (4,16) (5,25);
```


# 声明系统

声明系统的通用格式如下:

$$
\begin{aligned}
entity\ name\ alias_{opt}\ indexing{opt}\ body{opt};
\end{aligned}
$$

这里entity有`set, param, var, arc, minimize, maximize, subject to, node`, 这里`alias`是`=`号.

## Set declarations
$$
\begin{aligned}
set\ name\ alias_{opt}\ indexing{opt}\ attributes{opt};
\end{aligned}
$$

这里`attributes`:
```yaml
attribute: 
  dimen n 
  within sexpr 
  = sexpr 
  default sexpr
```

## Parameter declarations

$$
\begin{aligned}
param\ name\ alias_{opt}\ indexing{opt}\ attributes{opt};
\end{aligned}
$$

```yaml
attribute: 
  binary 
  integer 
  symbolic 
  relop expr 
  in sexpr 
  = expr 
  default expr 
relop: [<, <=, =, ==, !=, <>, >, >=]
```

这里symbolic表示可能是字符串或者数值

## Variable declarations

$$
\begin{aligned}
var\ name\ alias_{opt}\ indexing{opt}\ attributes{opt};
\end{aligned}
$$

```yaml
attribute: 
  binary 
  integer 
  symbolic 
  >= expr
  <= expr 
  := expr 
  default expr 
  = expr 
  coeff indexing_opt constraint expr
  cover indexing_opt constraint
  obj indexing_opt objective expr
  in sexpr 
  suffix sufname expr
```


# 编程逻辑

## set的理解

如果对应python中的概念, 实际ampl中的set应该是一个展平的list, list中的元素为tuple, 这个tuple就是member, set的维度指的是member这个tuple的维度. 比如这个例子, set就是多个2维的tuple组成的list, 多维的var或者param实际上就是将set A的元素作为key来索引:

```python
ap.eval('set A = {1..2, 2..3};')
ap.display('A')
set A := (1,2) (1,3) (2,2) (2,3);

ap.eval('var B {A} binary;')
ap.display('B')
B :=
1 2   0
1 3   0
2 2   0
2 3   0
```

## 下标索引

只用通过set构造出来的var和param才能使用下标索引, 对于set本身, 需要进行排序后通过内置函数进行访问, 而且ordered只支持1维的set.

```python
ap.eval('set D = {3..1 by -1} ordered;')
set D := 3 2 1;
ap.display('first(D)')
first(D) = 3
```

## 数据加载

对于参数来说, 指定十分麻烦, 但是还好ampl的python api提供了通过list/dict/pandas的方式来辅助指定参数:
```python
ap.eval("param domain { 1..3 };")
ap.param['domain'] = [2048, 384, 8192]
ap.display('domain')
domain [*] :=
1  2048
2   384
3  8192
;
```

## 中间变量定义

通常我们会定义一个中间变量为别的计算计算的结果, 实际上这些中间变量也都是通过约束来表示的, 在ortools里面是给我们封装好了, 但是在ampl中需要分两步定义:
```python
var loadA integer;
subject to LoadA_c : loadA = m * k;
```
