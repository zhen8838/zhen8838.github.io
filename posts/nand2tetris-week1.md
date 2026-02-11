---
title: Nand2Tetris week1
mathjax: true
toc: true
categories:
  - 体系结构
date: 2021-05-04 13:21:05
tags:
- Nand2Tetris
---

准备系统学习计算机体系结构，网上的课程还是挺多的，不过感觉coursera上这门[从与非门到俄罗斯方块](https://www.coursera.org/learn/build-a-computer/home/welcome)感觉好一些，虽然学校排名不高，但基于项目导向会比较容易吸收一点。

其他的课我找到有，有兴趣的同学也可以看看：
1.  [伯克利计算机体系结构cs152](https://inst.eecs.berkeley.edu/~cs152/sp21/)


2.  [ETH Digital Design & Computer Architecture](https://www.youtube.com/watch?v=AJBmIaUneB0&t=519s)

3.  [ETH Computer Architecture](https://www.youtube.com/watch?v=c3mPdZA-Fmc&list=PL5Q2soXY2Zi9xidyIgBxUz7xRPS-wisBN) 这是上一门的进阶课程。

4.  [berkeley 机器学习硬件](https://inst.eecs.berkeley.edu/~ee290-2/sp21/) 好课，看了下lab就知道内容很硬核，但下载lab代码都需要内部凭证。。

5.  [LAFF-On Programming for High Performance](https://learning.edx.org/course/course-v1:UTAustinX+UT.PHP.16.01x+2T2020/home) cpu性能调优，注册费50刀

6.  [Computation Structures 1-3](https://learning.edx.org/course/course-v1:MITx+6.004.1x_3+3T2016/home) 由三部分组成，内容应该是蛮多的。


<!--more-->


# Boolean Logic

## 交换律

$$
\begin{aligned}
  x \land y = y \land x \\
  x \lor y = y \lor x
\end{aligned}
$$

## 结合律

$$
\begin{aligned}
  x \land (y \land z) = (x \land y) \land z \\
  x \lor (y \lor z) = (x \lor y) \lor z \\
\end{aligned}
$$

## 分配律

$$
\begin{aligned}
  x \land (y \lor z) = (x \land y) \lor (x \land z) \\
  x \lor (y \land z) = (x \lor y) \land (x \lor z)
\end{aligned}
$$

## 摩根定律

$$
\begin{aligned}
  \lnot ( x \land y ) = \lnot x \lor \lnot y \\
  \lnot ( x \lor y ) = \lnot x \lor \lnot y 
\end{aligned}
$$


## example

$$
\begin{aligned}
  & \lnot (\lnot (x) \land \lnot(x \lor y)) \\
  = &  \lnot (\lnot (x) \land (\lnot(x)  \land \lnot (y))) \\
  = & \lnot((\lnot (x) \land \lnot(x))  \land \lnot (y))) \\
  = & \lnot(\lnot (x) \land \lnot (y))) \\
  = & \lnot(\lnot (x \lor y)) \\
  = & (x \lor y)
\end{aligned}
$$

或者可以用真值表的方法得到化简的结果。


# Boolean functional synthesis

## 从真值表推导出公式

| x   | y   | z   | f   | $\lnot x \land  \lnot y \land  \lnot z $ | $\lnot x \land y \land  \lnot z $ | $x \land \not y \land  \lnot z $ |
| --- | --- | --- | --- | ---------------------------------------- | --------------------------------- | -------------------------------- |
| 0   | 0   | 0   | 1   | 1                                        | 0                                 | 0                                |
| 0   | 0   | 1   | 0   | 0                                        | 0                                 | 0                                |
| 0   | 1   | 0   | 1   | 0                                        | 1                                 | 0                                |
| 0   | 1   | 1   | 0   | 0                                        | 0                                 | 0                                |
| 1   | 0   | 0   | 1   | 0                                        | 0                                 | 1                                |
| 1   | 0   | 1   | 0   | 0                                        | 0                                 | 0                                |
| 1   | 1   | 0   | 0   | 0                                        | 0                                 | 0                                |
| 1   | 1   | 1   | 0   | 0                                        | 0                                 | 0                                |


最后把三个分步骤的都用**或**进行连接就得到了一个公式。

## 利用NAND构建任意操作

###  NAND = (not (x and y))

| x   | y   | NAND |
| --- | --- | ---- |
| 0   | 0   | 1    |
| 0   | 1   | 1    |
| 1   | 0   | 1    |
| 1   | 1   | 0    |


1.  $\lnot x = x \text{ nand } x$
1.  $x \land y = \lnot(x \text{ nand } y)$
1.  $x \lor y = \lnot x \text{ nand } \lnot y$


# Logic Gates

下面这个图可以很好的解释交换律：
![](nand2tetris-week1/and.png)

# Hardware Description Language

HDL没有执行顺序，所以我们可以在代码块中以任意顺序编写，我们需要关注的是如何使用更少的连接或者消耗完成相同的功能。

# Hardware Simulation 

使用一个hdl文件，装载到模拟器中，然后交互式的进行测试。或者自己编写测试脚本test script进行测试。

# Multi-Bit Buses

在hdl文件中可以直接输入`a[16]`，来表示16位输入，并且可以用索引的方式取出其中的位，或者使用`[0..7]`的方式取一定范围的数据。


# project 1

第一课的作业就是让我们利用`NAND`写hdl然后去构造各个`boolean`表达式的逻辑门。


```vhdl
/**
 * And gate: 
 * out = 1 if (a == 1 and b == 1)
 *       0 otherwise
 */

CHIP And {
    IN a, b;
    OUT out;

    PARTS:
    // Put your code here:
    Nand(a=a, b=b, out=c);
    Not(in=c, out=out);
}

/**
 * 16-bit bitwise And:
 * for i = 0..15: out[i] = (a[i] and b[i])
 */

CHIP And16 {
    IN a[16], b[16];
    OUT out[16];

    PARTS:
    // Put your code here:
    And(a=a[0], b=b[0], out=out[0]);
    And(a=a[1], b=b[1], out=out[1]);
    And(a=a[2], b=b[2], out=out[2]);
    And(a=a[3], b=b[3], out=out[3]);
    And(a=a[4], b=b[4], out=out[4]);
    And(a=a[5], b=b[5], out=out[5]);
    And(a=a[6], b=b[6], out=out[6]);
    And(a=a[7], b=b[7], out=out[7]);
    And(a=a[8], b=b[8], out=out[8]);
    And(a=a[9], b=b[9], out=out[9]);
    And(a=a[10], b=b[10], out=out[10]);
    And(a=a[11], b=b[11], out=out[11]);
    And(a=a[12], b=b[12], out=out[12]);
    And(a=a[13], b=b[13], out=out[13]);
    And(a=a[14], b=b[14], out=out[14]);
    And(a=a[15], b=b[15], out=out[15]);
}

/**
 * Demultiplexor:
 * {a, b} = {in, 0} if sel == 0
 *          {0, in} if sel == 1
 */

CHIP DMux {
    IN in, sel;
    OUT a, b;

    PARTS:
    // Put your code here:
    Not(in=sel, out=notsel);
    And(a=in, b=notsel, out=a);

    And(a=sel, b=in, out=b);
}


/**
 * 4-way demultiplexor:
 * {a, b, c, d} = {in, 0, 0, 0} if sel == 00
 *                {0, in, 0, 0} if sel == 01
 *                {0, 0, in, 0} if sel == 10
 *                {0, 0, 0, in} if sel == 11
 */

CHIP DMux4Way {
    IN in, sel[2];
    OUT a, b, c, d;

    PARTS:
    // Put your code here:
    // NOTE [0000] -> [3210]
    DMux(in=in, sel=sel[1], a=pa, b=pc);
    DMux(in=pa, sel=sel[0], a=a, b=b);
    DMux(in=pc, sel=sel[0], a=c, b=d);
}

/**
 * 8-way demultiplexor:
 * {a, b, c, d, e, f, g, h} = {in, 0, 0, 0, 0, 0, 0, 0} if sel == 000
 *                            {0, in, 0, 0, 0, 0, 0, 0} if sel == 001
 *                            etc.
 *                            {0, 0, 0, 0, 0, 0, 0, in} if sel == 111
 */

CHIP DMux8Way {
    IN in, sel[3];
    OUT a, b, c, d, e, f, g, h;

    PARTS:
    // Put your code here:
    DMux4Way(in=in, sel=sel[1..2], a=pa, b=pc, c=pe, d=pg);
    DMux(in=pa, sel=sel[0], a=a, b=b);
    DMux(in=pc, sel=sel[0], a=c, b=d);
    DMux(in=pe, sel=sel[0], a=e, b=f);
    DMux(in=pg, sel=sel[0], a=g, b=h);
}

/** 
 * Multiplexor:
 * out = a if sel == 0
 *       b otherwise
 */

CHIP Mux {
    IN a, b, sel;
    OUT out;

    PARTS:
    // Put your code here:
    And(a=sel, b=b, out=selandb);
    Or(a=selandb, b=a, out=selandbanda);

    Not(in=sel, out=notsel);
    And(a=notsel, b=a, out=notselanda);
    Or(a=notselanda, b=b, out=notselandaorb);

    And(a=selandbanda, b=notselandaorb, out=out);
}

/**
 * 16-bit multiplexor: 
 * for i = 0..15 out[i] = a[i] if sel == 0 
 *                        b[i] if sel == 1
 */

CHIP Mux16 {
    IN a[16], b[16], sel;
    OUT out[16];

    PARTS:
    // Put your code here:
    Mux(a=a[0], b=b[0], sel=sel, out=out[0]);
    Mux(a=a[1], b=b[1], sel=sel, out=out[1]);
    Mux(a=a[2], b=b[2], sel=sel, out=out[2]);
    Mux(a=a[3], b=b[3], sel=sel, out=out[3]);
    Mux(a=a[4], b=b[4], sel=sel, out=out[4]);
    Mux(a=a[5], b=b[5], sel=sel, out=out[5]);
    Mux(a=a[6], b=b[6], sel=sel, out=out[6]);
    Mux(a=a[7], b=b[7], sel=sel, out=out[7]);
    Mux(a=a[8], b=b[8], sel=sel, out=out[8]);
    Mux(a=a[9], b=b[9], sel=sel, out=out[9]);
    Mux(a=a[10], b=b[10], sel=sel, out=out[10]);
    Mux(a=a[11], b=b[11], sel=sel, out=out[11]);
    Mux(a=a[12], b=b[12], sel=sel, out=out[12]);
    Mux(a=a[13], b=b[13], sel=sel, out=out[13]);
    Mux(a=a[14], b=b[14], sel=sel, out=out[14]);
    Mux(a=a[15], b=b[15], sel=sel, out=out[15]);
}


/**
 * 4-way 16-bit multiplexor:
 * out = a if sel == 00
 *       b if sel == 01
 *       c if sel == 10
 *       d if sel == 11
 */

CHIP Mux4Way16 {
    IN a[16], b[16], c[16], d[16], sel[2];
    OUT out[16];

    PARTS:
    // Put your code here:
    Mux16(a=a, b=b, sel=sel[0], out=pa);
    Mux16(a=c, b=d, sel=sel[0], out=pc);
    Mux16(a=pa, b=pc, sel=sel[1], out=out);
}

/**
 * 8-way 16-bit multiplexor:
 * out = a if sel == 000
 *       b if sel == 001
 *       etc.
 *       h if sel == 111
 */

CHIP Mux8Way16 {
    IN a[16], b[16], c[16], d[16],
       e[16], f[16], g[16], h[16],
       sel[3];
    OUT out[16];

    PARTS:
    // Put your code here:
    Mux4Way16(a=a, b=b, c=c, d=d, sel=sel[0..1], out=pa);
    Mux4Way16(a=e, b=f, c=g, d=h, sel=sel[0..1], out=pb);
    Mux16(a=pa, b=pb, sel=sel[2], out=out);
}

/**
 * Not gate:
 * out = not in
 */

CHIP Not {
    IN in;
    OUT out;

    PARTS:
    // Put your code here:
    Nand(a=in, b=in, out=out);
}

/**
 * 16-bit Not:
 * for i=0..15: out[i] = not in[i]
 */

CHIP Not16 {
    IN in[16];
    OUT out[16];

    PARTS:
    // Put your code here:
    Nand(a=in[0], b=in[0], out=out[0]);
    Nand(a=in[1], b=in[1], out=out[1]);
    Nand(a=in[2], b=in[2], out=out[2]);
    Nand(a=in[3], b=in[3], out=out[3]);
    Nand(a=in[4], b=in[4], out=out[4]);
    Nand(a=in[5], b=in[5], out=out[5]);
    Nand(a=in[6], b=in[6], out=out[6]);
    Nand(a=in[7], b=in[7], out=out[7]);
    Nand(a=in[8], b=in[8], out=out[8]);
    Nand(a=in[9], b=in[9], out=out[9]);
    Nand(a=in[10], b=in[10], out=out[10]);
    Nand(a=in[11], b=in[11], out=out[11]);
    Nand(a=in[12], b=in[12], out=out[12]);
    Nand(a=in[13], b=in[13], out=out[13]);
    Nand(a=in[14], b=in[14], out=out[14]);
    Nand(a=in[15], b=in[15], out=out[15]);
}

 /**
 * Or gate:
 * out = 1 if (a == 1 or b == 1)
 *       0 otherwise
 */

CHIP Or {
    IN a, b;
    OUT out;

    PARTS:
    // Put your code here:
    Not(in=a, out=nota);
    Not(in=b, out=notb);
    Nand(a=nota, b=notb, out=out);
}


/**
 * 16-bit bitwise Or:
 * for i = 0..15 out[i] = (a[i] or b[i])
 */

CHIP Or16 {
    IN a[16], b[16];
    OUT out[16];

    PARTS:
    // Put your code here:
    Or(a=a[0], b=b[0], out=out[0]);
    Or(a=a[1], b=b[1], out=out[1]);
    Or(a=a[2], b=b[2], out=out[2]);
    Or(a=a[3], b=b[3], out=out[3]);
    Or(a=a[4], b=b[4], out=out[4]);
    Or(a=a[5], b=b[5], out=out[5]);
    Or(a=a[6], b=b[6], out=out[6]);
    Or(a=a[7], b=b[7], out=out[7]);
    Or(a=a[8], b=b[8], out=out[8]);
    Or(a=a[9], b=b[9], out=out[9]);
    Or(a=a[10], b=b[10], out=out[10]);
    Or(a=a[11], b=b[11], out=out[11]);
    Or(a=a[12], b=b[12], out=out[12]);
    Or(a=a[13], b=b[13], out=out[13]);
    Or(a=a[14], b=b[14], out=out[14]);
    Or(a=a[15], b=b[15], out=out[15]);
}

/**
 * 8-way Or: 
 * out = (in[0] or in[1] or ... or in[7])
 */

CHIP Or8Way {
    IN in[8];
    OUT out;

    PARTS:
    // Put your code here:
    Or(a=in[0], b=in[1], out=p1);
    Or(a=p1, b=in[2], out=p2);
    Or(a=p2, b=in[3], out=p3);
    Or(a=p3, b=in[4], out=p4);
    Or(a=p4, b=in[5], out=p5);
    Or(a=p5, b=in[6], out=p6);
    Or(a=p6, b=in[7], out=out);
}

/**
 * Exclusive-or gate:
 * out = not (a == b)
 */

CHIP Xor {
    IN a, b;
    OUT out;

    PARTS:
    // Put your code here:
    Not(in=a, out=nota);
    Not(in=b, out=notb);
    And(a=nota, b=b, out=notaandb);
    And(a=a, b=notb, out=aandnotb);
    Or(a=notaandb, b=aandnotb, out=out);
}

```
