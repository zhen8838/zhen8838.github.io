---
title: Nand2Tetris week2
mathjax: true
toc: true
categories:
  - 体系结构
date: 2021-05-06 22:33:48
tags:
- Nand2Tetris
---

第二周，主要实现基于boolean的数学运算以及ALU

<!--more-->


# Binary Addition

着就和十进制加法类似，输入a与b，输出加和以及进位，迭代即可。为了实现的简洁，把加法器分为`半加器`（两个数相加输出和与进位）、`全加器`（三个数相加输出和与进位）、`加法器`，

# Negative Numbers

如果我们要用一个位来表示正负，那么想当然用最左边那一位就好了。
```
000  0
001  1
010  2
011  3
100  4
101  5
110  6
111  7
```

那么直观的来做一下：

```
000  0
001  1
010  2
011  3
100  -0
101  -1
110  -2
111  -3
```

我们发现来这里有个不自然的点，就是`000=0,100=-0`，所以我们转变思路,用补码的方式实现负数，这样就避免了重复。
正数部分与负数部分分别表示为：
$$
\begin{aligned}
  \text{pos} &\in [0\ldots 2^{n-1}-1]\\
  \text{neg} &\in [-1\ldots -2^{n-1}] 
\end{aligned}
$$

```
|- 000  0 
| |- 001  1
| | |- 010  2     
| | |  |- 011  3
| | |  |- 100  -4  (4 - 8)
| | |- 101  -3  (5 - 8) 
| |- 110  -2  (6 - 8)
|- 111  -1  (7 - 8)
```
观察以上数列，不难看出负数的计算公式为$x-2^n$

接下来有趣的来了，对于负数的加法我们甚至不需要做什么特别的，按照整数相加的方式可以得到正确的结果:

```
   -1       (7-8)                111
+  -2     + (6-8)              + 110
------  = -------- = ------  =  ------   
   -3       13-16     5-8        1101 = 101 = -3
```

并且对于两数相减，也可以直接用加上一个负数进行等价替换，非常方便。


最后还有就是取负数，可以在上面的表中发现，每个正数的反码等于正数+1的负数，通过简单的数学变换即可证明，这里的$2^n-1$正好就是全1，全1减去x就是x去反，因此-x就是x取反后+1。
$$
\begin{aligned}
  2^n-x = 1 + ((2^n-1) - x)  
\end{aligned}
$$

总之，我们不用设计更多的运算电路去完成减法等操作了。

# ALU

ALU其实比较简单。就是这里不能用规约的写法，对于判断全0可能比较麻烦。所以需要自己编写一些辅助函数。

还有就是不能取内部数据的子集，比较蛋疼。


# project 2



```vhdl

// File name: projects/02/ALU.hdl

/**
 * The ALU (Arithmetic Logic Unit).
 * Computes one of the following functions:
 * x+y, x-y, y-x, 0, 1, -1, x, y, -x, -y, !x, !y,
 * x+1, y+1, x-1, y-1, x&y, x|y on two 16-bit inputs, 
 * according to 6 input bits denoted zx,nx,zy,ny,f,no.
 * In addition, the ALU computes two 1-bit outputs:
 * if the ALU output == 0, zr is set to 1; otherwise zr is set to 0;
 * if the ALU output < 0, ng is set to 1; otherwise ng is set to 0.
 */

// Implementation: the ALU logic manipulates the x and y inputs
// and operates on the resulting values, as follows:
// if (zx == 1) set x = 0        // 16-bit constant
// if (nx == 1) set x = !x       // bitwise not
// if (zy == 1) set y = 0        // 16-bit constant
// if (ny == 1) set y = !y       // bitwise not
// if (f == 1)  set out = x + y  // integer 2's complement addition
// if (f == 0)  set out = x & y  // bitwise and
// if (no == 1) set out = !out   // bitwise not
// if (out == 0) set zr = 1
// if (out < 0) set ng = 1

CHIP ALU {
    IN  
        x[16], y[16],  // 16-bit inputs        
        zx, // zero the x input?
        nx, // negate the x input?
        zy, // zero the y input?
        ny, // negate the y input?
        f,  // compute out = x + y (if 1) or x & y (if 0)
        no; // negate the out output?

    OUT 
        out[16], // 16-bit output
        zr, // 1 if (out == 0), 0 otherwise
        ng; // 1 if (out < 0),  0 otherwise

    PARTS:
    // Put you code here:
    Mux16(a=x, b=false, sel=zx, out=xout);
    Not16(in=xout, out=notxout);
    Mux16(a=xout, b=notxout, sel=nx, out=nxout);

    Mux16(a=y, b=false, sel=zy, out=yout);
    Not16(in=yout, out=notyout);
    Mux16(a=yout, b=notyout, sel=ny, out=nyout);

    Add16(a=nxout, b=nyout, out=addout);
    And16(a=nxout, b=nyout, out=andout);

    Mux16(a=andout, b=addout, sel=f, out=fout);

    Not16(in=fout, out=notfout);
    Mux16(a=fout, b=notfout, sel=no, out=finalout);
    Or16(a=finalout, b=false, out=out);

    // Adder16(a=finalout, b=true, out=o, carry=sign);
    // Not(in=sign, out=zr);
    Or16Way(in=finalout, out=sign);
    Not(in=sign, out=zr);

    Or16(a=finalout, b=false, out[15]=ng);
}
// File name: projects/02/Adder16.hdl

/**
 * Adds two 16-bit values.
 * The most significant carry bit is ignored.
 */

CHIP Add16 {
    IN a[16], b[16];
    OUT out[16];

    PARTS:
    // Put you code here:
    HalfAdder(a=a[0], b=b[0], sum=out[0], carry=ca);
    FullAdder(a=a[1], b=b[1], c=ca, sum=out[1], carry=cb);
    FullAdder(a=a[2], b=b[2], c=cb, sum=out[2], carry=cc);
    FullAdder(a=a[3], b=b[3], c=cc, sum=out[3], carry=cd);
    FullAdder(a=a[4], b=b[4], c=cd, sum=out[4], carry=ce);
    FullAdder(a=a[5], b=b[5], c=ce, sum=out[5], carry=cf);
    FullAdder(a=a[6], b=b[6], c=cf, sum=out[6], carry=cg);
    FullAdder(a=a[7], b=b[7], c=cg, sum=out[7], carry=ch);
    FullAdder(a=a[8], b=b[8], c=ch, sum=out[8], carry=ci);
    FullAdder(a=a[9], b=b[9], c=ci, sum=out[9], carry=cj);
    FullAdder(a=a[10], b=b[10], c=cj, sum=out[10], carry=ck);
    FullAdder(a=a[11], b=b[11], c=ck, sum=out[11], carry=cl);
    FullAdder(a=a[12], b=b[12], c=cl, sum=out[12], carry=cm);
    FullAdder(a=a[13], b=b[13], c=cm, sum=out[13], carry=cn);
    FullAdder(a=a[14], b=b[14], c=cn, sum=out[14], carry=co);
    FullAdder(a=a[15], b=b[15], c=co, sum=out[15], carry=cp);
}
// File name: projects/02/Adder16.hdl

/**
 * Adds two 16-bit values.
 * The most significant carry bit is ignored.
 */

CHIP Adder16 {
    IN a[16], b[16];
    OUT out[16], carry;

    PARTS:
    // Put you code here:
    HalfAdder(a=a[0], b=b[0], sum=out[0], carry=ca);
    FullAdder(a=a[1], b=b[1], c=ca, sum=out[1], carry=cb);
    FullAdder(a=a[2], b=b[2], c=cb, sum=out[2], carry=cc);
    FullAdder(a=a[3], b=b[3], c=cc, sum=out[3], carry=cd);
    FullAdder(a=a[4], b=b[4], c=cd, sum=out[4], carry=ce);
    FullAdder(a=a[5], b=b[5], c=ce, sum=out[5], carry=cf);
    FullAdder(a=a[6], b=b[6], c=cf, sum=out[6], carry=cg);
    FullAdder(a=a[7], b=b[7], c=cg, sum=out[7], carry=ch);
    FullAdder(a=a[8], b=b[8], c=ch, sum=out[8], carry=ci);
    FullAdder(a=a[9], b=b[9], c=ci, sum=out[9], carry=cj);
    FullAdder(a=a[10], b=b[10], c=cj, sum=out[10], carry=ck);
    FullAdder(a=a[11], b=b[11], c=ck, sum=out[11], carry=cl);
    FullAdder(a=a[12], b=b[12], c=cl, sum=out[12], carry=cm);
    FullAdder(a=a[13], b=b[13], c=cm, sum=out[13], carry=cn);
    FullAdder(a=a[14], b=b[14], c=cn, sum=out[14], carry=co);
    FullAdder(a=a[15], b=b[15], c=co, sum=out[15], carry=carry);
}
// File name: projects/02/FullAdder.hdl

/**
 * Computes the sum of three bits.
 */

CHIP FullAdder {
    IN a, b, c;  // 1-bit inputs
    OUT sum,     // Right bit of a + b + c
        carry;   // Left bit of a + b + c

    PARTS:
    // Put you code here:
    HalfAdder(a=a, b=b, sum=sumab, carry=carryab);
    Xor(a=sumab, b=c, out=sum);
    And(a=a, b=b, out=aband);
    And(a=b, b=c, out=bcand);
    And(a=a, b=c, out=acand);
    Or(a=aband, b=bcand, out=abcor);
    Or(a=abcor, b=acand, out=carry);
}

/**
 * Computes the sum of two bits.
 */

CHIP HalfAdder {
    IN a, b;    // 1-bit inputs
    OUT sum,    // Right bit of a + b 
        carry;  // Left bit of a + b

    PARTS:
    // Put you code here:
    Xor(a=a, b=b, out=sum);
    And(a=a, b=b, out=carry);
}

// File name: projects/02/Inc16.hdl

/**
 * 16-bit incrementer:
 * out = in + 1 (arithmetic addition)
 */

CHIP Inc16 {
    IN in[16];
    OUT out[16];

    PARTS:
   // Put you code here:
    HalfAdder(a=in[0], b=true, sum=out[0], carry=ca);
    FullAdder(a=in[1], b=false, c=ca, sum=out[1], carry=cb);
    FullAdder(a=in[2], b=false, c=cb, sum=out[2], carry=cc);
    FullAdder(a=in[3], b=false, c=cc, sum=out[3], carry=cd);
    FullAdder(a=in[4], b=false, c=cd, sum=out[4], carry=ce);
    FullAdder(a=in[5], b=false, c=ce, sum=out[5], carry=cf);
    FullAdder(a=in[6], b=false, c=cf, sum=out[6], carry=cg);
    FullAdder(a=in[7], b=false, c=cg, sum=out[7], carry=ch);
    FullAdder(a=in[8], b=false, c=ch, sum=out[8], carry=ci);
    FullAdder(a=in[9], b=false, c=ci, sum=out[9], carry=cj);
    FullAdder(a=in[10], b=false, c=cj, sum=out[10], carry=ck);
    FullAdder(a=in[11], b=false, c=ck, sum=out[11], carry=cl);
    FullAdder(a=in[12], b=false, c=cl, sum=out[12], carry=cm);
    FullAdder(a=in[13], b=false, c=cm, sum=out[13], carry=cn);
    FullAdder(a=in[14], b=false, c=cn, sum=out[14], carry=co);
    FullAdder(a=in[15], b=false, c=co, sum=out[15], carry=cp);
}
// File name: projects/01/Or8Way.hdl

/**
 * 8-way Or: 
 * out = (in[0] or in[1] or ... or in[7])
 */

CHIP Or16Way {
    IN in[16];
    OUT out;

    PARTS:
    // Put your code here:
    Or8Way(in=in[0..7], out=a);
    Or8Way(in=in[8..15], out=b);
    Or(a=a, b=b, out=out);
}

```
                              
