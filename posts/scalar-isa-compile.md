---
title: 标量指令集编译器简易实现
mathjax: true
toc: true
categories:
  - 编译器
date: 2022-04-23 13:52:16
tags:
- 指令集
---

之前没有接触过标量isa的编译器该怎么写,所以需要学习一下.
主要参考自RednaxelaFX的[寄存器分配问题](https://www.zhihu.com/question/29355187/answer/51935409)
以及[`chibicc`](https://github.com/rui314/chibicc)简易c编译器.

<!--more-->

# x86 通用寄存器使用建议

| 寄存器 | Callee Save | 描述                                                |
| ------ | ----------- | --------------------------------------------------- |
| %rax   |             | 结果寄存器;同时被用于idiv/imul指令中                |
| %rbx   | yes         | 杂项寄存器                                          |
| %rcx   |             | 第4个参数寄存器                                     |
| %rdx   |             | 第3个参数寄存器; 也被用在idiv / imul指令            |
| %rsp   |             | 栈指针                                              |
| %rbp   | yes         | 帧指针                                              |
| %rsi   |             | 第2个参数寄存器                                     |
| %rdi   |             | 第1个参数寄存器                                     |
| %r8    |             | 第5个参数寄存器                                     |
| %r9    |             | 第6个参数寄存器                                     |
| %r10   |             | 杂项寄存器                                          |
| %r11   |             | 杂项寄存器                                          |
| %r12   | yes         | 杂项寄存器                                          |
| %r13   | yes         | 杂项寄存器                               杂项寄存器 |
| %r14   | yes         | 杂项寄存器                                          |
| %r15   | yes         | 杂项寄存器                                          |


- %rax 通常用于存储函数调用的返回结果，同时也用于乘法和除法指令中。在imul 指令中，两个64位的乘法最多会产生128位的结果，需要 %rax 与 %rdx 共同存储乘法结果，在div 指令中被除数是128 位的，同样需要%rax 与 %rdx 共同存储被除数。
- %rsp指向了内存中堆栈的栈顶,堆栈的 pop 和push 操作就是通过改变 %rsp 的值即移动堆栈指针的位置来实现的。
- %rbp是当前的栈帧指针,标记当前栈帧的起始位置
- **Callee Save**表示当前寄存器的值是`被调用者保存`,也就是发生函数调用的时候,这些寄存器的值在进去子函数后,子函数先保存这些寄存器的值,然后在返回上一级时恢复.
- **Caller Save**表示在进行子函数调用前，就需要由调用者提前保存好这些寄存器的值，保存方法通常是把寄存器的值压入堆栈中，调用者保存完成后，在被调用者（子函数）中就可以随意覆盖这些寄存器的值.

# 基于chibicc检查具体行为

我利用[`chibicc`](https://github.com/rui314/chibicc)对一些代码进行编译,然后调试汇编进行检查他的行为.

## 函数调用帧栈指针行为

```c
#include "test.h"

int foo(int a,int b){
  return a + b;
}

void main(){
  foo(1,2);
}
```

### 1. 在main函数中

这里先把imm加载,然后push到栈上,然后再pop到两个寄存器上.开始调用foo


```x86asm
0040117D: 48 C7 C0 02 00 00 00       movq   $0x2, %rax 
00401184: 50                         pushq  %rax
00401185: 48 C7 C0 01 00 00 00       movq   $0x1, %rax
0040118C: 50                         pushq  %rax
0040118D: 48 8D 05 1F 00 00 00       leaq   0x1f(%rip), %rax  ; foo 
00401194: 5F                         popq   %rdi
00401195: 5E                         popq   %rsi
00401196: 49 89 C2                   movq   %rax, %r10
00401199: 48 C7 C0 00 00 00 00       movq   $0x0, %rax
004011A0: 41 FF D2                   callq  *%r10 # 此时rsp = 0xbc10, rbp = 0xbca0
```

#### step 1 调用前

```
%rbp -> | xxx             |  high 
        | xxx             |   ^
%rsp -> | main 函数最后参数 |   |
        | empty           |   |
        | empty           |   |
        | empty           |   |
        | empty           |  low
```

#### step 2 开始调用

`callq  *%r10`后的结果如下:
因为`call`会把`call`下一条指令的地址压到栈上作为return要用的address.

```
%rbp -> | xxx             |  high 
        | xxx             |   ^
        | main 函数最后参数 |   |
%rsp -> | return address  |   |
        | empty           |   |
        | empty           |   |
        | empty           |  low
```

### 2. 在foo函数中


```x86asm
004011B3: 55                         pushq  %rbp # 保存之前的rbp之后, rsp = 0xbc08
004011B4: 48 89 E5                   movq   %rsp, %rbp
004011B7: 48 83 EC 10                subq   $0x10, %rsp
004011BB: 48 89 65 F8                movq   %rsp, -0x8(%rbp)
004011BF: 89 7D F4                   movl   %edi, -0xc(%rbp)
004011C2: 89 75 F0                   movl   %esi, -0x10(%rbp)
004011C5: 48 8D 45 F0                leaq   -0x10(%rbp), %rax
004011C9: 48 63 00                   movslq (%rax), %rax
004011CC: 50                         pushq  %rax
004011CD: 48 8D 45 F4                leaq   -0xc(%rbp), %rax
004011D1: 48 63 00                   movslq (%rax), %rax
004011D4: 5F                         popq   %rdi
004011D5: 01 F8                      addl   %edi, %eax
004011D7: EB 00                      jmp    0x4011d9
004011D9: 48 89 EC                   movq   %rbp, %rsp
004011DC: 5D                         popq   %rbp
004011DD: C3                         retq   
```


#### step 1

`push %rbq`

⚠️ rsp的push是先递减然后修改对应的值!
```
%rbp -> | xxx             |  high 
        | xxx             |   ^
        | xxx             |   |
        | return address  |   |
%rsp -> | old rbp         |   |
        | empty           |   |
        | empty           |  low
```

#### step 2

`movq   %rsp, %rbp`

```
              | xxx             |  high 
              | xxx             |   ^
              | xxx             |   |
              | return address  |   |
%rbp, %rsp -> | old rbp         |   |
              | empty           |   |
              | empty           |  low
```
#### step 3

`subq   $0x10, %rsp`

```
              | xxx             |  high <-┐
              | xxx             |   ^     |
              | xxx             |   |     |
              | return address  |   |     |
%rbp       -> | old rbp         |   | ----┘
              | empty           |   |
              | empty           |   |
              | empty           |   |
%rsp       -> | empty           |  low
```

#### step 4

从寄存器中把参数写入内存. 他这里还有存了一个rsp,可能是有别的用途.?
```x86asm
movq   %rsp, -0x8(%rbp)
movl   %edi, -0xc(%rbp)
movl   %esi, -0x10(%rbp)
```

```
              | xxx             |  high <---┐
              | xxx             |   ^       |
              | xxx             |   |       |
              | return address  |   |       |
%rbp       -> | old rbp         |   | ------┘
              | empty           |   |
              | old rsp         |   | ------┐
              | 1  (arg 0)      |   |       ⏐
%rsp       -> | 2  (arg 1)      |  low <----┘
```

#### step 5

`addl   %edi, %eax`这里把计算结果存入`eax`,`eax`是`rax`的一半.

#### step 6

```x86asm
004011D7: EB 00                      jmp    0x4011d9
004011D9: 48 89 EC                   movq   %rbp, %rsp
```

返回时, 先jmp到return的位置, 然后rsp指向当前帧顶部:

```
              | xxx             |  high <---┐
              | xxx             |   ^       |
              | xxx             |   |       |
              | return address  |   |       |
%rbp %rsp  -> | old rbp         |   | ------┘
              | empty           |   |
              | old rsp         |   | ------┐
              | 1  (arg 0)      |   |       ⏐
              | 2  (arg 1)      |  low <----┘
```

#### step 7

```x86asm
004011DC: 5D                         popq   %rbp
```
接下来恢复rbp到上一帧的栈顶, 此时`rsp`指向返回地址.

```
%rbp  ->    | xxx             |  high <---┐
            | xxx             |   ^       |
            | xxx             |   |       |
%rsp  ->    | return address  |   |       |
            | old rbp         |   | ------┘
            | empty           |   |
            | old rsp         |   | ------┐
            | 1  (arg 0)      |   |       ⏐
            | 2  (arg 1)      |  low <----┘
```

#### step 8

```x86asm
004011DD: C3                         retq 
```

return实际上是先推出rsp的中的值,然后根据此地址进行跳转.这里的return address就是之前call的下一条指令.

```
%rbp  ->    | xxx             |  high <---┐
            | xxx             |   ^       |
%rsp  ->    | xxx             |   |       |
            | return address  |   |       |
            | old rbp         |   | ------┘
            | empty           |   |
            | old rsp         |   | ------┐
            | 1  (arg 0)      |   |       ⏐
            | 2  (arg 1)      |  low <----┘
```

## 函数通过栈传参行为分析

在x86中,通常通过6个寄存器进行int类型参数传递,分别是`rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9`. 如果是浮点类型的参数,利用的是8个浮点寄存器.当参数为大的结构体/联合体,或者参数个数超过寄存器能容纳的数量时, 将通过栈传递参数.

栈传递参数是在caller中进行的, 将函数参数从右到左的压到栈上(便于支持变长参数):
```
%rbp -> | xxx             |  high 
        | xxx             |   ^
        | callee arg 9    |   |
        | callee arg 8    |   |
%rsp -> | callee arg 7    |   |
        | empty           |   |
        | empty           |   |
        | empty           |   |
        | empty           |  low
```

压完栈之后进入函数中后,帧栈位置如下:


```
%rbp -> | xxx             |  high <---┐
        | xxx             |   ^       |
        | callee arg 9    |   |       |
        | callee arg 8    |   |       |
        | callee arg 7    |   |       |
        | return address  |   |       |
%rsp -> | old rbp         |   | ------┘
        | empty           |   |
        | empty           |  low
```

在代码生成前我们就需要确定所有的参数是通过寄存器传递还是栈传递,因此在子函数中获取local var只需要给出之前分配变量位置时设定的偏移即可.

同时要注意,结构体的压栈顺序也是倒序的,例如结构体如下:
```c
typedef struct
{
    int n;
    int c;
    int h;
    int w;
} shape_t;

typedef struct
{
    shape_t shape;
    unsigned int addr;
} buffer_t;
```

压栈的时候是先把栈向下到对应位置,然后向上copy, 最终的数据摆放应该是如下的:
```
%rbp -> | xxx             |  high <---┐
        | xxx             |   ^       |
        | buffer.addr     |   |       |
        | buffer.shape.w  |   |       |
        | buffer.shape.h  |   |       |
        | buffer.shape.c  |   |       |
        | buffer.shape.n  |   |       |
        | return address  |   |       |
%rsp -> | old rbp         |   | ------┘
        | empty           |   |
        | empty           |  low
```



## 函数调用相对地址计算

我才发现在调用函数的时候是通过`%rip`寄存器去寻址的,给出如下函数:

```c
int foo(int a) { return a + 1; }
int foo2(int a) { return foo(a) + 2; }

int main() {
  int b = 1;
  foo2(b);
  return 0;
}
```

编译结果:
注意到下面调用函数时使用了`lea foo2(%rip), %rax`来获得对应的地址. 然后我查看了`%rip`的作用是:


>  **The role of the %rip register**
>  The `%rip` register on x86-64 is a special-purpose register that always holds the memory address of the next instruction to execute in the program's code segment. The processor increments `%rip` automatically after each instruction, and control flow instructions like branches set the value of `%rip` to change the next instruction.
>  Perhaps surprisingly, `%rip` also shows up when an assembly program refers to a global variable. See the sidebar under "Addressing modes" below to understand how `%rip`-relative addressing works.

也就是他指向了下一个指令的地址.

```x86asm
main:
  . # 忽略一些指令
  .
  .
  mov $4, %rcx
  lea -4(%rbp), %rdi
  mov $0, %al
  rep stosb
  lea -4(%rbp), %rax
  push %rax
  mov $1, %rax
  pop %rdi
  mov %eax, (%rdi)
  lea -4(%rbp), %rax
  movsxd (%rax), %rax
  push %rax
  lea foo2(%rip), %rax
  pop %rdi
  mov %rax, %r10
  mov $0, %rax
  call *%r10
  add $0, %rsp
  mov $0, %rax
  jmp .L.return.main
  mov $0, %rax
.L.return.main:
  mov %rbp, %rsp
  pop %rbp
  ret
  .globl foo2
  .text
  .type foo2, @function
foo2:
  push %rbp
  mov %rsp, %rbp
  sub $16, %rsp
  mov %rsp, -8(%rbp)
  mov %edi, -12(%rbp)
  mov $2, %rax
  push %rax
  sub $8, %rsp
  lea -12(%rbp), %rax
  movsxd (%rax), %rax
  push %rax
  lea foo(%rip), %rax
  pop %rdi
  mov %rax, %r10
  mov $0, %rax
  call *%r10
  add $8, %rsp
  pop %rdi
  add %edi, %eax
  jmp .L.return.foo2
.L.return.foo2:
  mov %rbp, %rsp
  pop %rbp
  ret
  .globl foo
  .text
  .type foo, @function
foo:
  push %rbp
  mov %rsp, %rbp
  sub $16, %rsp
  mov %rsp, -8(%rbp)
  mov %edi, -12(%rbp)
  mov $1, %rax
  push %rax
  lea -12(%rbp), %rax
  movsxd (%rax), %rax
  pop %rdi
  add %edi, %eax
  jmp .L.return.foo
.L.return.foo:
  mov %rbp, %rsp
  pop %rbp
  ret
```


接下来我再用`gnu as`进行汇编得到:




```x86asm
main:
 push   rbp
 mov    rbp,rsp
 sub    rsp,0xa0
 mov    QWORD PTR [rbp-0x10],rsp
 mov    DWORD PTR [rbp-0xa0],0x0
 mov    DWORD PTR [rbp-0x9c],0x30
 mov    QWORD PTR [rbp-0x98],rbp
 add    QWORD PTR [rbp-0x98],0x10
 mov    QWORD PTR [rbp-0x90],rbp
 add    QWORD PTR [rbp-0x90],0xffffffffffffff78
 mov    QWORD PTR [rbp-0x88],rdi
 mov    QWORD PTR [rbp-0x80],rsi
 mov    QWORD PTR [rbp-0x78],rdx
 mov    QWORD PTR [rbp-0x70],rcx
 mov    QWORD PTR [rbp-0x68],r8
 mov    QWORD PTR [rbp-0x60],r9
 movsd  QWORD PTR [rbp-0x58],xmm0
 movsd  QWORD PTR [rbp-0x50],xmm1
 movsd  QWORD PTR [rbp-0x48],xmm2
 movsd  QWORD PTR [rbp-0x40],xmm3
 movsd  QWORD PTR [rbp-0x38],xmm4
 movsd  QWORD PTR [rbp-0x30],xmm5
 movsd  QWORD PTR [rbp-0x28],xmm6
 movsd  QWORD PTR [rbp-0x20],xmm7
 mov    rcx,0x4
 lea    rdi,[rbp-0x4]
 mov    al,0x0
 rep stos BYTE PTR es:[rdi],al
 lea    rax,[rbp-0x4]
 push   rax
 mov    rax,0x1
 pop    rdi
 mov    DWORD PTR [rdi],eax
 lea    rax,[rbp-0x4]
 movsxd rax,DWORD PTR [rax]
 push   rax
 lea    rax,[rip+0x0]        # b4 <main+0xb4>
 pop    rdi
 mov    r10,rax
 mov    rax,0x0
 call   r10
 add    rsp,0x0
 mov    rax,0x0
 jmp    d6 <main+0xd6>
 mov    rax,0x0
 mov    rsp,rbp
 pop    rbp
 ret    
foo2:
 push   rbp
 mov    rbp,rsp
 sub    rsp,0x10
 mov    QWORD PTR [rbp-0x8],rsp
 mov    DWORD PTR [rbp-0xc],edi
 mov    rax,0x2
 push   rax
 sub    rsp,0x8
 lea    rax,[rbp-0xc]
 movsxd rax,DWORD PTR [rax]
 push   rax
 lea    rax,[rip+0x0]        # 105 <foo2+0x2a>
 pop    rdi
 mov    r10,rax
 mov    rax,0x0
 call   r10
 add    rsp,0x8
 pop    rdi
 add    eax,edi
 jmp    11c <foo2+0x41>
 mov    rsp,rbp
 pop    rbp
 ret    
foo:
 push   rbp
 mov    rbp,rsp
 sub    rsp,0x10
 mov    QWORD PTR [rbp-0x8],rsp
 mov    DWORD PTR [rbp-0xc],edi
 mov    rax,0x1
 push   rax
 lea    rax,[rbp-0xc]
 movsxd rax,DWORD PTR [rax]
 pop    rdi
 add    eax,edi
 jmp    144 <foo+0x23>
 mov    rsp,rbp
 pop    rbp
 ret    
```


上面有个很奇怪的地方,`lea`不是应该得到的是`foo`的地址, 他这里的注释的解释如下:
```x86asm
lea    rax,[rip+0x0]        # b4 <main+0xb4> , b4 是下一个指令的地址, <main + 0xb4>就是main为0,加上偏移b4

lea    rax,[rip+0x0]        # 105 <foo2+0x2a>, 105 是下一个指令的地址, <foo2 + 0x2a>就是foo2为0xdb,加上偏移2a
```
