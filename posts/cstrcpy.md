---
title: c与c++字符串赋值
date: 2018-08-06 13:57:07
categories: 
-   编程语言
tags:
-   C
-   CPP
-   踩坑经验
---


最近用c++写的一个程序，我想用一个`const char *p`对一个`char head[2]`赋值，我使用`strcpy`赋值之后一直出现错误。我就写了个小程序去验证了一下。

<!--more-->


# 程序


```cpp
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
class CHARGET {
  private:
    char _head[2];

  public:
    CHARGET(const char *p, int opt) {
        memset(_head, 0, 2);
        if (opt == 0) {
            _head[0] = *p++;
            _head[1] = *p++;
        } else if (opt == 1) {
            strcpy(_head, p);
        } else if (opt == 2) {
            strncpy(_head, p, 2);
        } else if (opt == 3) {
            strncpy(_head, p, 3);
        } else {
            return;
        }
    }

    void show(void) { printf("%s%.2d", _head, 2); }
    ~CHARGET() {}
};

char test[2];

void CHARCPY(const char *p) {
    test[0] = *p++;
    test[1] = *p++;
}

int main(int argc, char const *argv[]) {
    /* 类中 直接指针幅值 */
    printf("类中直接指针幅值\r\n");
    CHARGET c1("HJ", 0);
    c1.show();
    printf("\n\n");

    /* strncpy 两字节 */
    printf("strcpy\r\n");
    CHARGET c2("HJ", 1);
    c2.show();
    printf("\n\n");

    /* strncpy 两字节 */
    printf("strncpy 两字节\r\n");
    CHARGET c3("HJ", 2);
    c3.show();
    printf("\n\n");

    /* strncpy 三字节 */
    printf("strncpy 三字节\r\n");
    CHARGET c4("HJ", 3);
    c4.show();
    printf("\n\n");

    /* 类外 直接指针幅值 */
    printf("类外直接指针幅值\r\n");
    CHARCPY("HJ");
    printf("%s%.2d", test, 2);
    printf("\n\n");

    return 0;
}
```



# 运行结果

```sh
类中直接指针幅值
HJ?T�+�02

strcpy
HJ02

strncpy 两字节
HJHJ02

strncpy 三字节
HJ02

类外直接指针幅值
HJ02

```

# 分析

- **类中指针赋值**

    直接通过指针进行赋值：
    
    ```c
        _head[0] = *p++;
        _head[1] = *p++;
    ```
    

    输出：
    ```sh
    HJ?T�+�02
    ```

    `HJ`之后带有一串乱码，应该是由于c++在指针赋值时没有将第三位赋值为`\0`，导致打印输出时识别不到结束符一直输出。

-   **strcpy赋值**

    直接用`strcpy`，看起来好像还行，但是到了下一步就发现问题了。

-   **strncpy两字节**

    再新建一个类，然后对该类的`_head`进行`strncpy`两个字节。

    输出：
    ```sh
    HJHJ02
    ```
    应该是由于`strncpy`也没有在最后添加`\0`，并且新创建的变量的地址正好在之前变量的地址之前，最后出现了这个现象。

-   **strncpy三字节**

    再新建一个类，然后对该类的`_head`进行`strncpy`三个字节。

    输出:
    ```sh
    HJ02
    ```
    说明这个时候是在字符串结尾添加了`\0`。

-   **类外直接指针幅值**

    不使用类的方式，直接用函数的方式，进行赋值。

    输出：
    ```sh
    HJ02
    ```

    是没有问题的。


