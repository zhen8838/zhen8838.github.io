---
title: Makefile中使用shell赋值变量
categories:
  - 工具使用
date: 2019-04-06 20:11:13
tags:
-   踩坑经验
-   Makefile
---

今天我为了`makefile`的方便起见,将一些变量通过脚本的形式给到`makefile`中,但是通过`shell`命令给`makefile`变量赋值让我头疼了一波.. 🙄

<!--more-->

# shell中执行方式
在`shell`中赋值非常简单
```sh
➜ PRE_CKPT=log/20190405-174159        
➜ NUM=`python3 tools/get_trian_num.py ${PRE_CKPT}`
➜ echo ${NUM}
4700
```

# 在makefile中实现

1.  第一次写法
```makefile
freeze: 
	NUM=`python3 tools/get_trian_num.py ${PRE_CKPT}`
	echo ${NUM}
```
执行,发现什么都没有输出
```sh
➜ make freeze PRE_CKPT=log/20190405-174159
NUM=`python3 tools/get_trian_num.py log/20190405-174159`
echo 
```

这里的问题是,我们通过`shell`命令赋值的是`shell`的变量,这个变量还不是`makefile`的变量.所以我们需要通过`$$VAR`的方式调用这个变量.

2.  第二次写法
```makefile
freeze: 
	NUM=`python3 tools/get_trian_num.py ${PRE_CKPT}`
	echo $$NUM
```
执行,发现还是没有输出
```sh
➜ make freeze PRE_CKPT=log/20190405-174159
NUM=`python3 tools/get_trian_num.py log/20190405-174159`
echo $NUM
```

这里是因为`makefile`中命令如果没有使用`;\`来连接,是无法共享变量的.所以还得修改

3.  第三次写法

```makefile
freeze: 
	@NUM=`python3 tools/get_trian_num.py ${PRE_CKPT}`; \
	echo $$NUM
```
终于有了我想要的输出 😊
```sh
➜  make freeze PRE_CKPT=log/20190405-174159
4700
```