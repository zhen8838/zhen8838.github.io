---
title: Python格式化配置
mathjax: true
toc: true
categories:
  - 编程语言
date: 2020-04-27 00:27:01
tags:
- Python
---

之前一直用`autopep8`作为格式化公式，后来发现不能设置缩进两个空格就换成了`yapf`。但`yapf`实在是一言难尽，我不太喜欢这种整个文档都帮你格式化的，`autopep8`这样可以只考虑每一行的内部的格式化就足够了，可以给程序员更多的调整空间。



<!--more-->

今天本来想说哪怕`autopep8`不支持缩进两个空格也要给他加上去，但是找了一番发现他虽然帮助文档里面没有写，但是实际上指定了缩进参数也是没问题的。所以给出一个`vscode`的配置：

```json
"python.formatting.autopep8Args": [
    "--indent-size=2",
    "-j=2",
    "--max-line-length=100",
],
"python.formatting.provider": "autopep8",
```