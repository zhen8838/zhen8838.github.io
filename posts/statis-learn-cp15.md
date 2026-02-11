---
title: 统计学习方法:概率潜在语义模型
mathjax: true
toc: true
categories:
  - 机器学习
date: 2020-07-21 10:04:15
tags:
- 统计学习方法
- 概率论
---

最近因为各种杂事,导致我很难专心做一些事情.我觉得需要反省一下自己.


概率潜在语义分析(probabilistic latent semantic analysis, PLSA),也称概率潜在语义索引(probabilistic latent semantic indexing, PLSI),是一种利用概率生成模型对文本集合进行话题分析的无监督学习方法.


<!--more-->

此方法理解起来并不难,其实和潜在语义模型类似,只不过他的`单词-话题矩阵`和`话题-文本矩阵`在PLSA中是以非负的概率的形式的进行表示.同时我们使用EM算法进行迭代时可忽略话题特征值矩阵.

我将迭代过程绘制在单纯形中,大家可以发现这个还是很有趣的.