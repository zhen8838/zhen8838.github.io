---
title: WordCloud库使用
categories:
  - 编程语言
mathjax: true
toc: true
date: 2018-11-20 15:26:47
tags:
- Python
- WordCloud
---

最近在学`Python`，还是比较有意思的，今天试了下`WordCloud`库写了个小程序


<!--more-->

# 程序



```python
import wordcloud as wc
import os


"""
需要解析的字符串如下：
tags: 
-   Linux
categories: 
-   学习 
"""


if __name__ == "__main__":
    dirstr = '/media/zqh/文档/Blog/gitio/source/_posts/'
    dirls = [name for name in os.listdir(
        "/media/zqh/文档/Blog/gitio/source/_posts") if '.md' in name]
    # 获取我的所有tag
    tags = []
    for filename in dirls:
        istag = False
        f = open(dirstr+filename, 'r', encoding='utf-8')
        for lines in f:
            if istag == False:
                if 'tags' in lines:
                    istag = True
                else:
                    continue
            elif istag == True:
                if '---' in lines:
                    istag = False
                elif '-' in lines:
                    tags.append(list(lines.strip().split())[1])
                else:
                    continue

    c = wc.WordCloud(width=600, height=600,
                     font_path="/usr/share/fonts/deepin-font-install/YaHei Consolas Hybrid/YaHei.Consolas.1.11b.ttf")
    c.generate(' '.join(tags))
    c.to_file('/media/zqh/文档/Blog/gitio/source/_posts/wordcloud/1.png')

```




# 效果

![效果图](./wordcloud/1.png)


# 实例2

```Python
import wordcloud
import jieba

if __name__ == "__main__":
    c = wordcloud.WordCloud(width=1000, height=700,
                            font_path="/usr/share/fonts/deepin-font-install/YaHei Consolas Hybrid/YaHei.Consolas.1.11b.ttf", background_color='white')
    f = open('关于实施乡村振兴战略的意见.txt')
    s = f.read()
    f.close()
    words = jieba.lcut(s)
    c.generate(' '.join(words))
    c.to_file('./2.png')

```

# 效果2

![效果2](./wordcloud/2.png)