---
title: 代码块自动添加折叠
categories:
  - 工具使用
mathjax: true
toc: true
date: 2018-12-11 16:38:56
tags:
-   Python
---

今天放代码的时候,突然觉得代码~~太烂~~~太长,放在那边就扫了大家浏览的兴致,所以准备给所有的代码段加个折叠块,但是加折叠块必须要每个文件修改,很蛋疼,所以就写了个小工具去自动添加~

**我这个只能添加一次,运行两次就炸了~**

<!--more-->

# fold.py

```python
import os
if __name__ == "__main__":
    path = '/media/zqh/文档/Blog/gitio/source/_posts'  # 文件夹目录
    files = os.listdir(path)  # type:list
    files = [it for it in files if '.md' in it]  # 排除别的类型文件
    startflag = True
    cnt = 1
    for i in range(len(files)):
        with open(path+'/'+files[i], 'r+') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if startflag:
                    if ('```c' in lines[i]) or ('```cpp' in lines[i]) or\
                            ('```python' in lines[i]):
                        pos = lines[i].find('`')
                        strl = list(lines[i])
                        strl.insert(pos, '{% fold 点击显示内容 %}\n'+pos*' ')
                        lines[i] = ''.join(strl)
                        # print(lines[i])
                        startflag = False
                else:
                    if '```' in lines[i]:
                        pos = lines[i].find('`')
                        strl = list(lines[i])
                        strl.insert(-1, '\n'+pos*' '+'{% endfold %}\n')
                        lines[i] = ''.join(strl)
                        # print(lines[i])
                        print('修改了{}处'.format(cnt))
                        cnt += 1
                        startflag = True
            f.seek(0)  # 回到文件头
            f.truncate()  # 清空文件
            # 重新写入文件
            text = ''.join(lines)
            f.write(text)

```
