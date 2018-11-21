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
