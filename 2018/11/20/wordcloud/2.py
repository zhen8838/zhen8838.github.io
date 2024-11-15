
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
