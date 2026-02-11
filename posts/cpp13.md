---
title: c++成员函数中static变量
date: 2018-09-17 11:33:37
tags:
-   CPP
-   踩坑经验
categories: 
-   编程语言
---

我最近在做郭炜老师的编程题目,[这道题](http://cxsjsxmooc.openjudge.cn/2018t3fallall/013/)我实现的过程中出现了一些蛋疼的错误,进行一个记录.

<!--more-->


# 源代码

```cpp
#include <iostream>
#include <vector>
using namespace std;
#define Dcout(x) cout << #x << ": " << (x) << endl

class HeadQuarter {
  public:
    bool Stop= false; /* 停止flag */
    struct Warrior {
        string name;
        int strength;
        int cnt;
    };
    enum camp { red= 0, blue= 1 };
    /* 输入阵营颜色以及各个战士的生命值 */
    HeadQuarter(HeadQuarter::camp color, int HealthPoint, int dragon, int ninja,
                int iceman, int lion, int wolf) {
        /* 生成五个战士信息结构体 */
        Warrior Dragon= {"dragon", dragon, 0}, Ninja= {"ninja", ninja, 0},
                Iceman= {"iceman", iceman, 0}, Lion= {"lion", lion, 0},
                Wolf= {"wolf", wolf, 0};
        /* 分不同阵营设置生产顺序 */
        if (color == camp::red) { /* 红色阵营 */
            Color= "red";         /* 依次加入数组 */
            seqcycle.push_back(Iceman);
            seqcycle.push_back(Lion);
            seqcycle.push_back(Wolf);
            seqcycle.push_back(Ninja);
            seqcycle.push_back(Dragon);
        } else { /* 蓝色阵营 */
            Color= "blue";
            seqcycle.push_back(Lion);
            seqcycle.push_back(Dragon);
            seqcycle.push_back(Ninja);
            seqcycle.push_back(Iceman);
            seqcycle.push_back(Wolf);
        }
        /* 初始化所有计数器 */
        HP= HealthPoint;
        Warriorcnt= 0;
        TimeCount= 0;
        /* 寻找到最小消耗 */
        for (auto &&it : seqcycle) {
            MinHp= min(MinHp, it.strength);
        }
    }
    /* 产生战士函数 */
    void GenWarrior() {
        /* 定位 */
        static auto index= seqcycle.begin();
        /* index 循环函数 */
        auto indexloop= [this]() {
            index= index == seqcycle.end() ? seqcycle.begin() : index + 1;
        };
        /* 至少可以制造一个 */
        if (HP > MinHp) {
            /* 找到可以制造的那个 */
            for (; HP < index->strength; indexloop()) {
            }
            /* 开始制造 */
            printf("%03d %s %s %d born with strength %d,%d %s in %s "
                   "headquarter\n",
                   TimeCount++, Color.c_str(), index->name.c_str(),
                   ++Warriorcnt, index->strength, ++index->cnt,
                   index->name.c_str(), Color.c_str());
            HP-= index->strength; /* 减去生命值 */
            indexloop();          /* 循环 */
        } else {
            if (!Stop) { /* 开始无法制造 */
                printf("%03d %s headquarter stops making warriors\n", TimeCount,
                       Color.c_str());
                Stop= true;
            }
        }
    }

  private:
    vector<Warrior> seqcycle; /* 生产序列 */
    int HP;                   /* 基地生命元 */
    string Color;             /* 阵营颜色 */
    int TimeCount;            /* 时间计数器 */
    int Warriorcnt;           /* 战士计数器 */
    int MinHp= 10000;         /* 最小的生成消耗生命值 */
};

#define BLUE
#define RED

int main(int argc, char const *argv[]) {
    int caseNumber= 0;
    int caseHp= 0;
    int casedragon= 0, caseninja= 0, caseiceman= 0, caselion= 0, casewolf= 0;
    cin >> caseNumber;
    for (int i= 1; i <= caseNumber; i++) {
        cin >> caseHp;
        cin >> casedragon >> caseninja >> caseiceman >> caselion >> casewolf;
        HeadQuarter *RedCamp=
            new HeadQuarter(HeadQuarter::red, caseHp, casedragon, caseninja,
                            caseiceman, caselion, casewolf);
        HeadQuarter *BlueCamp=
            new HeadQuarter(HeadQuarter::blue, caseHp, casedragon, caseninja,
                            caseiceman, caselion, casewolf);
        cout << "Case:" << caseNumber << endl;
#ifdef RED
        while (RedCamp->Stop != true) {
            RedCamp->GenWarrior();
        }
#endif
#ifdef BLUE
        while (BlueCamp->Stop != true) {
            BlueCamp->GenWarrior();
        }
#endif
        delete RedCamp;
        delete BlueCamp;
    }

    return 0;
}
```



# 样例输入
首先我通过宏控制程序编译出三个版本`all.out`,`blue.out`,`red.out`,其中`all.out`是全部输出,另外两个是单独输出红色或者蓝色.

样例输入:
```
1
20
3 4 5 6 7
```
样例输出:
```
Case:1
000 red iceman 1 born with strength 5,1 iceman in red headquarter
000 blue lion 1 born with strength 6,1 lion in blue headquarter
001 red lion 2 born with strength 6,1 lion in red headquarter
001 blue dragon 2 born with strength 3,1 dragon in blue headquarter
002 red wolf 3 born with strength 7,1 wolf in red headquarter
002 blue ninja 3 born with strength 4,1 ninja in blue headquarter
003 red headquarter stops making warriors
003 blue iceman 4 born with strength 5,1 iceman in blue headquarter
004 blue headquarter stops making warriors
```

# 测试

## 测试红色

```sh
➜  exam cat test.txt | ./red.out
Case:1
000 red iceman 1 born with strength 5,1 iceman in red headquarter
001 red lion 2 born with strength 6,1 lion in red headquarter
002 red wolf 3 born with strength 7,1 wolf in red headquarter
003 red headquarter stops making warriors
```

## 测试蓝色

```sh
➜  exam cat test.txt | ./blue.out
Case:1
000 blue lion 1 born with strength 6,1 lion in blue headquarter
001 blue dragon 2 born with strength 3,1 dragon in blue headquarter
002 blue ninja 3 born with strength 4,1 ninja in blue headquarter
003 blue iceman 4 born with strength 5,1 iceman in blue headquarter
004 blue headquarter stops making warriors
```

## 测试全部

```sh
➜  exam cat test.txt | ./all.out
Case:1
000 red iceman 1 born with strength 5,1 iceman in red headquarter
001 red lion 2 born with strength 6,1 lion in red headquarter
002 red wolf 3 born with strength 7,1 wolf in red headquarter
003 red headquarter stops making warriors
000 blue ninja 1 born with strength 4,1 ninja in blue headquarter
001 blue dragon 2 born with strength 3,1 dragon in blue headquarter
002 blue (null) 3 born with strength 0,1 (null) in blue headquarter
003 blue (null) 4 born with strength 0,1 (null) in blue headquarter
004 blue (null) 5 born with strength 0,1 (null) in blue headquarter
[1]    17104 done                cat test.txt |
       17105 segmentation fault  ./all.out
```

# 定位问题
从上面的结果可以看出,单独输出一组没有问题,但是两组一起输出就会出现问题.本来应该是
两个对象单独输出,互不影响,但是现在red对象输出后,blue对象输出就出现了问题.因此我
在代码中加入log语句.

## 测试红色

```sh
➜  exam cat test.txt | ./red.out
Case:1
000 red iceman 1 born with strength 5,1 iceman in red headquarter
index - seqcycle.begin(): 1
001 red lion 2 born with strength 6,1 lion in red headquarter
index - seqcycle.begin(): 2
002 red wolf 3 born with strength 7,1 wolf in red headquarter
index - seqcycle.begin(): 3
003 red headquarter stops making warriors
```
## 测试两组

```sh
➜  exam cat test.txt | ./all.out
Case:1
000 red iceman 1 born with strength 5,1 iceman in red headquarter
index - seqcycle.begin(): 1
000 blue lion 1 born with strength 6,1 lion in blue headquarter
index - seqcycle.begin(): -8
001 red wolf 2 born with strength 7,1 wolf in red headquarter
index - seqcycle.begin(): 3
001 blue ninja 2 born with strength 4,1 ninja in blue headquarter
index - seqcycle.begin(): -6
002 red dragon 3 born with strength 3,1 dragon in red headquarter
index - seqcycle.begin(): 5
002 blue (null) 3 born with strength 0,1 (null) in blue headquarter
index - seqcycle.begin(): -4
003 red (null) 4 born with strength 0,1 (null) in red headquarter
index - seqcycle.begin(): 7
003 blue (null) 4 born with strength 0,1 (null) in blue headquarter
index - seqcycle.begin(): -2
index - seqcycle.begin(): 9
index - seqcycle.begin(): 10
index - seqcycle.begin(): 11
index - seqcycle.begin(): 12
index - seqcycle.begin(): 13
index - seqcycle.begin(): 14
index - seqcycle.begin(): 15
[1]    32721 done                cat test.txt |
       32722 segmentation fault  ./all.out
```

终于看出问题了,应该是我在函数中使用的`static auto index= seqcycle.begin();`,这个静态的`index`是两个类所共有的,每次对`index`进行循环操作`index= index == seqcycle.end() ? seqcycle.begin() : index + 1;`,其中的`seqcycle.begin()`却是每个类独有的,就导致了每次
公共的`index`加1,两次同时操作就导致了`index`最终超出了`seqcycle`的范围,出现了数组越界.


# 解决方案

现在只能在类中定义一个私有变量`index`来对数据进行定位.
