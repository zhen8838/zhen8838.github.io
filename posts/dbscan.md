---
title: DBSCAN算法原理及实现
date: 2018-10-30 13:02:47
mathjax: true
tags:
-   聚类方法
categories:
-   机器学习
---


因为模式识别需要分组讲一个聚类算法，所以我挑选了这个算法。

<!--more-->


# 介绍
DBSCAN(Density-Based Spatial Clustering of Applications with Noise，具有噪声的基于密度的聚类方法)是一种很典型的密度聚类算法，和K-Means，BIRCH这些一般只适用于凸样本集的聚类相比，DBSCAN既可以适用于凸样本集，也可以适用于非凸样本集。下面我们就对DBSCAN算法的原理做一个总结。

# 原理

DBSCAN是一种基于密度的聚类算法，这类密度聚类算法一般假定类别可以通过样本分布的紧密程度决定。同一类别的样本，他们之间的紧密相连的，也就是说，在该类别任意样本周围不远处一定有同类别的样本存在。

通过将紧密相连的样本划为一类，这样就得到了一个聚类类别。通过将所有各组紧密相连的样本划为各个不同的类别，则我们就得到了最终的所有聚类类别结果。

# DBSCAN密度定义

DBSCAN类的重要参数也分为两类，一类是DBSCAN算法本身的参数，一类是最近邻度量的参数，下面我们对这些参数做一个总结。

1.   **eps**： DBSCAN算法参数，即我们的ϵ-邻域的距离阈值，和样本距离超过ϵ的样本点不在ϵ-邻域内。默认值是0.5.一般需要通过在多组值里面选择一个合适的阈值。eps过大，则更多的点会落在核心对象的ϵ-邻域，此时我们的类别数可能会减少， 本来不应该是一类的样本也会被划为一类。反之则类别数可能会增大，本来是一类的样本却被划分开。

2.  **MinPts**： DBSCAN算法参数，即样本点要成为核心对象所需要的ϵ-邻域的样本数阈值。默认值是5. 一般需要通过在多组值里面选择一个合适的阈值。通常和eps一起调参。在eps一定的情况下，MinPts过大，则核心对象会过少，此时簇内部分本来是一类的样本可能会被标为噪音点，类别数也会变多。反之MinPts过小的话，则会产生大量的核心对象，可能会导致类别数过少。

3.  **metric**：最近邻距离度量参数。可以使用的距离度量较多，一般来说DBSCAN使用默认的欧式距离（即p=2的闵可夫斯基距离）就可以满足我们的需求。可以使用的距离度量参数有许多。这里我采用欧式距离$ \sqrt{\sum\limits_{i=1}^{n}(x_i-y_i)^2} $

# DBSCAN图例
![eps](./dbscan/eps.png)



# C++实现之错误案例


```cpp

#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>
using namespace std;
#define Dcout(x) cout << #x << ": " << (x) << endl
struct Node {
    double x, y;
    int pts, clusters, belongto; /* 记录相邻点的个数  所属簇 */
    enum { Core= 0, Border= 1, Noise= 2 } tag; /* 初始都定义为噪声点 */
    Node() : x(0), y(0), pts(1), clusters(0), tag(Noise) {}
    /* 默认簇0为噪声簇 */
    Node(double a, double b) : x(a), y(b), pts(1), clusters(0), tag(Noise) {}
};
class DBSCAN {
  public:
    vector<set<int> *> cluster; /* 簇数组>=1 */
    vector<struct Node> a_node; /* 所有点数组 */
    // set<int> c_node;            /* 核心点数组 */
    DBSCAN(double eps, int minpts, vector<struct Node> &l)
        : _eps(eps), _minpts(minpts) {
        _total= l.size();
        a_node= l;
        dis= new double *[_total]; /* 申请内存 */
        for (int i= 0; i < _total; ++i) { dis[i]= new double[_total]; }
        cluster.push_back(new set<int>); /* 簇数组必定大于1 */
    }

    ~DBSCAN() {
        delete[] dis;
        for (auto &it : cluster) { delete it; }
    }
    /* @brief 找到所有核心点
     *
     * */
    void FindCore(void) {
        for (int i= 0; i < _total; ++i) {
            for (int j= i + 1; j < _total; ++j) {
                dis[i][j]= EuclideanDis(a_node[i], a_node[j]);
                if (dis[i][j] <= _eps) { /* 距离小于设定值 */
                    a_node[i].pts++;     /* 点计数加一 */
                    a_node[j].pts++;
                    /* 设置tag */
                    SetTag(i);
                    SetTag(j);
                }
            }
        }
    }
    /* @brief 将核心点联通
     *
     * */
    void ConncetCore(void) {
        for (int i= 0; i < _total; ++i) {
            for (int j= i + 1; j < _total; ++j) {
                if (a_node[i].tag == Node::Core &&
                    a_node[j].tag == Node::Core) {
                    if (dis[i][j] <= _eps) {
                        /* 如果核心点联通就添加到同一簇中 */
                        InsertCluster(i, j);
                    }
                }
            }
        }
    }
    /* @brief 添加标签
     *
     * */
    void AddLabel(void) {
        for (int i= 0; i < cluster.size(); ++i) {
            for (auto j= cluster[i]->begin(); j != cluster[i]->end(); ++j) {
                a_node[*j].clusters= i + 1; /* 为点打标签 */
            }
        }
    }

  private:
    double _eps;         /*  半径 */
    int _total, _minpts; /* 总数 最小密度  */
    double **dis;        /* 距离数组 */

    /* @brief 求解欧氏距离
     *
     * */
    inline double EuclideanDis(const struct Node &a, const struct Node &b) {
        return sqrt(((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y)));
    }
    inline void SetTag(int &a) {
        if (a_node[a].pts > 0 && a_node[a].pts < _minpts) {
            a_node[a].tag= Node::Border;
        }
        if (a_node[a].pts >= _minpts) { a_node[a].tag= Node::Core; }
    }
    inline void InsertCluster(const int &a, const int &b) {
        for (size_t i= 0; i < cluster.size(); ++i) {
            if (cluster[i]->empty()) { /* 此集合为空直接加入 */
                cluster[i]->insert(a);
                cluster[i]->insert(b);
                return;
            }
            if (cluster[i]->count(a) || cluster[i]->count(b)) {
                /* 找到元素直接都加入 */
                cluster[i]->insert(a);
                cluster[i]->insert(b);
                return;
            }
        }
        /* 如果之前的簇中都没有相通的 */
        cluster.push_back(new set<int>);
        cluster.back()->insert(a);
        cluster.back()->insert(b);
    }
};

int main(int argc, char const *argv[]) {
    vector<struct Node> nodes;
    ifstream fin("in"); //读取文件
    if (!fin) { return -1; }
    char s[100];
    for (double x, y; fin.getline(s, 100);) {
        sscanf(s, "%lf,%lf", &x, &y);
        nodes.push_back(Node(x, y));
    }

    DBSCAN dbscan(0.1, 10, nodes); /* 设置eps 以及 minpts */
    dbscan.FindCore();             /* 寻找核心节点 */

    dbscan.ConncetCore(); /* 联通核心点 */
    dbscan.AddLabel();    /* 添加类别标签 */

    ofstream fout("out"); //创建一个输出的文件
    if (!fout) { return -1; }
    for (auto &it : dbscan.a_node) { //将变量的值写入文件
        if (it.tag == Node::Border) {
            it.clusters= dbscan.a_node[it.belongto].clusters;
        }
        fout << it.clusters << endl;
    }
    fout.close(); //关闭文件
    return 0;
}

```



# 错误图像
![eps](dbscan/error.png)
可以发现当前参数被分为多个类。我找了半天才发现，原来是我加入簇中的时候，只循环遍历了所有簇中有没有该元素，但是忽略了一点，有可能此前的两个簇，都是互不相交的。但是到最后的两个元素，另其突然相交了。所以需要修改加入簇的操作。


# 正确程序

```cpp
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <stack>
#include <vector>
using namespace std;
#define Dcout(x) cout << #x << ": " << (x) << endl
struct Node {
    double x, y;
    int pts, cluster; /* 记录相邻点的个数  所属簇 */
    bool visited= false;
    vector<int> contains;
    enum { Core= 0, Border= 1, Noise= 2 } tag; /* 初始都定义为噪声点 */
    Node() : x(0), y(0), pts(1), cluster(0), tag(Noise) {}
    /* 默认簇0为噪声簇 */
    Node(double a, double b) : x(a), y(b), pts(1), cluster(-1), tag(Noise) {}
    Node(double a, double b, int c)
        : x(a), y(b), pts(1), cluster(c), tag(Noise) {}
};
class DBSCAN {
  public:
    vector<set<int> *> cluster; /* 簇数组>=1 */
    vector<Node> a_node;        /* 所有点数组 */
    vector<Node *> c_node;      /* 核心点数组 */
    DBSCAN(double eps, int minpts, vector<struct Node> &l)
        : _eps(eps), _minpts(minpts) {
        _total= l.size();
        a_node= l;
        dis= new double *[_total]; /* 申请内存 */
        for (int i= 0; i < _total; ++i) { dis[i]= new double[_total]; }
    }

    ~DBSCAN() { delete[] dis; }
    /* @brief 找到所有核心点
     *
     * */
    void FindCore(void) {
        for (int i= 0; i < _total; ++i) {
            for (int j= i + 1; j < _total; ++j) {
                dis[i][j]= EuclideanDis(a_node[i], a_node[j]);
                if (dis[i][j] <= _eps) { /* 距离小于设定值 */
                    a_node[i].pts++;     /* 点计数加一 */
                    a_node[j].pts++;
                }
            }
        }
        for (int i= 0; i < _total; ++i) { /* 核心点加入核心点数组 */
            if (a_node[i].pts >= _minpts) {
                a_node[i].tag= Node::Core;
                c_node.push_back(&a_node[i]);
            }
        }
    }
    /* @brief 将核心点联通
     *
     * */
    void ConnectCore(void) {
        for (int i= 0; i < c_node.size(); ++i) {
            for (int j= i + 1; j < c_node.size(); ++j) {
                /* 将所有直接相通的core连接 */
                if (EuclideanDis(*c_node[i], *c_node[j]) < _eps) {
                    c_node[i]->contains.push_back(j);
                    c_node[j]->contains.push_back(i);
                }
            }
        }
    }
    void DFSCore(void) {
        /* 使用DFS进行遍历并加入簇中 */
        int cnt= -1;
        for (int i= 0; i < c_node.size(); i++) {
            stack<Node *> ps;
            if (c_node[i]->visited) continue;
            ++cnt;
            a_node[c_node[i]->cluster].cluster= cnt;
            // Dcout(c_node[i]->cluster);
            ps.push(c_node[i]);
            Node *v;
            while (!ps.empty()) {
                v= ps.top();
                v->visited= 1;
                ps.pop();
                /* 这里的有个问题在于起始点不一定是母节点 */
                /* 并且也没有保存母节点，导致母节点没有被分类 */
                for (int j= 0; j < v->contains.size(); j++) {
                    if (c_node[v->contains[j]]->visited) continue;
                    c_node[v->contains[j]]->cluster= cnt;
                    c_node[v->contains[j]]->visited= true;
                    ps.push(c_node[v->contains[j]]);
                }
            }
        }
    }
    void AddBorder(void) {
        /* 最后把边界点也加入簇中 */
        for (int i= 0; i < _total; ++i) {
            if (a_node[i].tag == Node::Core) { continue; }
            if (a_node[i].tag == Node::Noise) { a_node[i].cluster= -1; }
            for (int j= 0; j < c_node.size(); ++j) {
                if (EuclideanDis(a_node[i], *c_node[j]) < _eps) {
                    a_node[i].tag= Node::Border;
                    a_node[i].cluster= c_node[j]->cluster;
                    break;
                }
            }
        }
    }

  private:
    double _eps;         /*  半径 */
    int _total, _minpts; /* 总数 最小密度  */
    double **dis;        /* 距离数组 */

    /* @brief 求解欧氏距离
     *
     * */
    inline double EuclideanDis(const struct Node &a, const struct Node &b) {
        return sqrt(((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y)));
    }
};

int main(int argc, char const *argv[]) {
    vector<struct Node> nodes;
    ifstream fin("in"); //读取文件
    if (!fin) {
        return -1;
    } else {
        int i= -1;
        char s[100];
        for (double x, y; fin.getline(s, 100);) {
            sscanf(s, "%lf,%lf", &x, &y);
            nodes.push_back(Node(x, y, ++i));
        }
    }

    DBSCAN dbscan(0.1, 10, nodes); /* 设置eps 以及 minpts */
    dbscan.FindCore();             /* 寻找核心节点 */
    dbscan.ConnectCore();          /* 连接核心节点 */
    dbscan.DFSCore();              /* DFS遍历并设置类别 */
    dbscan.AddBorder();            /* 将边界点加入 */

    ofstream fout("out"); //创建一个输出的文件
    if (!fout) {
        return -1;
    } else {
        for (auto &it : dbscan.a_node) { //将变量的值写入文件
            fout << it.cluster << endl;
        }
        fout.close(); //关闭文件
    }
    return 0;
}

```



## 思路
现在的思路是将所有的核心节点构造成网络图，通过DFS遍历的方式，确定所有簇


# 正确图像
![true](dbscan/ture.png)


