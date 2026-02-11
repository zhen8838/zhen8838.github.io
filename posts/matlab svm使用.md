---
title: Matlab svm使用
date: 2018-05-30 16:53:42
categories: 
-   机器学习 
tags: 
-   Matlab
-   SVM
---

这里是对svm的函数做一个使用的总结，为了以后便于翻看。

<!--more-->


#   fitcsvm函数
fitcsvm这个函数是用于训练分类模型的。主要用法有:
1.  `Mdl = fitcsvm(___,Name,Value)`
    这个用法就比较容易理解，例子如下：
    ```matlab
    SVMmodel = fitcsvm(data3,theclass,'KernelFunction','rbf','BoxConstraint',Inf,'ClassNames',[-1,1]);
    ```

    **data3**：
    是一组用于训练的模型数据，一行作为一组数据。

    **theclass**：
    是训练数据的标签数据，每一行对应一行数据。

    **Name**：
    这后面的KernelFunction、BoxConstraint等等都是变量名。

    **Value**：
    这里的值就为变量的值，比如 `'KernelFunction','rbf'`就代表了核函数选择的是rbf核。
    
    **SVMmodel**：
    这个函数的返回值就是一个训练好的模型。

#   fitrgp函数

fitrgp函数是高斯过程拟合函数，我发现这个函数的效果比较不错，所以介绍一下：

1.`gprMdl = fitrgp(X,y)`
    例子如下：
    ```matlab
    regressionGP = fitrgp(trian_x,trian_y,'BasisFunction', 'constant','KernelFunction','matern52','Standardize',true);
    ```

    **trian_x**:
    训练数据，一行作为一组数据。

    **trian_y**：
    测试数据：每一行对应一行数据。
    后面依旧是NAME：Value。再此不赘述。

#   predict函数
predict函数是进行预测的。主要用法有：

1.   `yp = predict(sys,data,K)`
    这个用法的例子如下：
    `yp = predict(datatest,testlabel)`

    **datatest**：
    为用于测试的数据。

    **testlabel**：
    是测试数据的标签。

    **K**：
    这里K没有用到，这个K就是预测地平线，现在我还不知道什么意思，他的默认是1。所以这里没有用到。
    
    **yp**:
    预测的输出响应。

