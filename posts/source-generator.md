---
title: C# Source Generator使用
mathjax: true
toc: true
categories:
  - 编程语言
date: 2021-10-21 15:43:14
tags:
- CSharp
---

C#中提供一个代码生成的功能,基于模板的生成和这个是没法比的.因为我们是直接调用编译器对当前项目进行分析,然后生成代码.

<!--more-->

# 需求

需要为nncase的IR设计图匹配的功能,我的思路是为每个节点设置对应类型的pattern节点,那么问题就在于IR中有那么多的OP和类型,手动拷贝和修改是低效率和易出错的,因此我需要分析IR的类型然后生成对应的IR Pattern类型.


# 问题

中间遇到了各种各样的问题

## 1. 项目配置

首先对于生成器,我们需要新建一个项目,并且他`TargetFramework`必须设置为`netstandard2.0`.

## 2. 分析指定的项目

我们可以把新建的生成器看作为一个分析器,某一个项目引用他,那么当前的分析器就分析当前项目(并且引用时需要设置属性 `OutputItemType="Analyzer"`). 比如:
比如下面这个例子,我们的Nncase.IR引用了生产器,此时的代码就是基于Nncase.IR分析所产生的.

```sh
|-- Nncase.Pattern.Generator ----
|                               |
|-- Nncase.IR     <--------------
```

## 3. 编写指南

### 3.1 尽量使用`ISyntaxContextReceiver`.

我们可以直接通过`ctx.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.IR.Expr")`获取一些重要的基类对象的symbol.

### 3.2 尽量使用`SemanticModel`

比如我们从receiver获得到语法节点后,转换为symbol可以更加方便的获取类型的属性/继承/接口/类型等等信息.

```csharp
// 0. check inherit from base class;
if (classSymbol.BaseType is not { IsGenericType: true, Name: "RewriteRule" })
    Diagnostics.Add(Diagnostic.Create(RecriverUtil.ClassNotFromBaseClassError, Location.None, classSymbol.ToDisplayString(), "RewriteRule"));

// 1. check is Partial
if (!classDeclaration.Modifiers.Any(tok => tok.IsKind(SyntaxKind.PartialKeyword)))
    Diagnostics.Add(Diagnostic.Create(RecriverUtil.ClassNotPartialError, Location.None, classSymbol.ToDisplayString()));
```

### 3.3 如何构建正确的语法树

因为他的api太多太杂, 每个语法节点都有自己类型,然后方法多的根本不知道用哪个. 所以需要用visual studio,先用syntax visualizer看一下代码的语法树再分析.

或者使用`https://roslynquoter.azurewebsites.net`.


## 4. Debug代码生成过程

务必请使用vs 2022, 我录制了一个[启动调试的视频](https://github.com/dotnet/roslyn-sdk/issues/850#issuecomment-1038725567).



## 5. 直接分析整个项目生成代码

后来发现内置的源代码生成有个问题就是他只能对当前项目进行分析,生成的代码也只能在这个项目中,但是我这里的需求是分析的项目a和b为c生成代码,这样的话必须要abc同时依赖代码分析器了. 会造成项目结构混乱的问题. 后面我是重新写了一个第三方项目,直接分析整个sln, 然后生成文件的形式. 代码放在了 https://github.com/zhen8838/PatternSourceGenerator.0

不过我现在发现其实可以先得到symbol信息