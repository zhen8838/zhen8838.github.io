---
title: egg 浅析
mathjax: true
toc: true
categories:
  - 编译器
date: 2022-02-27 13:18:39
tags:
- 中端优化
- Equality Saturation
---

主要分析egraphs-good也就是[egg](https://github.com/egraphs-good/egg)这个库的实现机制.因为最近发现适配到基于relay的ir中存在一些问题,因此还是需要仔细研究一下他的实现细节.

<!--more-->

# 1. Language

Language应该就是代表的是enode, 由一个op以及若干children组成. 他的构建机制每个dsl中自己实现`from_op`函数.
```rust

pub struct Symbol(u32);

pub struct Id(u32);

pub struct SymbolLang {
    /// The operator for an enode
    pub op: Symbol,
    /// The enode's children `Id`s
    pub children: Vec<Id>,
}

impl FromOp for SymbolLang {
    type Error = Infallible;

    fn from_op(op: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        Ok(Self {
            op: op.into(),
            children,
        })
    }
}
```
此时`SymbolLang`中op对应的类型是`symbol (u32)`,他的内部维护了一个string hashset,然后调用`op.into()`从hashset中取得对应的index作为symbol.
这里的children的类型是`id (u32)`, 本意是表示eclass的id, 但如果没有加入egraph之前实际上是共用symobl的值.

# 2. RecExpr<T>

RecExpr表示是一组由用户定义的language组成的递归的expression, 比如我构建输入`a + b`, 那么此时RecExpr的nodes由 `[+, a, b]`,`[a]`,`[b]`三个language节点组成. 即保存了输入表达式下的所有 language node.

```rust
pub struct RecExpr<L> {
    nodes: Vec<L>,
}
```

他是通过递归的parser构建的,每次解析一个 language node然后放入到RecExpr中去.

```rust
impl<L: FromOp> FromStr for RecExpr<L> {
    type Err = RecExprParseError<L::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use RecExprParseError::*;

        fn parse_sexp_into<L: FromOp>(
            sexp: &Sexp,
            expr: &mut RecExpr<L>,
        ) -> Result<Id, RecExprParseError<L::Error>> {
          ...
        }

        let mut expr = RecExpr::default();
        let sexp = symbolic_expressions::parser::parse_str(s.trim()).map_err(BadSexp)?;
        parse_sexp_into(&sexp, &mut expr)?;
        Ok(expr)
    }
}
```

# 3. EClass

EClass定义.

```rust
pub struct EClass<L, D> {
    /// 当前的eclass id
    pub id: Id,
    /// 所有等价的enode.
    pub nodes: Vec<L>,
    /// The analysis data associated with this eclass.
    pub data: D,
    /// eclass的反向连接 (也就是哪些enode使用了当前的eclass, 是为了在repair的时候进行递归).
    pub(crate) parents: Vec<(L, Id)>,
}
```

# 4. EGraph

```rust
pub struct EGraph<L: Language, N: Analysis<L>> {
    /// The `Analysis` given when creating this `EGraph`.
    pub analysis: N,
    /// The `Explain` used to explain equivalences in this `EGraph`.
    pub(crate) explain: Option<Explain<L>>,
    unionfind: UnionFind,
    /// Stores each enode's `Id`, not the `Id` of the eclass.
    /// Enodes in the memo are canonicalized at each rebuild, but after rebuilding new
    /// unions can cause them to become out of date.
    #[cfg_attr(feature = "serde-1", serde(with = "vectorize"))]
    memo: HashMap<L, Id>, // memo 表示的就是hashcon, 即enode -> eclass id
    // pending 表示的是后续需要repair的enode节点和对应原始的eclass id(没有find过).
    pending: Vec<(L, Id)>,
    analysis_pending: IndexSet<(L, Id)>,
    #[cfg_attr(
        feature = "serde-1",
        serde(bound(
            serialize = "N::Data: Serialize",
            deserialize = "N::Data: for<'a> Deserialize<'a>",
        ))
    )]
    // 保存了eclass id 对应 eclass的map, 虽然eclass中已经保存了id, 但是在egraph中加一个字典后续更加方便.
    classes: HashMap<Id, EClass<L, N::Data>>,
    #[cfg_attr(feature = "serde-1", serde(skip))]
    #[cfg_attr(feature = "serde-1", serde(default = "default_classes_by_op"))]
    pub(crate) classes_by_op: HashMap<std::mem::Discriminant<L>, HashSet<Id>>,
    /// 表示当前的egraph是否需要repair, 修复后clean=true.
    #[cfg_attr(feature = "serde-1", serde(skip))]
    pub clean: bool,
}
```

# 5. Rewrite

```rust
pub struct Rewrite<L, N> {
    /// The name of the rewrite.
    pub name: Symbol,
    /// The searcher (left-hand side) of the rewrite.
    pub searcher: Arc<dyn Searcher<L, N> + Sync + Send>,
    /// The applier (right-hand side) of the rewrite.
    pub applier: Arc<dyn Applier<L, N> + Sync + Send>,
}
```

# 5.1 rebuild

egraph 首先添加进入后都是没有clean的,所以需要rebuild一次

# 5.1.2 收集class_by_op用于类型匹配.

egg的添加是,遍历这个eclass中所有的enode,然后enode把他所属的eclass id存入`discriminant`的key中.

```rust
let mut add = |n: &L| {
    #[allow(clippy::mem_discriminant_non_enum)]
    let key = std::mem::discriminant(n);
    log::debug!("Add : {:?} class id : {:?} into key : {:?} ", n, class.id, key);
    classes_by_op
        .entry(key)
        .or_default()
        .insert(class.id)
};

let mut nodes = class.nodes.iter();
if let Some(mut prev) = nodes.next() {
    add(prev);
    for n in nodes { // 如果这个eclass中有多个enode, 检查后续的节点是否与之前的节点相同,不相同就继续添加id.
        if !prev.matches(n) {
            add(n);
            prev = n;
        }
    }
}
```
 
因为rust的enum是可以提供完全不同结构的类型, 因此`discriminant`就是映射他的结构类型到int key, 他的好处就是你可以添加很多很多不同类型的ir,这样基于类型的enode匹配就很简单的从字典中获取一个入口eclass开始匹配即可. 虽然如下所示,App可能存在很多个enode,但是至少能从类型上消除很大一部分的候选了.

```rust
[DEBUG egg::egraph] Add : App([93, 94]) class id : 54 into key : Discriminant(5) 
[DEBUG egg::egraph] Add : Let([45, 49, 53]) class id : 54 into key : Discriminant(7) 
[DEBUG egg::egraph] Add : Add([5, 47]) class id : 48 into key : Discriminant(3) 
[DEBUG egg::egraph] Add : Lambda([26, 41]) class id : 42 into key : Discriminant(6) 
[DEBUG egg::egraph] Add : App([35, 33]) class id : 36 into key : Discriminant(5)
```

# 5.1 Rewrite Marco

通过一个`rewrite!`的宏,将lhs,rhs构造成两部分.
```rust
pub struct Rewrite<L, N> {
    /// The name of the rewrite.
    pub name: Symbol,
    /// 可以是从expr构建/ 也可以是自定义的匹配的方式
    pub searcher: Arc<dyn Searcher<L, N> + Sync + Send>,
    /// 获得对应的结果, 可以是expr也可以自定义构建
    pub applier: Arc<dyn Applier<L, N> + Sync + Send>,
}
```

# 5.2 Pattern Match

这里的匹配是调用rewriter的search进行搜索. 首先这里的searcher是从字符串构造, 首先通过字符串解析为`PatternAst`
```rust
pub type PatternAst<L> = RecExpr<ENodeOrVar<L>>;

fn from_str(s: &str) -> Result<Self, Self::Err> {
    PatternAst::from_str(s).map(Self::from)
}
```
然后通过`PatternAst`构造出新的Pattern对象. 
```rust
impl<L: Language> Pattern<L> {
  /// Creates a new pattern from the given pattern ast.
    pub fn new(ast: PatternAst<L>) -> Self {
      let ast = ast.compact();
        let program = machine::Program::compile_from_pat(&ast);
        Pattern { ast, program }
    }

    /// Returns a list of the [`Var`]s in this pattern.
    pub fn vars(&self) -> Vec<Var> {
      let mut vars = vec![];
        for n in self.ast.as_ref() {
          if let ENodeOrVar::Var(v) = n {
            if !vars.contains(v) {
              vars.push(*v)
                }
            }
        }
        vars
    }
}
```
这里要注意到有machine的机制, egg是写了一个类似于虚拟机的东西, 将pattern解析为匹配的指令码, 然后通过虚拟机执行指令从而完成匹配的功能.
```rust

struct Machine {
    reg: Vec<Id>,
    // a buffer to re-use for lookups
    lookup: Vec<Id>,
}

pub struct Program<L> {
    instructions: Vec<Instruction<L>>,
    subst: Subst,
}

pub(crate) fn compile_from_pat(pattern: &PatternAst<L>) -> Self {
    let program = Compiler::new(pattern).compile();
    log::debug!("Compiled {:?} to {:?}", pattern.as_ref(), program);
    program
}
```


pattern的search是将自身的pattern ast转换为Enode,然后使用discriminant获取这个enode的op的类型,再从egraph中寻找所有的elass进行下一步匹配. 


但是我还是没懂`discriminant`是怎么获取的key是怎样的.官方文档上说此函数返回值只关心enum的类型,而不关心具体的值,这个就很奇怪了.

```rust
pub fn search(&self, egraph: &EGraph<L, N>) -> Vec<SearchMatches<L>> {
    self.searcher.search(egraph)
}

fn search(&self, egraph: &EGraph<L, A>) -> Vec<SearchMatches<L>> {
    match self.ast.as_ref().last().unwrap() {
        ENodeOrVar::ENode(e) => { // 首先搜索的节点是一个enode
            #[allow(clippy::mem_discriminant_non_enum)]
            let key = std::mem::discriminant(e); // 获取当前enode的key
            match egraph.classes_by_op.get(&key) {
                None => vec![],
                Some(ids) => ids
                    .iter()
                    .filter_map(|&id| self.search_eclass(egraph, id))
                    .collect(),
            }
        }
        ENodeOrVar::Var(_) => egraph
            .classes()
            .filter_map(|e| self.search_eclass(egraph, e.id))
            .collect(),
    }
}
```

