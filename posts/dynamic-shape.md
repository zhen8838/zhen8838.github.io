---
title: tvm dynamic shape 学习
mathjax: true
toc: true
categories:
  - 编译器
date: 2023-11-15 11:52:53
tags:
- TVM
- 动态shape
---

探究tvm dynamic shape的实现.

<!--more-->


# tvm ir design

![](dynamic-shape/ir.png)

将relax ir的语法dump出来可以知道, 这里与relay那种数据流的ir不同, dataflow中的每个操作使用一个var binding来存储.

```python
@R.function
def fn1(a: R.Tensor(("n", 10), 'float32'), b: R.Tensor((1,), 'float32')):
    with R.dataflow():
        n = T.int64()
        c: R.Tensor((n, 10)) = a + b
        R.output(c)
    return c
```

```python
Function(
    params=[
        Var(
            name_hint="a",
            struct_info=TensorStructInfo(
                dtype=float32,
                shape=ShapeExpr(
                    values=[
                        PrimExpr(value=`n`),
                        PrimExpr(value=`T.int64(10)`)
                    ],
                    struct_info=ShapeStructInfo(
                        ndim=2,
                        values=[
                            PrimExpr(value=`n`),
                            PrimExpr(value=`T.int64(10)`)
                        ]
                    )
                )
            )
        ),
        Var(
            name_hint="b",
            struct_info=TensorStructInfo(
                dtype=float32,
                shape=ShapeExpr(
                    values=[PrimExpr(value=`T.int64(1)`)],
                    struct_info=ShapeStructInfo(
                        ndim=1,
                        values=[PrimExpr(value=`T.int64(1)`)]
                    )
                )
            )
        )
    ],
    body=SeqExpr(
        blocks=[
            BindingBlock(
                bindings=[
                    VarBinding(
                        var=Var(
                            name_hint="c",
                            struct_info=TensorStructInfo(
                                dtype=,
                                shape=ShapeExpr(
                                    values=[
                                        PrimExpr(value=`n`),
                                        PrimExpr(value=`T.int64(10)`)
                                    ],
                                    struct_info=ShapeStructInfo(
                                        ndim=2,
                                        values=[
                                            PrimExpr(value=`n`),
                                            PrimExpr(value=`T.int64(10)`)
                                        ]
                                    )
                                )
                            )
                        ),
                        value=Call(
                            op=Op(name="relax.add"),
                            args=[
                                Var(
                                    name_hint="a",
                                    struct_info=TensorStructInfo(
                                        dtype=float32,
                                        shape=ShapeExpr(
                                            values=[
                                                PrimExpr(value=`n`),
                                                PrimExpr(value=`T.int64(10)`)
                                            ],
                                            struct_info=ShapeStructInfo(
                                                ndim=2,
                                                values=[
                                                    PrimExpr(value=`n`),
                                                    PrimExpr(value=`T.int64(10)`)
                                                ]
                                            )
                                        )
                                    )
                                ),
                                Var(
                                    name_hint="b",
                                    struct_info=TensorStructInfo(
                                        dtype=float32,
                                        shape=ShapeExpr(
                                            values=[PrimExpr(value=`T.int64(1)`)],
                                            struct_info=ShapeStructInfo(
                                                ndim=1,
                                                values=[PrimExpr(value=`T.int64(1)`)]
                                            )
                                        )
                                    )
                                )
                            ],
                            struct_info=TensorStructInfo(
                                dtype=,
                                shape=ShapeExpr(
                                    values=[
                                        PrimExpr(value=`n`),
                                        PrimExpr(value=`T.int64(10)`)
                                    ],
                                    struct_info=ShapeStructInfo(
                                        ndim=2,
                                        values=[
                                            PrimExpr(value=`n`),
                                            PrimExpr(value=`T.int64(10)`)
                                        ]
                                    )
                                )
                            )
                        )
                    )
                ]
            )
        ],
        body=Var(
            name_hint="c",
            struct_info=TensorStructInfo(
                dtype=,
                shape=ShapeExpr(
                    values=[
                        PrimExpr(value=`n`),
                        PrimExpr(value=`T.int64(10)`)
                    ],
                    struct_info=ShapeStructInfo(
                        ndim=2,
                        values=[
                            PrimExpr(value=`n`),
                            PrimExpr(value=`T.int64(10)`)
                        ]
                    )
                )
            )
        ),
        struct_info=TensorStructInfo(
            dtype=,
            shape=ShapeExpr(
                values=[
                    PrimExpr(value=`n`),
                    PrimExpr(value=`T.int64(10)`)
                ],
                struct_info=ShapeStructInfo(
                    ndim=2,
                    values=[
                        PrimExpr(value=`n`),
                        PrimExpr(value=`T.int64(10)`)
                    ]
                )
            )
        )
    ),
    ret_struct_info=TensorStructInfo(
        dtype=,
        shape=ShapeExpr(
            values=[
                PrimExpr(value=`n`),
                PrimExpr(value=`T.int64(10)`)
            ],
            struct_info=ShapeStructInfo(
                ndim=2,
                values=[
                    PrimExpr(value=`n`),
                    PrimExpr(value=`T.int64(10)`)
                ]
            )
        )
    ),
    is_pure=1,
    attrs={"global_symbol": "fn1"},
    struct_info=FuncStructInfo(
        params=[
            TensorStructInfo(
                dtype=float32,
                shape=ShapeExpr(
                    values=[
                        PrimExpr(value=`n`),
                        PrimExpr(value=`T.int64(10)`)
                    ],
                    struct_info=ShapeStructInfo(
                        ndim=2,
                        values=[
                            PrimExpr(value=`n`),
                            PrimExpr(value=`T.int64(10)`)
                        ]
                    )
                )
            ),
            TensorStructInfo(
                dtype=float32,
                shape=ShapeExpr(
                    values=[PrimExpr(value=`T.int64(1)`)],
                    struct_info=ShapeStructInfo(
                        ndim=1,
                        values=[PrimExpr(value=`T.int64(1)`)]
                    )
                )
            )
        ],
        ret=TensorStructInfo(
            dtype=,
            shape=ShapeExpr(
                values=[
                    PrimExpr(value=`n`),
                    PrimExpr(value=`T.int64(10)`)
                ],
                struct_info=ShapeStructInfo(
                    ndim=2,
                    values=[
                        PrimExpr(value=`n`),
                        PrimExpr(value=`T.int64(10)`)
                    ]
                )
            )
        ),
        purity=True
    )
)
```

而根据[relax shape设计文档](https://github.com/tlc-pack/relax/wiki/Relax-Shape-Computation-Design#a1-bringing-symbolic-shape-closer-to-expr)下面这种情况应该是无法支持的:
```python
@R.function
def fn2(a: R.Tensor(("n", 10), 'float32'), b: R.Tensor((1,), 'float32')):
    with R.dataflow():
        n = T.int64()
        c = a + b
        cshape: R.Shape() = R.shape_of(c)
        d = R.reshape(c, [1, cshape[0], cshape[1], 1])
        R.output(d)
    return d
```

我在思考是不是应该有一种直接基于数据流的方式来添加symbolic shape的信息?