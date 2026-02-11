---
title: 模板元编程实战(第一章)
mathjax: true
toc: true
date: 2021-08-16 20:44:30
categories:
  - 编程语言
tags:
- 模板元编程
- CPP
---

模板元编程实战,第一章节.

<!--more-->



```c++
#include "commom.hpp"
#include <gtest/gtest.h>
```



# 元函数介绍 

元函数会在编译期被调用与执行。在编译阶段,编译器只能构造常量作为其中间结果,无法构造并维护可以记录系统状态并随之改变的量,因此编译期可以使用的函数(即元函数)只能是无副作用的函数。



```c++
constexpr int func(int a) { return a + 1; }
// NOTE 下面这个函数是随着调用次数改变输出的值,所以无法作为元函数.
// static int count = 3;
// constexpr int func(int a) { return a + (count++); }

TEST(chapter1, _1_1_1)
{
  // NOTE 此函数没有固定输入输出,所以可以同时作为编译期函数或运行时函数.
  constexpr int a = func(3);
  int b = 4;
  int c = func(b);
  ic(a, c);
}
```




## 类型元函数
如果说上面的函数是操作$y=f(x)$,那么他的输入是一个数值.但其实在c++中我们可以把类型看作是一种数值,对类型进行计算.


```c++

template <typename T>
struct Func_
{
  using type = T;
};

template <>
struct Func_<int>
{
  using type = uint64_t;
};

template <>
struct Func_<uint32_t>
{
  using type = uint64_t;
};
```


不过上面的元函数表述方法太过繁琐,我们可以用更加简化的方式来调用,由于using的时候默认会认为你在声明namespace,所以需要加上typename修饰来表明这是一个类型.


```c++
template <typename T>
using Fun = typename Func_<T>::type;

TEST(chapter1, _1_1_2)
{
  // NOTE 我们构建的类型映射,把int 或者 uint32都转换到了uint64,然后利用比较他的类型是否和uint64相同.
  Fun<int> a = 0x1;
  ic(std::is_same<decltype(a), uint64_t>::value);
}
```



# 各种元函数表示方法


```c++

template <int a>
constexpr int no_struct_fun = a + 1; // NOTE 这样也可以是一个元函数,是不是很神奇, 不过他这样只能有一个返回值

template <int a>
struct struct_fun // NOTE 结构体的好处就是可以保存多个返回值
{
  using type = int;
  using ref_type = int &;
  using const_ref_type = const int &;
  constexpr static size_t size = sizeof(int);
};

```

## 模板类型参数与容器模板

  模板元编程最重要的就是把类型也看作是一种数据,要知道我们编写的程序在编译时必然被编译器存储,那么代码的类型也是一种变量存储在编译器中的,因此我们合理地调用类型数据,可以发挥更大的作用.

###  模板作为元函数的输入

NOTE 我们可以传入一个模版类型,这个模板类型可以接收多个一个或多个模板类型的,此时对应的数学表达式类似于: 

$$
\text{Func}(T_1,t_2)=T_1(t_2)
$$


```c++

template <template <typename> class T1, typename T2>
struct TypeCall_
{
  using type = typename T1<T2>::type;
};
template <template <typename> class T1, typename T2>
using TypeCall = typename TypeCall_<T1, T2>::type;

TEST(chapter1, _1_2_1)
{
  TypeCall<std::remove_reference, int &> h = 3;
  ic(h);
}
```



### 模板作为元函数的输出
  
  NOTE 其实我个人觉得这只能算是多个元函数的compose,元函数中很



```c++

template <int AddorRemoveRef>
struct OptFunc_;

template <>
struct OptFunc_<0>
{
  template <typename T>
  using type = std::add_lvalue_reference<T>;
};

template <>
struct OptFunc_<1>
{
  template <typename T>
  using type = std::remove_reference<T>;
};

template <typename T, int AddorRemoveRef>
using OptFunc = typename OptFunc_<AddorRemoveRef>::template type<T>;

TEST(chapter1, _1_2_2)
{
  OptFunc<int, 1>::type h = 1;
  ic(h);
}

```


### 容器模板

容器模板就是一种可以保存数值数据或者类型数据的一个容器.他就是一个类型,但是他可以保存以上两种数据.


```c++
template <int... Vals>
struct IntContainer
{
  // NOTE 即IntContainer这个类型中存储了一系列int值
};

template <typename... Types>
struct TypeContainer
{
  // NOTE 存储了一系列类型
};

// 以下两个是比较复杂的情况, ①保存了一系列的模板类型
template <template <typename> typename... Types>
struct TemplateContainer
{
};

template <template <typename...> typename... Types>
struct TemplateContainer2
{
};
```



## 编译期实现分支、循环



### 典型的顺序执行元函数



```c++
template <typename T>
struct RemoveCV_
{
private:
  using inner_type = typename std::remove_reference<T>::type;

public:
  using type = typename std::remove_reference<inner_type>::type;
};

template <typename T>
using RemoveCV = typename RemoveCV_<T>::type;

TEST(chapter1, _1_3_1)
{
  RemoveCV<const int &> h = 1;
  ic(h);
}
```



### 分支执行的代码

NOTE 其实分支执行的方式有好多,我自己都能写出好几个,但是找到一个比较通用优雅的写法可能还挺难


1. 通过conditional实现分支, 这种方法就是用在结构体模板继承时使用的(通过选择来继承类型,然后得到当前的值)

```c++
template <int T>
struct IsOdd_ : std::conditional_t<(T % 2) == 1, std::true_type, std::false_type>
{
};

template <int T>
constexpr bool IsOdd = IsOdd_<T>::value;

```


2. 通过特化匹配来实现分支,比如我们设计了一个isfloat结构体,默认都是false,通过类型分发的可以自定义不同类型是否是float.

```c++
template <typename T>
struct isFloat_ : std::false_type
{
};

template <>
struct isFloat_<float> : std::true_type
{
};

template <>
struct isFloat_<uint64_t> : std::true_type
{
};

template <typename T>
constexpr auto isFloat = isFloat_<T>::value;
```


3. std::enable_if来实现分支,这个是比较好用的,下面就是一个简单的应用. 首先利用enable_if来匹配当前参数的大致类型,然后对于array类型,写了一个traits去获得他的size,然后再for循环.这其实就是任意类型打印的雏形了.


```c++
template <typename T>
struct is_array : std::false_type
{
};
template <typename T, size_t N>
struct is_array<std::array<T, N>> : std::true_type
{
};

template <typename T>
constexpr bool is_array_v = is_array<T>::value;

template <typename T>
struct array_traits
{
};
template <typename T, size_t N>
struct array_traits<std::array<T, N>>
{
  constexpr static size_t size = N;
};

template <typename T>
std::enable_if_t<is_array_v<T>, void> print_any(const T &v)
{
  std::cout << "arr :";
  for (size_t i = 0; i < array_traits<T>::size; i++)
  {
    std::cout << v[i] << " , ";
  }
  std::cout << std::endl;
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, void> print_any(const T &v)
{
  std::cout << "value :" << v << std::endl;
}

TEST(chapter1, _1_3_2)
{
  // 1
  ic(IsOdd<1>, IsOdd<2>);
  // 2
  ic(isFloat<float>, isFloat<uint64_t>, isFloat<double>);
  // 3
  std::array<float, 10> arr = {1, 2, 3, 4, 5, 6, 7};
  print_any(arr);
  print_any(true);
}
```



### 循环执行的代码
通常我们需要用递归的方式进行执行


```c++
template <size_t Input>
constexpr size_t Onescount = (Input % 2) + Onescount<(Input / 2)>;

template <>
constexpr size_t Onescount<0> = 0;

TEST(chapter1, _1_3_3)
{
  constexpr size_t res = Onescount<45>;
  ic(res);
}
```



## 练习


### 练习1
构造一个输入为类型输出为值的元函数


```c++
template <typename T>
struct get_type_size : std::integral_constant<size_t, sizeof(T)>
{
};

TEST(chapter1, practice_1)
{
  ic(get_type_size<std::tuple<int, float, double>>::value);
  ic(get_type_size<float>::value);
}

```


### 练习2
元函数的输入参数甚至可以是类型与数值混合的。尝试构造个元函数,其输入参数为一个类型以及一个整数。如果该类型所对应对象的大小等于该整数,那么返回true,否则返回 false。


```c++
template <typename T, size_t N>
struct get_type_size2 : std::bool_constant<sizeof(T) == N>
{
};

TEST(chapter1, practice_2)
{
  ic(get_type_size2<std::tuple<int, float, double>, 16>::value);
  ic(get_type_size2<float, 16>::value);
  ic(get_type_size2<uint64_t, 8>::value);
}
```



### 练习3
其他的元函数表现形式

如果我们所有的操作都是在操作类型,我们可以用继承的方式把类型进行传递,这样就不需要中间变量.
当然在类型操作不足的时候,我们可以利用一些操作补足他们,比如下面这个例子就是先利用一个constexpr函数求值,此时这个值的类型是integral_constant类型,我们再decltype得到他的类型,再获取他的值.(这是个简单的例子,可能看不出这样有什么方便的)


```c++
template <size_t A, size_t B>
constexpr auto add()
{
  return std::integral_constant<size_t, A + B>();
}

template <size_t A, size_t B>
struct add_ : decltype(add<A, B>())
{
};

TEST(chapter1, question_3)
{
  ic(add_<1, 2>::value);
}

```


### 练习4
构造一个元函数,返回另一个元函数

```c++
template <typename T>
struct reduce
{
  using type = std::integral_constant<size_t, 0>;
};

template <size_t A, size_t B>
struct reduce<std::index_sequence<A, B>>
{
  using type = std::integral_constant<size_t, A + B>;
};

template <size_t A, size_t B, size_t... Ns>
struct reduce<std::index_sequence<A, B, Ns...>>
{

  using type = typename reduce<std::index_sequence<A + B, Ns...>>::type;
};

TEST(chapter1, question_4)
{
  ic(reduce<std::index_sequence<1, 2, 3, 4>>::type::value);
}
```



### 练习5
 
使用 SFINAE构造一个元函数:输入一个类型T,当T存在子类型type时该元函数返回true,否则返回 false
 

```c++
template <typename T, typename = void>
struct has_type : std::false_type
{
};

template <typename T>
struct has_type<T, std::void_t<typename T::type>> : std::true_type
{
};

TEST(chapter1, question_5)
{
  ic(has_type<reduce<std::index_sequence<1, 2, 3, 4>>>::value);
  ic(has_type<int>::value);
}

```


### 练习6
使用在本章中学到的循环代码书写方式,编写一个元函数,输入一个类型数组,输出一个无符号整型数组,输出数组中的每个元素表示了输入数组中相应类型变量的大小。
 

```c++
template <typename... TArgs>
struct get_sizes
{
  constexpr static std::array<size_t, sizeof...(TArgs)> arr = {sizeof(TArgs)...};
};

TEST(chapter1, question_6)
{
  ic(get_sizes<int, float, double, int8_t, uint32_t>::arr);
}
```



### 练习7

使用分支短路逻辑实现一个元函数,给定一个整数序列,判断其中是否存在值为1 的元素。如果存在,就返回true,否则返回 false


```c++

template <size_t V>
constexpr bool is_zero = (V == 0);

template <bool cur, typename TNext>
constexpr static bool AndValue = false;

template <typename TNext>
constexpr static bool AndValue<true, TNext> = TNext::value;

template <typename T>
struct has_one
{
  constexpr static bool value = false;
};

template <size_t V>
struct has_one<std::index_sequence<V>>
{
  constexpr static bool value = is_zero<V>;
};

template <size_t V, size_t... Ns>
struct has_one<std::index_sequence<V, Ns...>>
{
  constexpr static bool cur_is_zero = is_zero<V>;
  constexpr static bool value = AndValue<cur_is_zero, has_one<std::index_sequence<Ns...>>>;
};

TEST(chapter1, question_7)
{
  ic(has_one<std::index_sequence<0, 0, 0, 0, 0, 0, 0, 0, 1>>::value);
  ic(has_one<std::index_sequence<0, 0, 0, 0, 0, 0, 0, 0>>::value);
  ic(has_one<std::index_sequence<0, 1, 0, 0, 0, 0, 0, 0>>::value);
}

```
