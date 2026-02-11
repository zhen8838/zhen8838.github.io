---
title: csharp 问题记录
mathjax: true
toc: true
categories:
  - 编程语言
date: 2021-10-26 15:07:17
tags:
- CSharp
---

记录一些开发csharp时遇到的问题以及解决方案。

<!--more-->

# 注意隐式转换与构造函数的冲突问题

因为cshapr每次都要写new,所以添加了一些隐式类型转换语法糖. 但是下面的代码就会出现问题, 就是`new WildCardPattern(Name)`这里其实并不是调用默认的`WildCardPattern(string Name, ExprPattern? Pattern)`, 而是又被隐式类型转换成了`WildCardPattern`然后准备调用复制构造函数构造,但是隐式类型转换的时候就陷入递归了.

```csharp
public sealed record WildCardPattern(string Name, ExprPattern? Pattern) : ExprPattern
{
    private static int _globalCardIndex = 0;

    public WildCardPattern() : this($"wc_{_globalCardIndex++}", null)
    {
    }

    public static implicit operator WildCardPattern(string Name) => new WildCardPattern(Name);

    public override bool MatchLeaf(Expr expr) => (Pattern?.MatchLeaf(expr) ?? true) && MatchCheckedType(expr);
}
```


# C# delegate的本质

最近在弄一个直接`jit`生成代码然后用c#动态调用的东西,但是需要动态调用你必须要告诉当前的函数指针一个`delegate`的定义,不然`c#`就不知道你要输入什么返回什么. 那么既然是`jit`,我们的就不能提前写好这个定义, 所以需要动态构造一个`delegate`.

给定一个类:
```csharp
public class CustomType
{
    public delegate float declf(float x, float y);
}
```
他的`delegate`的修饰符并不是一种`attr`,而是一种表示他继承自`MulticastDelegate`, 所以你可以发现他的是一个`NestedType`,并且他的基类是`MulticastDelegate`.
```csharp
var t = cls_type.GetNestedType("declf");
Assert.Equal(t.BaseType, typeof(MulticastDelegate));
```
同时他还具备了4个方法,其中一个是构造方法,以及三个重写的方法.
![](csharp-trick/delegate.png)

通过检查他的`il`我们可以发现`declf`就是个`class`,所以事情就简化成了构造这个类的问题.:
```csharp
.class private auto ansi '<Module>'
{
} // end of class <Module>

.class public auto ansi beforefieldinit CustomType
	extends [mscorlib]System.Object
{
	// Nested Types
	.class nested public auto ansi sealed declf
		extends [mscorlib]System.MulticastDelegate
	{
		// Methods
		.method public hidebysig specialname rtspecialname 
			instance void .ctor (
				object 'object',
				native int 'method'
			) runtime managed 
		{
		} // end of method declf::.ctor

		.method public hidebysig newslot virtual 
			instance float32 Invoke (
				float32 x,
				float32 y
			) runtime managed 
		{
		} // end of method declf::Invoke

		.method public hidebysig newslot virtual 
			instance class [mscorlib]System.IAsyncResult BeginInvoke (
				float32 x,
				float32 y,
				class [mscorlib]System.AsyncCallback callback,
				object 'object'
			) runtime managed 
		{
		} // end of method declf::BeginInvoke

		.method public hidebysig newslot virtual 
			instance float32 EndInvoke (
				class [mscorlib]System.IAsyncResult result
			) runtime managed 
		{
		} // end of method declf::EndInvoke

	} // end of class declf


	// Methods
	.method public hidebysig specialname rtspecialname 
		instance void .ctor () cil managed 
	{
		// Method begins at RVA 0x2050
		// Code size 7 (0x7)
		.maxstack 8

		IL_0000: ldarg.0
		IL_0001: call instance void [mscorlib]System.Object::.ctor()
		IL_0006: ret
	} // end of method CustomType::.ctor

} // end of class CustomType
```
接下来就照葫芦画瓢把这个类的定义给生成出来,然后我们再用这个定义的类型拿去`binding`那个`dll`里面的函数,然后我们就可以动态的`invoke`生成的代码了~
```csharp
AssemblyName aName = new AssemblyName("DynamicAssemblyExample");
AssemblyBuilder ab = AssemblyBuilder.DefineDynamicAssembly(aName, AssemblyBuilderAccess.RunAndCollect);
ModuleBuilder mb = ab.DefineDynamicModule(aName.Name);
TypeBuilder tb = mb.DefineType("MyDynamicType", TypeAttributes.Public);
TypeBuilder nesttb = tb.DefineNestedType("DynamicDelegate", TypeAttributes.NestedPublic | TypeAttributes.Sealed, typeof(MulticastDelegate));
var ctor = nesttb.DefineConstructor(MethodAttributes.Public | MethodAttributes.HideBySig | MethodAttributes.SpecialName | MethodAttributes.RTSpecialName, CallingConventions.Standard | CallingConventions.HasThis, new[] { typeof(object), typeof(IntPtr) });

ILGenerator ctorIL = ctor.GetILGenerator();
ctorIL.Emit(OpCodes.Ldarg_0);
ctorIL.Emit(OpCodes.Ldarg_1);
ctorIL.Emit(OpCodes.Ret);

var invoke = nesttb.DefineMethod("Invoke", MethodAttributes.Public | MethodAttributes.Virtual | MethodAttributes.HideBySig | MethodAttributes.NewSlot, CallingConventions.Standard | CallingConventions.HasThis, typeof(float), new[] { typeof(float), typeof(float) });
var beginInvoke = nesttb.DefineMethod("BeginInvoke", MethodAttributes.Public | MethodAttributes.Virtual | MethodAttributes.HideBySig | MethodAttributes.NewSlot, CallingConventions.Standard | CallingConventions.HasThis, typeof(IAsyncResult), new[] { typeof(float), typeof(float), typeof(IAsyncResult), typeof(object) });

var endInvoke = nesttb.DefineMethod("EndInvoke", MethodAttributes.Public | MethodAttributes.Virtual | MethodAttributes.HideBySig | MethodAttributes.NewSlot, CallingConventions.Standard | CallingConventions.HasThis, typeof(float), new[] { typeof(IAsyncResult) });
var created_class = tb.CreateType();
return created_class;****
```

# dotnet test crash时的解决方案

1. 开启--blame以及--blame-crash
2. 或者通过环境变量开启dump。
```sh
export DOTNET_DbgEnableMiniDump=1
export DOTNET_DbgMiniDumpType=4
export DOTNET_DbgMiniDumpName=/tmp/crash_%p.dmp
```
3. 通过lldb+sos插件调试core dump.
   1. 比如`lldb exec -c xx.dmp`。 这个exec可以是python或者dotnet。


安装dotnet sos，[参考](https://github.com/dotnet/diagnostics/blob/main/documentation/FAQ.md)

1. dotnet tool install --global dotnet-sos
2. dotnet tool install -g dotnet-symbol
3. dotnet sos install

```sh
sudo cp /Applications/Xcode.app/Contents/Developer/usr/bin/lldb /usr/local/bin
sudo install_name_tool -add_rpath /Applications/Xcode.app/Contents/SharedFrameworks /usr/local/bin/lldb
sudo codesign --force --sign - /usr/local/bin/lldb
```
然后用这个`/usr/local/bin/lldb`进行调试, 要注意使用的[这里的命令](https://learn.microsoft.com/en-us/dotnet/core/diagnostics/sos-debugging-extension#lldb-debugger) 才能看到csharp中的内容。
比如把`bt`替换成`dumpstack`


# P/invoke中使用AddressSanitizer

我需要调试ffi中的内存引用问题:
1. 首先编译时开启`-fsanitize=address -g`, 注意得编译为动态库。
2. 在macos中，可以在xcode中找到asan的动态库:
```sh
❯ find /Applications/Xcode.app -name '*libclang_rt.asan*osx_dynamic.dylib'
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/17/lib/darwin/libclang_rt.asan_osx_dynamic.dylib
```

3. 声明环境变量`export DYLD_INSERT_LIBRARIES=$ASAN_LIB`，并执行代码

