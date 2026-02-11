---
title: C# P/Invoke 总结
mathjax: true
toc: true
categories:
  - 编程语言
date: 2022-02-09 13:21:54
tags:
- CSharp
- C
---

关于C#调用本机lib时遇到的一些问题汇总.


<!--more-->

# C++打包

c++需要为导出的函数添加`extern 'C'`标识符. 然后在`cmake`中编译出`shared`的lib.

# C#调用方式

## DllImport

可以通过为函数添加`dllimport`的属性来指明需要使用的动态链接库. 同时我们可以不写明后缀,dotnet会自动根据平台寻找. 比如我打包出来的动态链接库为`libnncase_csharp.dylib`, 我们只要写:
```csharp
[DllImport("libnncase_csharp")]
static extern unsafe void interpreter_load_model(byte* buffer_ptr, int size);
```

这个方式适合于固定名字的动态库,并且加载了之后就不能卸载再重载.

## NativeLibrary.Load

目前我的方式就是动态的加载链接库,这种方式就是用起来比较麻烦, 首先我们要定义好对应的delegate签名,然后声明一系列的instance, 后面再加载对应的lib之后进行bind, 同时路径还得是固定的. 好处就是更新动态链接库不会受到影响.
```csharp
delegate bool delegate_init();
delegate_init interpreter_init;
TDelegate GetFFI<TDelegate>() => Marshal.GetDelegateForFunctionPointer<TDelegate>(NativeLibrary.GetExport(Handle, typeof(TDelegate).Name.Replace("delegate", "interpreter")));

Handle = NativeLibrary.Load("/Users/lisa/Documents/nncase/build/simulator/lib/libnncase_csharp.dylib");
interpreter_init = GetFFI<delegate_init>();
```

# vscode Debug Environment

我现在是先C#中调用我的动态链接库,然后在c++代码中再去加载完毕另外一个动态链接库,我发现如果我在命令行中运行就可以成功,但是如果是用vscode的debug模式就永远加载不了第二个dylib,因为vscode-csharp插件不知道为什么把DYLD_LIBRARY_PATH的环境变量给屏蔽掉了. 然后我尝试在dlopen之前设置DYLD_LIBRARY_PATH,发现还是不行,后面才知道如果当前设置了PATH等的环境变量,需要再开一个子进程才能生效.

这个暂时估计没办法解决.我提了issue.
****

## debug csharp with cpp

用dllimport的方式我尝试了各种方式去attch cpp的代码发现都不行,最终我还是在cpp中写了一个可执行程序然后用c#去起一个进程,然后再用这个时候直接attach到对应的可执行程序就可以debug了.这里还有个要注意点就是需要在可执行程序中加一个waitkey,不然程序等不到 attach就直接结束了.

# P/Invoke函数签名转换机制

## in/out/ref关键字的处理

在C#中,in, out, 和 ref 关键字用于控制参数如何在方法调用中传递.这些关键字在P/Invoke时也有特殊的意义,因为它们指示如何在托管代码和非托管代码之间传递数据.

1. in 关键字

当用于P/Invoke方法参数时,in 关键字指示参数应该从托管代码传递到非托管代码,但不期望从非托管代码返回数据到托管代码.
in 关键字用于优化传递大型结构体或数组时的性能,因为它避免了在非托管代码执行后将数据复制回托管内存的需要.
该关键字对于基本数据类型通常是不必要的,因为它们默认按值传递,但对于结构体或数组,它可以防止不必要的内存复制.

2. out 关键字

使用out关键字时,参数被假定为未初始化,并且非托管函数负责填充它.
在P/Invoke调用完成后,参数的值会从非托管代码复制回托管代码.
这适用于那些只需要从非托管函数输出数据的情况,不需要传递初始化的数据给非托管代码.

3. ref 关键字

ref 关键字用于两个方向的数据传递:它既将数据从托管代码传递到非托管代码,也将非托管代码的修改传回托管代码.
这适用于需要在非托管代码中被修改,并且修改后的值需要返回给托管代码的情况.
P/Invoke与这些关键字的交互

当你在P/Invoke声明中使用这些关键字时,你告诉CLR(公共语言运行时)如何在托管和非托管之间封送(marshal)数据.封送是指在不同运行环境(如托管和非托管代码)之间转换数据的过程.

对于in参数,CLR会创建数据的副本(如果需要)并将其传递到非托管代码,但在调用结束后不会检查非托管代码是否更改了数据.
对于out参数,CLR会跳过传递数据到非托管代码的步骤,但在调用结束后会从非托管代码读取数据并填充托管参数.
对于ref参数,CLR会传递数据到非托管代码,并在调用结束后读取可能的更改并更新托管参数.
这些关键字对于优化性能和确保数据正确传递非常重要.在使用这些关键字时,你应该清楚地了解非托管函数的预期行为,以便正确地选择使用 in, out, 或 ref.


如果你有一个C函数,它的签名是这样的:

```c
void inc(int const *a);
```

在C#中,你不能直接使用 `in int a` 来匹配 `int const *a`.相反,你应该使用 `ref` 关键字来传递一个 `int` 的引用,或者使用 `IntPtr` 来传递一个指针:

```csharp
// 使用 ref 关键字
[DllImport("YourLibrary.dll")]
public static extern void inc(ref int a);

// 使用 IntPtr
[DllImport("YourLibrary.dll")]
public static extern void inc(IntPtr a);
```

在调用时:

```csharp
int value = 5;
inc(ref value); // 使用 ref 关键字

// 或者,如果你使用 IntPtr
IntPtr ptr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(int)));
Marshal.WriteInt32(ptr, value);
inc(ptr);
Marshal.FreeHGlobal(ptr);
```

使用 `ref` 关键字是最简单的方法,因为它允许CLR为你处理指针的创建和数据的封送.如果你使用 `IntPtr`,你需要手动分配和释放非托管内存,并且还需要手动读写该内存.

## SafeHandle类

首先在csharp中有两种资源:

### 托管资源 Managed Resources:

这些是.NET环境下由公共语言运行时(CLR)管理的资源.托管资源主要指的是内存中的对象,如实例化的类或结构体.CLR负责分配和释放这些对象的内存,通常通过垃圾回收(GC)机制来完成.开发者通常不需要手动释放这些资源,因为GC会在对象不再被引用时自动清理它们.托管资源包括但不限于:

- 对象实例(如类的实例)
- 数组和集合
- 委托
- 事件处理器
- 以及其他所有由CLR管理的内存分配.
 
#### 非托管资源 Unmanaged Resources:

非托管资源是不由CLR直接管理的资源.它们通常是操作系统资源,如文件句柄,网络连接,数据库连接,图形界面句柄,以及任何其他需要通过平台调用(P/Invoke)或互操作服务(COM Interop)使用的资源.这些资源的分配和释放必须由开发者显式管理,因为垃圾回收器不会自动处理它们.常见的非托管资源包括:

- 文件和流(FileStream,StreamReader,StreamWriter)
- 网络套接字(Socket)
- 数据库连接(SqlConnection)
- 位图(Bitmap)等图形资源
- 通过P/Invoke调用分配的内存
- COM对象
- 其他需要调用操作系统API来管理的资源.


正确管理非托管资源是非常重要的,因为如果不释放这些资源,可能会导致内存泄漏或资源耗尽. 在.NET中,IDisposable接口和SafeHandle类是用来帮助开发者管理非托管资源的常用工具.当一个类实现了IDisposable接口,它应该提供一个Dispose方法,该方法负责释放类持有的所有非托管资源,以及可以释放的托管资源.而SafeHandle是一个专门设计用来封装非托管资源句柄的类,它提供了一个可靠的方式来确保非托管资源在不再需要时被正确释放.

#### SafeHandle的资源释放逻辑

它提供了两个重要的方法,`Dispose` 和终结器(由 `~SafeHandle()` 表示),它们在不同的情况下被调用:

1. **Dispose() 方法:**
   `Dispose` 方法是 `IDisposable` 接口的一部分,需要在你的代码中显式调用.当你确定不再需要 `SafeHandle` 包装的资源时,你应该调用 `Dispose` 方法.这是一种确定性的方式来释放非托管资源,因为它允许你精确控制资源的释放时间.调用 `Dispose` 方法通常会导致 `SafeHandle` 调用其 `ReleaseHandle` 方法来释放其封装的非托管资源,并将其标记为无效.

   ```csharp
   safeHandle.Dispose();
   ```

2. **终结器(Finalizer):**
   终结器由 `~SafeHandle()` 表示,并在垃圾回收器准备回收对象时自动调用.终结器的调用是非确定性的,因为你不能预知垃圾回收器何时会运行.当 `SafeHandle` 的实例不再有任何有效的托管引用时,垃圾回收器会在某个时间点回收它,并在此过程中调用终结器.终结器同样会尝试清理非托管资源,但它的执行时间是不可预测的.

   ```csharp
   ~SafeHandle() {
       Dispose(false);
   }
   ```

在 `SafeHandle` 的实现中,`Dispose` 方法通常是安全地处理托管和非托管资源的首选方法,而终结器是一种安全网,确保即使忘记调用 `Dispose` 方法,非托管资源也最终会被释放. 所以为了区分这两种情况, 基本上需要调用`Dispose(bool disposing)`方法来正确处理资源释放逻辑, 这个方法接受一个布尔值指示, 通常当 `Dispose()` 被显式调用时,`disposing` 是 `true`. 而终结器调用 `Dispose(false)` 时, 它只处理非托管资源, 因为托管资源可能已经被垃圾回收器清理了.

但是其实在`Dispose(bool disposing)`方法中并没有关心`disposing`, 其实为了防止在 `Dispose` 被调用后终结器再次释放资源, 显式使用`Dispose`方法会调用 `GC.SuppressFinalize(this)`, 这告诉垃圾回收器此对象的终结器不需要再被调用了,因为资源已经被显式清理了.

如果没有显式使用`Dispose`方法, 这里的终结器还是会调用`Dispose(false)`, 内部通过`InternalRelease`把资源释放掉. 在资源释放时, 首先通过检查`state`是否是open且只被引用一次且当前类构造的时候为`_ownsHandle`的形式, 检查成功后会将`state`设置为`close`状态, 这个时候要注意除非是使用`DangerousRelease`来释放, 否则必须会把`state`设置为`Disposed`. 等到`state`更新完毕, 才调用用户`override`的`ReleaseHandle`方法去处理非托管资源的释放. 

还有一点需要注意的是, 如果`extern`的释放函数是没法接受一个被`close`的`SafeHandle` 对象, 所以此时需要取出他的`handle`来调用资源释放函数. 释放后可以自行把`handle`设置为无效.

```csharp
namespace System.Runtime.InteropServices
{
    // This implementation does not employ critical execution regions and thus cannot
    // reliably guarantee handle release in the face of thread aborts.

    /// <summary>Represents a wrapper class for operating system handles.</summary>
    public abstract partial class SafeHandle : CriticalFinalizerObject, IDisposable
    {
#if DEBUG && CORECLR
        /// <summary>Indicates whether debug tracking and logging of SafeHandle finalization is enabled.</summary>
        private static readonly bool s_logFinalization = Environment.GetEnvironmentVariable("DEBUG_SafeHandle_FINALIZATION") == "1";
        /// <summary>Debug counter for the number of SafeHandles that have been finalized.</summary>
        private static long s_SafeHandlesFinalized;
#endif

        // IMPORTANT:
        // - Do not add or rearrange fields as the EE depends on this layout,
        //   as well as on the values of the StateBits flags.
        // - The EE may also perform the same operations using equivalent native
        //   code, so this managed code must not assume it is the only code
        //   manipulating _state.

#if DEBUG && CORECLR
        private readonly string? _ctorStackTrace;
#endif
        /// <summary>Specifies the handle to be wrapped.</summary>
        protected IntPtr handle;
        /// <summary>Combined ref count and closed/disposed flags (so we can atomically modify them).</summary>
        private volatile int _state;
        /// <summary>Whether we can release this handle.</summary>
        private readonly bool _ownsHandle;
        /// <summary>Whether constructor completed.</summary>
        private readonly bool _fullyInitialized;

        /// <summary>Bitmasks for the <see cref="_state"/> field.</summary>
        /// <remarks>
        /// The state field ends up looking like this:
        ///
        ///  31                                                        2  1   0
        /// +-----------------------------------------------------------+---+---+
        /// |                           Ref count                       | D | C |
        /// +-----------------------------------------------------------+---+---+
        ///
        /// Where D = 1 means a Dispose has been performed and C = 1 means the
        /// underlying handle has been (or will be shortly) released.
        /// </remarks>
        private static class StateBits
        {
            public const int Closed = 0b01;
            public const int Disposed = 0b10;
            public const int RefCount = unchecked(~0b11); // 2 bits reserved for closed/disposed; ref count gets 30 bits
            public const int RefCountOne = 1 << 2;
        }

        /// <summary>Creates a SafeHandle class.</summary>
        protected SafeHandle(IntPtr invalidHandleValue, bool ownsHandle)
        {
            handle = invalidHandleValue;
            _state = StateBits.RefCountOne; // Ref count 1 and not closed or disposed.
            _ownsHandle = ownsHandle;

            if (!ownsHandle)
            {
                GC.SuppressFinalize(this);
            }
#if DEBUG && CORECLR
            else if (s_logFinalization)
            {
                int lastError = Marshal.GetLastPInvokeError();
                _ctorStackTrace = Environment.StackTrace;
                Marshal.SetLastPInvokeError(lastError);
            }
#endif

            Volatile.Write(ref _fullyInitialized, true);
        }

        ~SafeHandle()
        {
            if (_fullyInitialized)
            {
                Dispose(disposing: false);
            }
        }

        internal bool OwnsHandle => _ownsHandle;

        protected internal void SetHandle(IntPtr handle) => this.handle = handle;

        public IntPtr DangerousGetHandle() => handle;

        public bool IsClosed => (_state & StateBits.Closed) == StateBits.Closed;

        public abstract bool IsInvalid { get; }

        public void Close() => Dispose();

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
#if DEBUG && CORECLR
            if (!disposing && _ctorStackTrace is not null)
            {
                long count = Interlocked.Increment(ref s_SafeHandlesFinalized);
                Internal.Console.WriteLine($"{Environment.NewLine}*** #{count} {GetType()} (0x{handle.ToInt64():x}) finalized! Ctor stack:{Environment.NewLine}{_ctorStackTrace}{Environment.NewLine}");
            }
#endif
            Debug.Assert(_fullyInitialized);
            InternalRelease(disposeOrFinalizeOperation: true);
        }

        public void SetHandleAsInvalid()
        {
            Debug.Assert(_fullyInitialized);

            Interlocked.Or(ref _state, StateBits.Closed);

            GC.SuppressFinalize(this);
        }

        protected abstract bool ReleaseHandle();

        public void DangerousAddRef(ref bool success)
        {
            Debug.Assert(_fullyInitialized);

            int oldState, newState;
            do
            {
                oldState = _state;
                ObjectDisposedException.ThrowIf((oldState & StateBits.Closed) != 0, this);
                newState = oldState + StateBits.RefCountOne;
            } while (Interlocked.CompareExchange(ref _state, newState, oldState) != oldState);
            success = true;
        }

        internal void DangerousAddRef()
        {
            bool success = false;
            DangerousAddRef(ref success);
        }

        public void DangerousRelease() => InternalRelease(disposeOrFinalizeOperation: false);

        private void InternalRelease(bool disposeOrFinalizeOperation)
        {
            Debug.Assert(_fullyInitialized || disposeOrFinalizeOperation);

            bool performRelease;
            int oldState, newState;
            do
            {
                oldState = _state;

                if (disposeOrFinalizeOperation && ((oldState & StateBits.Disposed) != 0))
                {
                    return;
                }

                ObjectDisposedException.ThrowIf((oldState & StateBits.RefCount) == 0, this);

                performRelease = ((oldState & (StateBits.RefCount | StateBits.Closed)) == StateBits.RefCountOne) &&
                                 _ownsHandle &&
                                 !IsInvalid;
                newState = oldState - StateBits.RefCountOne;
                if ((oldState & StateBits.RefCount) == StateBits.RefCountOne)
                {
                    newState |= StateBits.Closed;
                }
                if (disposeOrFinalizeOperation)
                {
                    newState |= StateBits.Disposed;
                }
            } while (Interlocked.CompareExchange(ref _state, newState, oldState) != oldState);

            if (performRelease)
            {
                int lastError = Marshal.GetLastPInvokeError();
                ReleaseHandle();
                Marshal.SetLastPInvokeError(lastError);
            }
        }
    }
}
```


#### SafeHandle使用例子

SafeHandle专门存储了一个指针, 在csharp中被释放的时候提供了各种callback让用户自己处理. 在P/invoke的过程中, 继承自safe handle的类就会被自动转换成一个指针对象. 并且下面这种结构体, 其实也可以被认为是一个指针, 也可以通过继承safe handle来处理.
```c
struct MlirContext {
  void *ptr;
};
```

SafeHandle如果是被extern的函数所构造, 那么必须要实现无参数的构造函数, 在调用extern的函数时, 实际上是先调用无参数构造函数然后再调用`SetHandle`把底层的指针传进去.

```csharp
public sealed class Context : SafeHandle, IEquatable<Context>
{
    private static HashSet<Context> _liveContextSet = new();

    public Context() : base(IntPtr.Zero, true)
    {
        System.Console.WriteLine("create");
    }

    public static Context Create() => mlirContextCreate();
}
```


在csharp中, 只要任何被csharp对象引用的对象都不会被gc掉, 但是在mlir的python binding中, 有个很麻烦的事情就是他为了避免重复构造对象,使用`Map<ptr,object>`来存储context/module/operation. 但是这个是在python binding部分内部实现的, 在python中并不知道字典的value存在引用, 所以当python中释放对象的时候可以在c++中清理这个字典. 但是csharp中字典是知道module这个key value


## 字符串处理

假设c中设计了StringRef进行字符串传递:
```c
typedef struct MlirStringRef {
  char *data;    ///< Pointer to the first symbol.
  size_t length; ///< Length of the fragment.
} StringRef;

void libParseMlirStringRef(StringRef ref) {
  printf("ref %s, %ld\n", ref.data, ref.length);
}

MlirStringRef MlirStringRefCreateFromCString(const char *str);
```

我发现如果直接使用如下方法:
```csharp
    [DllImport(LibraryName)]
    public static extern MlirStringRef mlirStringRefCreateFromCString([MarshalAs(UnmanagedType.LPStr)] string str);

    public static unsafe MlirStringRef mlirStringRefCreate(char* str, ulong length)
    {
        return new MlirStringRef { data = str, length = length };
    }

void main() {
    const string s = "hello!";
    ParseMlirStringRef(mlirStringRefCreateFromCString(s));
    ParseMlirStringRef(mlirStringRefCreateFromCString("hello!"));

    unsafe
    {
        var bytes = System.Text.Encoding.ASCII.GetBytes(s);
        fixed (byte* ptr = bytes)
        {
            var sref = mlirStringRefCreate((char*)ptr, (ulong)bytes.Length);
            ParseMlirStringRef(sref);
        }
    }
}
```

由于csharp默认的字符串编码只有Unicode和Ansi, 所以直接在底层使用没法直接使用string来构造, 只能先将string的编码转到ascii才能得到正确的结果:
```sh
ref �#m, 6
ref �#m, 6
ref hello!, 6
```

或者其实还可以用一个更简单的方法, 这样也可以得到正确的调用结果:
```csharp
    [DllImport(LibraryName)]
    public static extern MlirStringRef mlirStringRefCreateFromCString([MarshalAs(UnmanagedType.LPArray)] byte[] str);

void main() {
    const string s = "hello!";
    ParseMlirStringRef(mlirStringRefCreateFromCString(System.Text.Encoding.ASCII.GetBytes(s)));
    ParseMlirStringRef(mlirStringRefCreateFromCString(System.Text.Encoding.ASCII.GetBytes("hello!")));
}
```

在pybind11中, 如果str转换到std::string那么是utf-8编码的.


要注意一点, 如果c代码中直接返回一个`char *`指针, 然后C sharp中使用类似的方式:
```csharp
[DllImport(LibraryName)]
[return: MarshalAs(UnmanagedType.LPStr)]
public static extern  string isl_basic_map_get_tuple_name(IntPtr bmap, dim_type type);
```
这会引起潜在的问题, 因为csharp会自动管理并释放这个string, 如果这个内存是被c库中反复使用的话,就会出现指针异常.


# 包装类设计思路

看了mlir的python binding, 基本上也是一个`PyOperation`类中包含了一个指针, 然后基于这个类扩展出合适python的写法. 对于我来说, 准备是直接用继承自SafeHandle的类来对进行包装, 虽然这样写起来内部实现可能会复杂点, 内部调用cpi的和外部调用的都写在一起, 但是我觉得这些都是要写的, 写一起和分开没有很大区别, 使用csharp提供的internal保护应该就会很清爽.

# 引用计数问题

## CSharp 无法观测到c对象之间的依赖

isl中自己维护了一套引用计数的逻辑，但同时他还有一个context作为对象池，而用csharp调用时是看不到对象池的依赖关系的， 用两个例子可以很好的说明：

### keep live ctx 

这里让ctx不提前释放，但是可以观察到对于c对象的释放还是在ctx之前：

```csharp
public void TestSetMinAff{
  using (var ctx = Isl.ctx.Create()) {
    var set = new Isl.set(ctx, "[N] → { [i,j,k]: 05 i < 12 and 0 ≤ j< Nand 0 ≤ k < Nand O N < 123 }");
    var aff = set.max_multi_pw_aff();
    var min = aff.min_multi_val();
    var max = aff.max_multi_val();
    System.Console.WriteLine(min);
    System.Console WriteLine(max);
    GC.KeepAlive(ctx);
  }
}
```

```
［11,0,0］
isl_ctx.c:307: isl_ctx not freed as some objects still reference it
［11,121,121］
```

### manual free objects

```csharp
public void TestSetMinAffFreeManualy()
{
    using (var ctx = Isl.ctx.Create())
    {
        var set = new Isl.set(ctx, "[N] -> { [i,j,k]: 0<= i < 12 and 0 <= j < N and 0 <= k < N and 0 <= N < 123 }");
        var aff = set.max_multi_pw_aff();
        var min = aff.min_multi_val();
        var max = aff.max_multi_val();
        System.Console.WriteLine(min);
        System.Console.WriteLine(max);
        max.Dispose();
        min.Dispose();
        aff.Dispose();
        set.Dispose();
    }
}
```

这样才能把ctx的free放到最后处理。
```
{[11,0,0]}
{[11,121,121]}
free isl_multi_val 4307700432
free isl_multi_val 5510841360
free isl_multi_pw_aff 5510378384
free isl_set 5241109488
free isl_ctx 4307694592
```

