---
title: ncnn学习
mathjax: true
toc: true
categories:
  - 边缘计算
date: 2021-04-28 23:30:44
tags:
-   ncnn
---

对ncnn学习的一些汇总。


<!--more-->


# Mat内存分布

Mat是ncnn所有的数据对象集合，因此我们必须对其有所了解，下面这个函数就是其中一个构造函数（这里需要吐槽一下，ncnn的设计就是单幅图像输入，所以mat最大只支持三维），其中需要注意到的就是内存分配时需要对齐：

```cpp
void Mat::create(int _w, int _h, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;

    cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}
```

## 块对齐 (block aligin)

这是nihui对于mat内存排布的说明。

```
mat shape w=3 h=2 c=4

internal memory layout
[(a00 a01 a02) (a10 a11 a12) pad pad]
[(b00 b01 b02) (b10 b11 b12) pad pad]
[(c00 c01 c02) (c10 c11 c12) pad pad]
[(d00 d01 d02) (d10 d11 d12) pad pad]

each channel is 16byte aligned, 
padding values may be filled in channel gaps

mat.data -> address of a00
mat.row(1) -> address of a10
mat.channel(0).row(1) -> address of a10
mat.channel(1).row(1) -> address of b10
```

### by channel alignSize

ncnn通常把以单个通道的图像(h*w)进行读取，然后进行一些卷积操作，所以要by channel的对数据进行对齐，考虑到对于不同的元素有不同的`elemsize`,同时在分配时还需要对内存块大小进行对齐，为了快速的对hw的内存进行读写，他这里默认分配16bit的倍数。

```cpp
    // element size in bytes
    // 4 = float32/int32
    // 2 = float16
    // 1 = int8/uint8
    // 0 = empty
    size_t elemsize;
```

比如我们申请float矩阵`w=3,h=9,c=4`，那么每一个`channel`的内存块本来应该是`3*9=27 byte = 27*elemsize = 108 bit`，然而`108`不是`16`的倍数，所以用`alignSize`计算到最小16的倍数为112，然后再除elemsize得到cstep为28。
```cpp
    cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;
```

### whole alignSize

然后根据上述思路，整体的大小是以4倍大小分配的。
```cpp
size_t totalsize = alignSize(total() * elemsize, 4);
```

## 内存申请 (Malloc)


同时申请到的内存位置也需要对齐在内存上，便于我们的cpu整块读取，下面是整体的函数，我这里执行的是`posix_memalign`，不过还是有必要讲讲具体思路。

```cpp
#if __AVX__
// the alignment of all the allocated buffers
#define MALLOC_ALIGN 32
#else
// the alignment of all the allocated buffers
#define MALLOC_ALIGN 16
#endif

static inline void* fastMalloc(size_t size)
{
#if _MSC_VER
    return _aligned_malloc(size, MALLOC_ALIGN);
#elif (defined(__unix__) || defined(__APPLE__)) && _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
    void* ptr = 0;
    if (posix_memalign(&ptr, MALLOC_ALIGN, size))
        ptr = 0;
    return ptr;
#elif __ANDROID__ && __ANDROID_API__ < 17
    return memalign(MALLOC_ALIGN, size);
#else
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + MALLOC_ALIGN);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
#endif
}
```

### fastMalloc

#### malloc

假设我是内存对齐`MALLOC_ALIGN`是32位。
```cpp
unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + MALLOC_ALIGN);
```
这里加上`sizeof(void*)`是为了用来保存原始malloc出来的数据空间，然后加上`MALLOC_ALIGN`的大小，因为我们要进行多大的对齐，我们需要偏移的大小总是在`0~MALLOC_ALIGN`中，所以加上`MALLOC_ALIGN`即可。

如果我的`size`是`112`，那么实际申请的大小是`112+8+32=152`，假设我这里申请到的内存为`0x555555b1cb30`。

#### align ptr

接下来我们要对刚刚申请的`udata`进行一系列操作，首要的事情就是要把起步的内存块进行一个内存对齐，我们要对一个指针操作，那么需要先转成指针的指针，并且需要考虑到对内存进行偏移之后，我们需要保存原来的`block`的起始地址用于释放内存，否则就会出现问题，ncnn的思路就是原来申请的内存块的头部存放着原始地址，后面一块数据才是实际使用的。

所以我们会跳过一个小的内存开始对齐，其中对齐的方法也是和找到`MALLOC_ALIGN`：
```cpp
unsigned char **adata = alignPtr((unsigned char **)udata + 1, MALLOC_ALIGN);
```

然后我们返回对齐的指针adata，然后把adata前面一个地址空间保存raw的地址，最终的内存分布如下图所示：

```
             align with MALLOC_ALIGN
                0x40
                 |
  0x30    0x38   V       0x44     0x48    0x4C      0x50
     |------|----|--------|--------|--------|--------|
    head     head  data1     data2    data3    data4
             addr
```

### NCNN x86 cpu加速

想要使用ncnn的一些加速方法，需要从内存管理就开始适配，比如我想给定输入大小进行malloc，ncnn底层就会自动帮我padding到4的倍数，这就需要十分注意。所以这里显示的h，w是正确的，但是cstep并不是6。
```cpp
TEST(cpp_lang, ncnn_mat_create_shape)
{
    nncase::runtime_shape_t in_shape { 2, 3, 4 };
    Mat m((int)in_shape[0], (int)in_shape[1], (int)in_shape[2]);
    cout << m.cstep << endl; // 8
    cout << m.w  << endl; // 2
    cout << m.h  << endl; // 3
    cout << in_shape[0] * in_shape[1] << endl; // 6
}
```

#### elempack

首先获得输入的`elemsize`和`elempack`，然后默认输出的`out_elempack=1`
```cpp
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_elempack = 1;
```

根据平台特性设定`out_elempack`:

```cpp
#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX__
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1; // 如果是AVX，那么256bit一组，所以elempack设置为8
#else
        out_elempack = num_output % 4 == 0 ? 4 : 1; 
        // 如果是SSE，那么128bit一组，所以elempack设置为4
#endif
    }
#endif // __SSE2__
```

计算输出`elemsize`,假设原始输入c是3，他的elemsize是4，输入数据是3\*4=12， out channel是64，假设elemsize是4，输出数据是64\*4=256 , 现在输出要8个一组，所以256/8=32 所以输出的elemsize是32。最后输出`top blob`申请的内存就是`outw,outh,channel=8 (64/8),elemsize=32 (4*8)`

所以ncnn这都是channel通道上的packing，所以对于channel数大的情况下，卷积速度就快。

```cpp
    size_t out_elemsize = elemsize / elempack * out_elempack; 

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

```

#### 卷积执行操作

x86的cpu优化主要还是看packing的，所以他的卷积函数选择都是看输入packing大小和输出packing大小，主要我看了一下就这一些选项，其中的卷积函数都是类似的。
```cpp
if (elempack == 8 && out_elempack == 8)
if (elempack == 1 && out_elempack == 8)
if (elempack == 4 && out_elempack == 8)
if (elempack == 8 && out_elempack == 1)
if (elempack == 8 && out_elempack == 4)
```

同时这里卷积的输出后，他可能是packing的，所以ncnn还提供了`convert_packing`函数对pack的mat进行转换，不过他只支持output与pack的值为倍数关系时才能成功转换,这里进行转换之后原来的内存块padding的位置应该是被填充了，然后尾部会多余一些值。

```
                                            h*w
                                    w        w       w
                                c1 [ 0  1] [ 2  3] [ 4  5] [ p  p]
                                c2 [ 6  7] [ 8  9] [10 11] [ p  p]
                                c3 [12 13] [14 15] [16 17] [ p  p]
                                c4 [18 19] [20 21] [22 23] [ p  p]
                                            |
                                            | convert_packing
                                            v
                                           h*w
                w                           w                             w
c1 [ (0 6 12 18) (1 7 13 19) ] [ (2 8 14 20)  (3 9 15 21) ] [ (4 10 16 22)  (5 11 17 23)] [o o o o o o o o]
```