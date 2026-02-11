---
title: 使用c通过http发送文件
categories:
  - 编程语言
date: 2019-03-08 12:28:16
tags:
-   C
-   Linux
---

最近要用单片机通过`http`发送文件到服务器，所以写了个发送文件的`demo`。

<!--more-->

# 讲解

`http`协议本质上还是`socket`协议，我的测试`demo`是基于`Linux`的，所以我直接使用`Linux`的`socket`连接。

当我连接到服务器之后，我就需要直接发送`http`格式的头，这里要设置`post`的位置，内容长度，以及主机域名等等：
```
POST xxxxxxxxxx HTTP/1.1
Content-Length: xxxxxxx
Host: xxxxxxx
Content-Type: multipart/form-data;boundary=------FormBoundaryShouldDifferAtRuntime

```
然后再发送`http`的`body`：
```
------FormBoundaryShouldDifferAtRuntime
Content-Disposition: form-data; name="deviceId"

1
------FormBoundaryShouldDifferAtRuntime
Content-Disposition: form-data; name="file"; filename="debug.log"
Content-Type: application/octet-stream

[message-part-body; type: application/octet-stream, size: 2076 bytes]
------FormBoundaryShouldDifferAtRuntime--
```

要注意的就是这里所有的换行都是`\r\n`的，因为我是`linux`的系统，所以之前我一直计算字符串长度与`Content-Length`不匹配。

# 代码

```c

void set_header(char *pbuf, const char *host, int content_len) {
    sprintf(pbuf,
            "POST /device/upload_file.do HTTP/1.1\r\nHost: %s\r\nUser-Agent: "
            "curl/7.58.0\r\nAccept: */*\r\nContent-Length: %d\r\nContent-Type: "
            "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW; "
            "boundary=------------------------7b055677edfa30ed\r\n\r\n",
            host, content_len);
}

void set_content(char *pbuf, int id, const char *content) {
    sprintf(
        pbuf,
        "--------------------------7b055677edfa30ed\r\nContent-Disposition: "
        "form-data; "
        "name=\"deviceId\"\r\n\r\n%d\r\n--------------------------"
        "7b055677edfa30ed\r\nContent-Disposition: form-data; name=\"file\"; "
        "filename=\"test.txt\"\r\nContent-Type: "
        "text/plain\r\n\r\n%s\n\r\n--------------------------7b055677edfa30ed--\r\n",
        id, content);
}

```

这两个简单的函数是设置`http`的头与内容的。但是我这里写的只是一个简单`txt`的文件上传。如果要上传二进制文件的话，必须要把`body`部分分开来发送。

在`linux`下可以使用`sendfile`函数，但是在单片机中需要写别的函数。
