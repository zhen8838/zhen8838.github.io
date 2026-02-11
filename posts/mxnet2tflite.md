---
title: mxnet模型转tflite
mathjax: true
toc: true
categories:
  - 深度学习
date: 2020-02-24 17:30:52
tags:
-   Tensorflow
-   mxnet
---

今天尝试把`insightface`的模型转换到`tflite`格式，在此做个记录。

<!--more-->

# 安装依赖

转换工具使用的是微软的`mmdnn`，感觉好像有段时间没更新了，还是需要手动修改一些地方才可以正常运行。

```sh
pip install mxnet-cu101mkl scikit-learn mmdnn tensorflow==1.15.2
```

1. 修改`np.load`默认参数

    因为现在的`numpy`默认`load`的时候`allow_pickle=False`，所以需要进入`mmdnn`中全局搜索`np.load`，并修改其参数，我这次是`mxnet`转`tensorflow`，就只要改`conversion/common/DataStructure/emitter.py`和`conversion/tensorflow/tensorflow_emitter.py`文件就好了。
   

# 下载模型

从`insightface`的`modell zoo`中下载一个模型，我下载的是[`mobilenet`模型](https://uc3fb6447c5249ed356ba3872f6e.dl.dropboxusercontent.com/cd/0/get/AycBEeooWGoSc-W_dzCY47OaQLGTaYVtwgZAsORaGDfFx3mk7pyVMk7Dl6DSCX2vtM-arRRtvEFbvXLoVFuP1ykb7o7WrCfuuRkGOFtrj3kUtA/file?_download_id=39148373175258925047465484472256479451969124919058759030232633835&_notify_domain=www.dropbox.com&dl=1)


# 修改模型

我下载的这个模型可能是因为之前训练的模型构建问题，有一个名为`pre_fc1`的层的权重参数却不叫`pre_fc1_weight`，会导致转换出错。

![](mxnet2tflite/mxnet2tfliet-1.png)

需要修改一下模型参数名称。我网上看了下没发现有好的修改`param`参数名称的方法，直接替换`json`和`param`文件中的`fc1_weight`为`pre_fc1_weight`会导致加载失败，因此我就将`mmdnn/conversion/mxnet/mxnet_parser.py`中410行改成如下：
```python
if source_node.name=='pre_fc1':
    weight = self.weight_data.get('fc1' + "_weight").asnumpy().transpose((1, 0))
else:
    weight = self.weight_data.get(source_node.name + "_weight").asnumpy().transpose((1, 0))
```

同时，因为`insightface`的模型输入时的归一化操作被固定在了模型中，因此需要修改网络，我找了半天也没找到办法。。于是就直接修改`json`文件，使用下列程序删除前面两个节点：

```python
 s= json.load(open('/home/zqh/workspace/insightface/test/model-symbol.json.bak'))
 
 for nodes in s['nodes']:
     for inputs in nodes['inputs']:
         inputs[0]=inputs[0]-2
 
 for i in range( len(s['arg_nodes'])):
     s['arg_nodes'][i]=s['arg_nodes'][i]-2
 
 for i in range( len(s['node_row_ptr'])):
     s['node_row_ptr'][i]=s['node_row_ptr'][i]-2

 
 for heads in s['heads']:
     heads[0]=heads[0]-2
 
 ss=json.dumps(s)
 with open('/home/zqh/workspace/insightface/test/model-symbol.json','w') as f:
     print(ss, file=f)
```

**NOTE**： 上面的代码只是修改节点索引，后面需要手动把模型`json`文件不需要的节点删除。

# 转换模型

运行命令：
```sh
mmconvert -sf mxnet -in ../models/model-symbol.json -iw ../models/model-0000.params  -df tensorflow -om mbv1face --inputShape 3,112,112 --dump_tag SERVING
```
即可得到`tensorflow.savemodel`格式的模型，再利用`toco`：

```sh
toco --saved_model_dir mbv1face/ ----output_format tflite --output_file mbv1face.tflite
```