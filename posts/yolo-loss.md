---
title: Yolo中loss函数分析
categories:
  - 深度学习
date: 2019-03-08 15:21:49
mathjax: true
tags:
-   Yolo
-   Tensorflow
-   目标检测
---

今天又回顾了一下`yolo`中的`loss`函数,我对比了[`keras_yolo3`](https://github.com/qqwweee/keras-yolo3),[`keras-yolo2`](https://github.com/experiencor/keras-yolo2),以及`yolo`作者的实现.又有了一些新发现.

<!--more-->

# 总结
这里我首先总结一下,在做`yolo`的`box`回归的时候,都是要考虑到预测出`box`与真实`box`的`iou`,计算出`iou score`,并通过阈值来判断出`ignore mask`去除一些不需要的`noobj loss`.并且做`loss`的时候都是使用以`cell`大小为尺度的数值进行计算.但是对于哪一个效果好，我是无法评价的.

# keras_yolo3中的实现

在这个项目中,作者使用`while`来计算`pred box`与`true box`的`iou score`.他会把其他的所有`pred box`与`true box`计算`iou`.

```python
def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
```
我的实现类似这种方式,但是我使用的是矩阵的形式去计算的.

# keras_yolo2中的实现

他是首先把`true box`和对应位置的`pred box`计算出`iou score`,这个时候把`true box`的置信度乘上`iou score`,因为`iou score`是在`[0-1]`之间的,所以在计算置信度误差的时候计算与`pred_box_conf`的平方差就会更大,相当于加大惩罚力度:
```python
true_box_conf = iou_scores * y_true[..., 4]
.
.
.
loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
```

接下来分析`conf_mask`,他是把`pred box`和所有的`true box`计算`iou score`,然后获得对应的`pred box`的`best iou`.然后`best iou`小于阈值的`pred box`则是需要进行惩罚.其他的格子不需要进行惩罚.

```python
conf_mask  = tf.zeros(mask_shape)
.
.
.

conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * self.no_object_scale

# penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
conf_mask = conf_mask + y_true[..., 4] * self.object_scale
```


总体来说这个`loss`的实现多了一个点,调整了`true_box_conf`的值,但是我使用的误差计算是`sigmoid cross`所以无法使用.


# 原版yolo

原版的`loss`在`box confidence`也做了一些调整,作者首先也是计算每个`pred box`与所有的`true box`的`iou score`然后获得`best iou`,当大于阈值时去除对应的`noobj loss`.


接下来就开始有所变化了,作者遍历所有的`true box`,将每个`true box`与对应`cell`的5个`pred box`计算`iou score`然后获得`best iou`以及`best_n`(bast iou对应的anchor),接下来计算`obj loss`、`coordinate loss`、`class loss`就是单独针对那个`anchor`与`true box`进行计算的.

下面是我写的部分代码注释:

```c
void forward_region_layer(const layer l, network net)
{
    int i,j,b,t,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            if(!l.background) activate_array(l.output + index,   l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords + 1);
            if(!l.softmax && !l.softmax_tree) activate_array(l.output + index, l.classes*l.w*l.h, LOGISTIC);
        }
    }
    if (l.softmax_tree){
        int i;
        int count = l.coords + 1;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, l.n*l.w*l.h, 1, l.n*l.w*l.h, l.temperature, l.output + count);
            count += group_size;
        }
    } else if (l.softmax){
        int index = entry_index(l, 0, 0, l.coords + !l.background);
        softmax_cpu(net.input + index, l.classes + l.background, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output + index);
    }
#endif

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
    float avg_iou = 0;
    float recall = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        if(l.softmax_tree){
            int onlyclass = 0;
            for(t = 0; t < 30; ++t){
                box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
                if(!truth.x) break;
                int class = net.truth[t*(l.coords + 1) + b*l.truths + l.coords];
                float maxp = 0;
                int maxi = 0;
                if(truth.x > 100000 && truth.y > 100000){
                    for(n = 0; n < l.n*l.w*l.h; ++n){
                        int class_index = entry_index(l, b, n, l.coords + 1);
                        int obj_index = entry_index(l, b, n, l.coords);
                        float scale =  l.output[obj_index];
                        l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                        float p = scale*get_hierarchy_probability(l.output + class_index, l.softmax_tree, class, l.w*l.h);
                        if(p > maxp){
                            maxp = p;
                            maxi = n;
                        }
                    }
                    int class_index = entry_index(l, b, maxi, l.coords + 1);
                    int obj_index = entry_index(l, b, maxi, l.coords);
                    delta_region_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat, !l.softmax);
                    if(l.output[obj_index] < .3) l.delta[obj_index] = l.object_scale * (.3 - l.output[obj_index]);
                    else  l.delta[obj_index] = 0;
                    l.delta[obj_index] = 0;
                    ++class_count;
                    onlyclass = 1;
                    break;
                }
            }
            if(onlyclass) continue;
        }
        //* 遍历所有格子和box,计算每个格子和真实box的iou
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
                    float best_iou = 0;
                    // * 找到每个ground truth与这个pred box最大的iou
                    for(t = 0; t < 30; ++t){
                        box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                        }
                    }
                    //* 首先先把所有的格子当成没有目标来计算损失
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, l.coords);
                    avg_anyobj += l.output[obj_index]; //*误差累积 
                    // * 计算loss
                    l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                    if(l.background) l.delta[obj_index] = l.noobject_scale * (1 - l.output[obj_index]);
                    if (best_iou > l.thresh) { // * 如果最大iou大于阈值,说明预测出有目标
                        l.delta[obj_index] = 0; //* 删除这个noobj loss
                    }
                    // *已经训练12800张图片之后,那么直接把预测值当做真实值
                    if(*(net.seen) < 12800){
                        box truth = {0};
                        truth.x = (i + .5)/l.w;
                        truth.y = (j + .5)/l.h;
                        truth.w = l.biases[2*n]/l.w;
                        truth.h = l.biases[2*n+1]/l.h;
                        delta_region_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, l.delta, .01, l.w*l.h);
                    }
                }
            }
        }
        // * 最多30个真实预测对象
        for(t = 0; t < 30; ++t){
            box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
            
            if(!truth.x) break;//* 没有对象就退出
            float best_iou = 0;
            int best_n = 0;
            // * i,j是真实物体在图片中的位置
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = 0;
            truth_shift.y = 0;
            // * l.n是每个格子的anchor box数 找到真实的box对应的5个pred box
            for(n = 0; n < l.n; ++n){
                int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
                // * 预测出来的对象边框= 预测出来的值 * anchor box 大小/ 图像的大小
                if(l.bias_match){
                    pred.w = l.biases[2*n]/l.w;
                    pred.h = l.biases[2*n+1]/l.h;
                }
                pred.x = 0;
                pred.y = 0;
                // * 计算预测框与真实框的iou NOTE 为了便于计算,
                // * 他把对应的预测的x,y都设成0,只看w h,因为他们都是同一个
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou; //*找到最大iou
                    best_n = n;     //*与其所在的
                }
            }
            // *找到best_n对应box
            int box_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 0);
            // * 计算best_n 和 truth的iou
            float iou = delta_region_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, l.delta, l.coord_scale *  (2 - truth.w*truth.h), l.w*l.h);
            // ? 这里
            if(l.coords > 4){
                int mask_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 4);
                delta_region_mask(net.truth + t*(l.coords + 1) + b*l.truths + 5, l.output, l.coords - 4, mask_index, l.delta, l.w*l.h, l.mask_scale);
            }
            // *iou 大于 0.5 召回数加1
            if(iou > .5) recall += 1;
            avg_iou += iou;
            // * 获得目标位置
            int obj_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords);
            avg_obj += l.output[obj_index];
            // * 置信度误差= 尺度 * (1 - 输出置信度)
            l.delta[obj_index] = l.object_scale * (1 - l.output[obj_index]);
            if (l.rescore) {
                l.delta[obj_index] = l.object_scale * (iou - l.output[obj_index]);
            }
            // * 这里是无目标的 置信度误差= 尺度 * (0 - 输出置信度)
            if(l.background){
                l.delta[obj_index] = l.object_scale * (0 - l.output[obj_index]);
            }
            // * 获得真实的类对象
            int class = net.truth[t*(l.coords + 1) + b*l.truths + l.coords];
            if (l.map) class = l.map[class];
            // * 获得预测的类对象
            int class_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords + 1);
            // * 交叉熵 计算类别误差
            delta_region_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat, !l.softmax);
            ++count;
            ++class_count;
        }
    }
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, count);
}
```




# 思考

现在我觉得我可以采用`keras yolov2`的方法，计算出`pred box`与对应位置的`true box`的`iou score`，然后根据`iou score`可以加大对`pred box`的惩罚～