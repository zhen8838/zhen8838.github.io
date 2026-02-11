---
title: 排序算法小集
date: 2018-09-01 09:18:10
tags:
-   排序
categories:
-   数据结构
---


直接上程序

<!--more-->


## main.cpp

```cpp
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

int TEST1[800]= {
    985, 827, 960, 29,  410, 360, 931, 831, 845, 668, 199, 766, 72,  604, 470,
    877, 855, 228, 614, 507, 627, 388, 46,  949, 911, 519, 479, 636, 150, 195,
    189, 135, 374, 501, 165, 785, 213, 448, 616, 59,  468, 815, 177, 541, 419,
    648, 770, 627, 228, 736, 486, 207, 124, 532, 157, 388, 404, 988, 376, 554,
    184, 565, 41,  558, 419, 558, 695, 632, 359, 311, 43,  827, 479, 221, 368,
    250, 221, 138, 877, 449, 875, 715, 9,   351, 248, 518, 91,  652, 506, 468,
    558, 42,  385, 599, 953, 804, 158, 648, 789, 517, 312, 832, 696, 143, 405,
    417, 393, 626, 907, 623, 428, 134, 338, 437, 486, 938, 307, 577, 942, 165,
    397, 500, 208, 783, 452, 513, 939, 962, 513, 80,  831, 177, 913, 527, 320,
    318, 944, 66,  297, 852, 41,  725, 986, 731, 514, 824, 670, 173, 754, 612,
    338, 151, 113, 898, 286, 917, 411, 226, 879, 277, 306, 710, 454, 571, 237,
    127, 242, 182, 545, 539, 386, 586, 616, 372, 317, 482, 197, 339, 655, 951,
    952, 993, 454, 417, 244, 741, 334, 7,   967, 213, 284, 625, 275, 91,  549,
    512, 570, 791, 46,  115, 682, 432, 53,  298, 157, 370, 132, 706, 710, 139,
    657, 14,  484, 463, 431, 728, 204, 117, 736, 523, 682, 372, 149, 957, 463,
    50,  469, 385, 841, 868, 500, 523, 652, 553, 173, 809, 276, 657, 515, 338,
    796, 524, 704, 280, 988, 135, 361, 544, 604, 97,  68,  286, 821, 569, 243,
    285, 619, 64,  670, 812, 284, 523, 335, 937, 428, 860, 98,  56,  517, 614,
    746, 665, 490, 450, 297, 830, 585, 658, 375, 541, 107, 443, 827, 929, 364,
    422, 566, 983, 839, 588, 795, 475, 463, 482, 412, 892, 342, 863, 300, 211,
    829, 47,  876, 319, 497, 173, 150, 435, 832, 877, 976, 291, 672, 156, 220,
    36,  930, 138, 19,  769, 727, 166, 597, 190, 648, 361, 434, 342, 224, 735,
    553, 53,  782, 781, 725, 279, 954, 875, 714, 138, 104, 691, 430, 776, 199,
    2,   812, 129, 493, 183, 251, 220, 349, 848, 762, 349, 209, 197, 691, 434,
    284, 596, 839, 66,  377, 564, 697, 331, 791, 764, 822, 895, 807, 604, 671,
    6,   958, 483, 135, 451, 18,  386, 671, 367, 234, 786, 68,  796, 335, 759,
    230, 619, 355, 69,  37,  84,  634, 86,  416, 425, 850, 590, 673, 657, 194,
    344, 663, 152, 180, 151, 604, 198, 537, 627, 918, 124, 765, 986, 920, 100,
    98,  502, 71,  805, 923, 460, 890, 909, 547, 658, 687, 397, 248, 360, 407,
    794, 56,  422, 946, 236, 573, 902, 787, 463, 882, 57,  939, 647, 395, 859,
    100, 493, 361, 171, 299, 636, 632, 541, 546, 179, 199, 233, 928, 447, 593,
    687, 241, 649, 110, 539, 238, 35,  794, 377, 498, 676, 434, 437, 675, 181,
    648, 775, 675, 361, 947, 326, 998, 931, 867, 896, 462, 66,  129, 742, 865,
    74,  430, 458, 75,  540, 349, 665, 927, 143, 42,  778, 171, 828, 567, 847,
    10,  216, 622, 37,  929, 921, 715, 279, 852, 582, 175, 666, 0,   656, 409,
    217, 730, 839, 27,  158, 731, 376, 823, 658, 872, 866, 436, 395, 46,  356,
    242, 408, 924, 865, 445, 853, 138, 160, 485, 991, 94,  660, 657, 446, 317,
    418, 15,  399, 257, 42,  557, 988, 771, 381, 999, 995, 599, 787, 390, 645,
    143, 633, 406, 67,  850, 851, 273, 988, 364, 758, 331, 810, 770, 341, 609,
    87,  759, 624, 487, 369, 667, 396, 357, 790, 777, 708, 785, 728, 496, 527,
    726, 639, 512, 132, 59,  362, 983, 684, 703, 699, 442, 386, 862, 212, 79,
    471, 652, 191, 95,  491, 560, 114, 887, 269, 256, 17,  978, 393, 745, 474,
    921, 471, 465, 433, 603, 524, 148, 939, 208, 203, 990, 2,   589, 852, 567,
    669, 675, 219, 212, 123, 710, 772, 589, 949, 41,  846, 966, 19,  239, 712,
    493, 512, 535, 311, 946, 139, 835, 446, 430, 396, 649, 420, 750, 590, 625,
    317, 611, 300, 536, 823, 423, 598, 595, 13,  548, 989, 211, 866, 8,   450,
    930, 854, 315, 466, 165, 261, 605, 352, 59,  35,  100, 708, 807, 851, 298,
    432, 168, 910, 733, 57,  85,  508, 7,   681, 873, 907, 22,  84,  774, 30,
    535, 704, 236, 202, 170, 753, 463, 127, 458, 874, 514, 558, 934, 322, 761,
    232, 754, 282, 494, 839, 691, 580, 348, 698, 613, 221, 958, 635, 658, 84,
    17,  545, 140, 606, 747, 663, 711, 562, 790, 169, 436, 657, 80,  370, 979,
    193, 954, 85,  475, 801, 925, 518, 381, 625, 569, 346, 198, 527, 333, 856,
    611, 350, 401, 751, 308, 148, 414, 20,  710, 557, 541, 498, 214, 973, 868,
    545, 519, 823, 630, 346};

/**
 * @brief  冒泡法与插入法 交换次数相同 并且与数组中逆序数个数相同！
 *
 *      定理：任意N个不同元素组成的序列平均具有N(N-1)/4个逆序对
 *      定理：任何以交换相邻元素的排序算法，其平均时间复杂度为Ω(N^2)
 **/
void BubbleSort(int *array, int length) {
    int temp;
    for (int i= length; i > 0; i--) {
        for (int j= 0; j < (i - 1); j++) {
            if (array[j] > array[j + 1]) {
                temp        = array[j];
                array[j]    = array[j + 1];
                array[j + 1]= temp;
            }
        }
    }
}

void InsertionSort(int *array, int length) {
    int Tmp, i;
    for (int p= 1; p < length; p++) {
        Tmp= array[p]; /* 取值 */
        for (i= p; i > 0 && (array[i - 1] > Tmp); i--) {
            array[i]= array[i - 1]; /* 向后移动 */
        }
        array[i]= Tmp; /* 插入元素 */
    }
}

/**
 * @brief 希尔排序
 *        此方法是去间隔进行插入排序。
 *        可行性在于下一次增量排序不会影响上一次增量排序
 **/
void ShellSort(int *array, int length) {
    int Tmp, i;
    for (int j= length / 2; j > 0; j/= 2) {
        for (int p= j; p < length; p++) {
            Tmp= array[p]; /* 取值 */
            for (i= p; i >= j && (array[i - j] > Tmp); i-= j) {
                array[i]= array[i - j]; /* 向后移动 */
            }
            array[i]= Tmp; /* 插入元素 */
        }
    }
}

/**
 * @brief 交换元素
 **/
void Swap(int *a, int *b) {
    int tmp= *b;
    *b     = *a;
    *a     = tmp;
}
/**
 * @brief 将大的元素下沉
 **/
void PercDown(int *array, int index, int length) {
    int temp= array[index];
    for (int k= index * 2 + 1; k < length; k= k * 2 + 1) {
        //如果右边值大于左边值，指向右边
        if (k + 1 < length && array[k] < array[k + 1]) {
            k++;
        }
        //如果子节点大于父节点，将子节点值赋给父节点,并以新的子节点作为父节点（不用进行交换）
        if (array[k] > temp) {
            array[index]= array[k];
            index       = k;
        } else
            break;
    }
    // put the value in the final position
    array[index]= temp;
}
/**
 * @brief 堆排序
 *        堆排序是选择排序的改进
 *        构建成最大堆，再调整顺序
 **/
void HeapSort(int *array, int length) {
    /* 构建最大堆--数组形式 */
    for (int i= length / 2; i >= 0; i--) {
        PercDown(array, i, length);
    }
    for (int i= length - 1; i > 0; i--) {
        /* 移除最大元素 */
        Swap(&array[0], &array[i]);
        /* 以根节点为0下沉，且数组长度逐渐缩短 */
        PercDown(array, 0, i);
    }
}

int Median3(int *array, int left, int right) {
    int center= (left + right) / 2;
    if (array[left] > array[center]) {
        Swap(&array[left], &array[center]);
    }
    if (array[left] > array[right]) {
        Swap(&array[left], &array[right]);
    }
    if (array[center] > array[right]) {
        Swap(&array[center], &array[right]);
    }
    Swap(&array[center], &array[right - 1]);
    return array[right - 1];
}
void QuickSort(int *array, int left, int right) {

    if (30 <= right - left) {
        int pivot= Median3(array, left, right);
        int i    = left;
        int j    = right - 1;
        while (1) {
            while (array[++i] < pivot) {
            }
            while (array[--j] > pivot) {
            }
            if (i < j) {
                Swap(&array[i], &array[j]);
            } else {
                break;
            }
        }
        Swap(&array[i], &array[right - 1]);
        QuickSort(array, left, i - 1);
        QuickSort(array, i + 1, right);
    } else {
        InsertionSort(array + left, right - left + 1);
    }
}

/**
 * @brief 快速排序
 **/
void QuickSort(int *array, int length) {
    QuickSort(array, 0, length - 1);
}


int main(int argc, char const *argv[]) {
    struct timespec tpstart;
    struct timespec tpend;
    int *A= (int *)malloc(sizeof(TEST1));
    int *B= (int *)malloc(sizeof(TEST1));
    int *C= (int *)malloc(sizeof(TEST1));
    int *D= (int *)malloc(sizeof(TEST1));
    int *E= (int *)malloc(sizeof(TEST1));

    memcpy(A, TEST1, sizeof(TEST1));
    memcpy(B, TEST1, sizeof(TEST1));
    memcpy(C, TEST1, sizeof(TEST1));
    memcpy(D, TEST1, sizeof(TEST1));
    memcpy(E, TEST1, sizeof(TEST1));
    clock_gettime(CLOCK_MONOTONIC, &tpstart);
    BubbleSort(A, 800);
    clock_gettime(CLOCK_MONOTONIC, &tpend);
    printf("冒泡排序：%ld us\n", (tpend.tv_nsec - tpstart.tv_nsec) / 1000);

    clock_gettime(CLOCK_MONOTONIC, &tpstart);
    InsertionSort(B, 800);
    clock_gettime(CLOCK_MONOTONIC, &tpend);
    printf("插入排序：%ld us\n", (tpend.tv_nsec - tpstart.tv_nsec) / 1000);

    clock_gettime(CLOCK_MONOTONIC, &tpstart);
    ShellSort(C, 800);
    clock_gettime(CLOCK_MONOTONIC, &tpend);
    printf("希尔排序：%ld us\n", (tpend.tv_nsec - tpstart.tv_nsec) / 1000);

    clock_gettime(CLOCK_MONOTONIC, &tpstart);
    HeapSort(D, 800);
    clock_gettime(CLOCK_MONOTONIC, &tpend);
    printf("堆排序：%ld us\n", (tpend.tv_nsec - tpstart.tv_nsec) / 1000);

    clock_gettime(CLOCK_MONOTONIC, &tpstart);
    QuickSort(E, 800);
    clock_gettime(CLOCK_MONOTONIC, &tpend);
    printf("快速排序：%ld us\n", (tpend.tv_nsec - tpstart.tv_nsec) / 1000);

    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
    return 0;
}
```



## 执行
```sh
➜  sort clang++ -g ./main.cpp && ./a.out
冒泡排序：863 us
插入排序：336 us
希尔排序：84 us
堆排序：76 us
快速排序：54 us
```