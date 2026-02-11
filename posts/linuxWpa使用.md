---
title: linux wpa wifi自动连接
date: 2018-05-29 16:53:42
tags: 
-   Linux
-   香橙派
categories: 
-   边缘计算
---


我使用的板子是OrangePI zero Plus2，基于全志H5，安装好了armbian系统。
现在要使用wpa进行自动wifi连接，并且固定ip。

<!--more-->


## 编辑连接信息
在 /etc/network/ 目录下创建 wifi 热点配置文件 wpa_supplication.conf,添加内容如下：
```sh
ctrl_interface=/var/run/wpa_supplicant
ctrl_interface_group=0
ap_scan=1
network={
    ssid="wifi名字"
    scan_ssid=1
    key_mgmt=WPA-EAP WPA-PSK IEEE8021X NONE
    pairwise=TKIP CCMP
    group=CCMP TKIP WEP104 WEP40
    psk="wifi密码"
    priority=5
}
```

## 添加静态地址

编辑`/etc/network/interfaces`:
```git
# Network is managed by Network manager
auto lo
iface lo inet loopback
+ auto wlan0
+ iface wlan0 inet static
+ address 10.42.0.143
+ netmask 255.0.0.0
```

## 连接wifi

```sh
root@H5:~# wpa_supplicant -Dnl80211 -iwlan0 -c /etc/network/wpa_supplication.conf -B
```