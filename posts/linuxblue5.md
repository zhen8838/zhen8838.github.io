---
title: Raspi蓝牙:播放与录音
categories:
  - 边缘计算
date: 2019-02-27 10:11:40
tags:
-   蓝牙
-   树莓派
-   Linux
---

距离上一篇[文章](https://zhen8838.github.io/2018/06/23/linuxblue4/)已经过去太久了,现在尝试使用树莓派的蓝牙录音.

<!--more-->


# 添加用户到蓝牙组

```sh
sudo usermod -G bluetooth -a pi 
sudo reboot 
```

# 安装软件包

```sh
sudo apt-get install pulseaudio pulseaudio-module-bluetooth
```


# 搜索配对

其中`power on`是必须的,否则是无法`connect`成功的.
```sh
bluetoothctl
power on
agent on
default-agent
scan on
```
开始搜索后,先`quit`然后启动`pulseaudio`:

```sh
quit
sudo killall bluealsa
pulseaudio --start
```

然后回来链接设备,可能还要输入下`scan on`:
```sh
pair xx:xx:xx:xx:xx:xx
trust xx:xx:xx:xx:xx:xx
connect xx:xx:xx:xx:xx:xx
scan off
```

# 配置A2DP

```sh
pacmd list-cards
pacmd set-card-profile bluez_card.xx_xx_xx_xx_xx_xx a2dp_sink
pacmd set-default-sink bluez_sink.xx_xx_xx_xx_xx_xx.a2dp_sink
```

然后使用`paplay`播放声音

# 配置HSP

这里要用官方的命令输入到蓝牙中才可以正确配置.
```sh
sudo hcitool cmd 0x3F 0x01C 0x01 0x02 0x00 0x01 0x01
pacmd set-card-profile bluez_card.xx_xx_xx_xx_xx_xx headset_head_unit
pacmd set-default-sink bluez_sink.xx_xx_xx_xx_xx_xx.headset_head_unit
pacmd set-default-source bluez_source.xx_xx_xx_xx_xx_xx.headset_head_unit
```

# 录音以及播放~

```sh
parecord -v voice.wav
paplay -v voice.wav
```
然后就成功了.接下来准备做一下蓝牙录音编程相关的工作.