---
title: OrangePI蓝牙：蓝牙耳机
date: 2018-06-22 17:58:16
tags: 
-   Linux
-   树莓派
-   蓝牙
categories: 
-   边缘计算 
---

上一篇[文章](https://zhen8838.github.io/2018/06/20/Linuxblue3/)描述了如何与串口蓝牙进行通信，这一次记录如何使用蓝牙耳机。

<!--more-->


##  配对蓝牙耳机

首先我按照archlinux中的配对方法来进行配对。
```sh
root@H5:~# bluetoothctl
[bluetooth]# power on
[bluetooth]# agent on
[bluetooth]# default-agent
[bluetooth]# scan on
[NEW] Device 1C:52:16:02:40:9F 小米蓝牙耳机青春版
[bluetooth]# scan off
[bluetooth]# pair 1C:52:16:02:40:9F
Attempting to pair with 1C:52:16:02:40:9F
[CHG] Device 1C:52:16:02:40:9F Connected: yes
[CHG] Device 1C:52:16:02:40:9F UUIDs: 00001101-0000-1000-8000-00805f9b34fb
[CHG] Device 1C:52:16:02:40:9F UUIDs: 00001108-0000-1000-8000-00805f9b34fb
[CHG] Device 1C:52:16:02:40:9F UUIDs: 0000110b-0000-1000-8000-00805f9b34fb
[CHG] Device 1C:52:16:02:40:9F UUIDs: 0000110c-0000-1000-8000-00805f9b34fb
[CHG] Device 1C:52:16:02:40:9F UUIDs: 0000110e-0000-1000-8000-00805f9b34fb
[CHG] Device 1C:52:16:02:40:9F UUIDs: 0000111e-0000-1000-8000-00805f9b34fb
[CHG] Device 1C:52:16:02:40:9F ServicesResolved: yes
[CHG] Device 1C:52:16:02:40:9F Paired: yes
Pairing successful
[CHG] Device 1C:52:16:02:40:9F ServicesResolved: no
[CHG] Device 1C:52:16:02:40:9F Connected: no
```
到这里就出现问题了。

## 检查问题

首先查看错误日志
```sh
systemctl status bluetooth
```
错误日志如下：
```sh
● bluetooth.service - Bluetooth service
   Loaded: loaded (/lib/systemd/system/bluetooth.service; enabled; vendor preset
: enabled)
   Active: active (running) since Fri 2018-06-22 11:02:23 UTC; 12min ago
     Docs: man:bluetoothd(8)
 Main PID: 933 (bluetoothd)
   Status: "Running"
    Tasks: 1 (limit: 4915)
   CGroup: /system.slice/bluetooth.service
           └─933 /usr/lib/bluetooth/bluetoothd

Jun 22 11:02:23 H5 systemd[1]: Starting Bluetooth service...
Jun 22 11:02:23 H5 bluetoothd[933]: Bluetooth daemon 5.43
Jun 22 11:02:23 H5 systemd[1]: Started Bluetooth service.
Jun 22 11:02:23 H5 bluetoothd[933]: Starting SDP server
Jun 22 11:02:23 H5 bluetoothd[933]: Bluetooth management interface 1.14 initiali
zed
Jun 22 11:02:23 H5 bluetoothd[933]: Failed to obtain handles for "Service Change
d" characteristic
Jun 22 11:02:23 H5 bluetoothd[933]: Sap driver initialization failed.
Jun 22 11:02:23 H5 bluetoothd[933]: sap-server: Operation not permitted (1)
Jun 22 11:02:23 H5 bluetoothd[933]: Failed to set privacy: Rejected (0x0b)
Jun 22 11:11:01 H5 bluetoothd[933]: a2dp-sink profile connect failed for 1C:52:1
6:02:40:9F: Protocol not available
```
这里发现问题出现在`a2dp-sink profile connect failed for 1C:52:1
6:02:40:9F: Protocol not available`所以需要查找一波资料。

## 解决方法
先试试安装软件包,这里都是root权限下使用
```sh
apt-get install pulseaudio-module-bluetooth
killall pulseaudio
pulseaudio --start --log-target=syslog
systemctl restart bluetooth
```


## 出现问题
```sh
● bluetooth.service - Bluetooth service
   Loaded: loaded (/lib/systemd/system/bluetooth.service; enabled; vendor preset: enabled)
   Active: active (running) since Mon 2018-07-09 02:27:15 UTC; 8min ago
     Docs: man:bluetoothd(8)
 Main PID: 3733 (bluetoothd)
   Status: "Running"
    Tasks: 1 (limit: 4915)
   CGroup: /system.slice/bluetooth.service
           └─3733 /usr/lib/bluetooth/bluetoothd

Jul 09 02:27:14 H5 systemd[1]: Starting Bluetooth service...
Jul 09 02:27:14 H5 bluetoothd[3733]: Bluetooth daemon 5.43
Jul 09 02:27:15 H5 systemd[1]: Started Bluetooth service.
Jul 09 02:27:15 H5 bluetoothd[3733]: Starting SDP server
Jul 09 02:27:15 H5 bluetoothd[3733]: Bluetooth management interface 1.14 initialized
Jul 09 02:27:15 H5 bluetoothd[3733]: Failed to obtain handles for "Service Changed" characteristic
Jul 09 02:27:15 H5 bluetoothd[3733]: Sap driver initialization failed.
Jul 09 02:27:15 H5 bluetoothd[3733]: sap-server: Operation not permitted (1)
Jul 09 02:27:15 H5 bluetoothd[3733]: Endpoint registered: sender=:1.17 path=/MediaEndpoint/A2DPSink
```

## 解决问题

这里的`sap-server: Operation not permitted (1)`是`SIM Access Profile`,需要禁用他。

- 打开`/etc/systemd/system/bluetooth.target.wants/bluetooth.service`
- 改变：`ExecStart=/usr/lib/bluetooth/bluetoothd`为 `ExecStart=/usr/lib/bluetooth/bluetoothd --noplugin=sap`
- 重载systemctl `systemctl daemon-reload`
- 重启蓝牙服务 `systemctl restart bluetooth`
- 查看状态 
  ```sh
  ● bluetooth.service - Bluetooth service
   Loaded: loaded (/lib/systemd/system/bluetooth.service; enabled; vendor preset
   Active: active (running) since Mon 2018-07-09 03:08:12 UTC; 3s ago
     Docs: man:bluetoothd(8)
  Main PID: 1768 (bluetoothd)
   Status: "Running"
    Tasks: 1 (limit: 4915)
   CGroup: /system.slice/bluetooth.service
           └─1768 /usr/lib/bluetooth/bluetoothd --noplugin=sap

  ```

##   连接成功
现在打开`bluetoothctl`就可以连接了
```sh
bluetoothctl
[NEW] Controller 43:29:B1:55:01:01 H5 [default]
[NEW] Device 30:22:00:00:8F:67 porbox wireless
```
开启代理
```sh
agent on
default-agent
```
连接蓝牙耳机
```sh
connect 30:22:00:00:8F:67
Attempting to connect to 30:22:00:00:8F:67
[CHG] Device 30:22:00:00:8F:67 Connected: yes
Connection successful
[CHG] Device 30:22:00:00:8F:67 ServicesResolved: yes
```

## 配置输出

1.  **查看声卡**
  ```sh
  pactl list cards
  ```
  出现了如下

  ```sh
  Card #0
          Name: bluez_card.30_22_00_00_8F_67
          Driver: module-bluez5-device.c
          Owner Module: 22
          Properties:
                  device.description = "porbox wireless"
                  device.string = "30:22:00:00:8F:67"
                  device.api = "bluez"
                  device.class = "sound"
                  device.bus = "bluetooth"
                  device.form_factor = "hands-free"
                  bluez.path = "/org/bluez/hci0/dev_30_22_00_00_8F_67"
                  bluez.class = "0x240408"
                  bluez.alias = "porbox wireless"
                  device.icon_name = "audio-handsfree-bluetooth"
                  device.intended_roles = "phone"
          Profiles:
                  headset_head_unit: Headset Head Unit (HSP/HFP) (sinks: 1, sources: 1, priority: 20, available: yes)
                  a2dp_sink: High Fidelity Playback (A2DP Sink) (sinks: 1, sources: 0, priority: 10, available: yes)
                  off: Off (sinks: 0, sources: 0, priority: 0, available: yes)
          Active Profile: off
          Ports:
                  handsfree-output: Handsfree (priority: 0, latency offset: 0 usec)
                          Part of profile(s): headset_head_unit, a2dp_sink
                  handsfree-input: Handsfree (priority: 0, latency offset: 0 usec)
                          Part of profile(s): headset_head_unit

  ```

2.  **单声音配置**

  查看节点

  ```sh
  pacmd list-sinks
  ```
  出现

  ```sh
  1 sink(s) available.
  * index: 0
        name: <auto_null>
        driver: <module-null-sink.c>
        flags: DECIBEL_VOLUME LATENCY FLAT_VOLUME DYNAMIC_LATENCY
        state: SUSPENDED
        suspend cause: IDLE
        priority: 1000
        volume: front-left: 65536 / 100% / 0.00 dB,   front-right: 65536 / 100% / 0.00 dB
                balance 0.00
        base volume: 65536 / 100% / 0.00 dB
        volume steps: 65537
        muted: no
        current latency: 0.00 ms
        max request: 344 KiB
        max rewind: 344 KiB
        monitor source: 0
        sample spec: s16le 2ch 44100Hz
        channel map: front-left,front-right
                     Stereo
        used by: 0
        linked by: 0
        configured latency: 0.00 ms; range is 0.50 .. 2000.00 ms
        module: 13
        properties:
                device.description = "Dummy Output"
                device.class = "abstract"
                device.icon_name = "audio-card"
  ```
  
  设置默认连接

  ```sh
  pacmd set-default-sink 0
  pacmd set-card-profile 0 a2dp_sink
  ```

3.  **查看信息**

  ```sh
  pacmd info
  ```

  出现

  ```sh
  Memory blocks currently allocated: 1, size: 63.9 KiB.
  Memory blocks allocated during the whole lifetime: 121, size: 2.6 MiB.
  Memory blocks imported from other processes: 0, size: 0 B.
  Memory blocks exported to other processes: 0, size: 0 B.
  Total sample cache size: 0 B.
  Default sample spec: s16le 2ch 44100Hz
  Default channel map: front-left,front-right
  Default sink name: bluez_sink.30_22_00_00_8F_67.a2dp_sink
  Default source name: bluez_sink.30_22_00_00_8F_67.a2dp_sink.monitor
  Memory blocks of type POOL: 1 allocated/75 accumulated.
  Memory blocks of type POOL_EXTERNAL: 0 allocated/0 accumulated.
  Memory blocks of type APPENDED: 0 allocated/0 accumulated.
  Memory blocks of type USER: 0 allocated/0 accumulated.
  Memory blocks of type FIXED: 0 allocated/0 accumulated.
  Memory blocks of type IMPORTED: 0 allocated/46 accumulated.
  23 module(s) loaded.
      index: 0
          name: <module-device-restore>
          argument: <>
          used: -1
          load once: yes
          properties:
                  module.author = "Lennart Poettering"
                  module.description = "Automatically restore the volume/mute state of devices"
                  module.version = "10.0"
      index: 1
          name: <module-stream-restore>
          argument: <>
          used: -1
          load once: yes
          properties:
                  module.author = "Lennart Poettering"
                  module.description = "Automatically restore the volume/mute/device state of streams"
                  module.version = "10.0"
      index: 2
          name: <module-card-restore>
          argument: <>
          used: -1
          load once: yes
          properties:
                  module.author = "Lennart Poettering"
                  module.description = "Automatically restore profile of cards"
                  module.version = "10.0"
      index: 3
          name: <module-augment-properties>
          argument: <>
          used: -1
          load once: yes
          properties:
                  module.author = "Lennart Poettering"
                  module.description = "Augment the property sets of streams with additional static information"
                  module.version = "10.0"
      index: 4
          name: <module-switch-on-port-available>
          argument: <>
          used: -1
          load once: no
          properties:

      index: 5
          name: <module-udev-detect>
          argument: <>
          used: -1
          load once: yes
          properties:
                  module.author = "Lennart Poettering"
                  module.description = "Detect available audio hardware and load matching drivers"
                  module.version = "10.0"
      index: 6
          name: <module-bluetooth-policy>
          argument: <>
          used: -1
          load once: yes
          properties:
                  module.author = "Frédéric Dalleau, Pali Rohár"
                  module.description = "Policy module to make using bluetooth devices out-of-the-box easier"
                  module.version = "10.0"
      index: 7
          name: <module-bluetooth-discover>
          argument: <>
          used: -1
          load once: yes
          properties:
                  module.author = "João Paulo Rechi Vita"
                  module.description = "Detect available Bluetooth daemon and load the corresponding discovery module"
                  module.version = "10.0"
      index: 8
          name: <module-bluez5-discover>
          argument: <>
          used: -1
          load once: yes
          properties:
                  module.author = "João Paulo Rechi Vita"
                  module.description = "Detect available BlueZ 5 Bluetooth audio devices and load BlueZ 5 Bluetooth audio drivers"
                  module.version = "10.0"
      index: 9
          name: <module-native-protocol-unix>
          argument: <>
          used: -1
          load once: no
          properties:
                  module.author = "Lennart Poettering"
                  module.description = "Native protocol (UNIX sockets)"
                  module.version = "10.0"
      index: 10
          name: <module-default-device-restore>
          argument: <>
          used: -1
          load once: yes
          properties:
                  module.author = "Lennart Poettering"
                  module.description = "Automatically restore the default sink and source"
                  module.version = "10.0"
      index: 11
          name: <module-rescue-streams>
          argument: <>
          used: -1
          load once: yes
          properties:
                  module.author = "Lennart Poettering"
                  module.description = "When a sink/source is removed, try to move its streams to the default sink/source"
                  module.version = "10.0"
      index: 12
          name: <module-always-sink>
          argument: <>
          used: -1
          load once: yes
          properties:
                  module.author = "Colin Guthrie"
                  module.description = "Always keeps at least one sink loaded even if it's a null one"
                  module.version = "10.0"
      index: 14
          name: <module-intended-roles>
          argument: <>
          used: -1
          load once: yes
          properties:
                  module.author = "Lennart Poettering"
                  module.description = "Automatically set device of streams based on intended roles of devices"
                  module.version = "10.0"
      index: 15
          name: <module-suspend-on-idle>
          argument: <>
          used: -1
          load once: yes
          properties:
                  module.author = "Lennart Poettering"
                  module.description = "When a sink/source is idle for too long, suspend it"
                  module.version = "10.0"
      index: 16
          name: <module-console-kit>
          argument: <>
          used: -1
          load once: yes
          properties:
                  module.author = "Lennart Poettering"
                  module.description = "Create a client for each ConsoleKit session of this user"
                  module.version = "10.0"
      index: 17
          name: <module-systemd-login>
          argument: <>
          used: -1
          load once: yes
          properties:
                  module.author = "Lennart Poettering"
                  module.description = "Create a client for each login session of this user"
                  module.version = "10.0"
      index: 18
          name: <module-position-event-sounds>
          argument: <>
          used: -1
          load once: yes
          properties:
                  module.author = "Lennart Poettering"
                  module.description = "Position event sounds between L and R depending on the position on screen of the widget triggering them."
                  module.version = "10.0"
      index: 19
          name: <module-role-cork>
          argument: <>
          used: -1
          load once: yes
          properties:
                  module.author = "Lennart Poettering"
                  module.description = "Mute & cork streams with certain roles while others exist"
                  module.version = "10.0"
      index: 20
          name: <module-filter-heuristics>
          argument: <>
          used: -1
          load once: yes
          properties:
                  module.author = "Colin Guthrie"
                  module.description = "Detect when various filters are desirable"
                  module.version = "10.0"
      index: 21
          name: <module-filter-apply>
          argument: <>
          used: -1
          load once: yes
          properties:
                  module.author = "Colin Guthrie"
                  module.description = "Load filter sinks automatically when needed"
                  module.version = "10.0"
      index: 22
          name: <module-bluez5-device>
          argument: <path=/org/bluez/hci0/dev_30_22_00_00_8F_67>
          used: 0
          load once: no
          properties:
                  module.author = "João Paulo Rechi Vita"
                  module.description = "BlueZ 5 Bluetooth audio sink and source"
                  module.version = "10.0"
      index: 23
          name: <module-cli-protocol-unix>
          argument: <>
          used: -1
          load once: no
          properties:
                  module.author = "Lennart Poettering"
                  module.description = "Command line interface protocol (UNIX sockets)"
                  module.version = "10.0"
  1 sink(s) available.
    * index: 1
          name: <bluez_sink.30_22_00_00_8F_67.a2dp_sink>
          driver: <module-bluez5-device.c>
          flags: HARDWARE DECIBEL_VOLUME LATENCY FLAT_VOLUME
          state: SUSPENDED
          suspend cause: IDLE
          priority: 9030
          volume: front-left: 65536 / 100% / 0.00 dB,   front-right: 65536 / 100% / 0.00 dB
                  balance 0.00
          base volume: 65536 / 100% / 0.00 dB
          volume steps: 65537
          muted: no
          current latency: 0.00 ms
          max request: 4 KiB
          max rewind: 0 KiB
          monitor source: 1
          sample spec: s16le 2ch 44100Hz
          channel map: front-left,front-right
                      Stereo
          used by: 0
          linked by: 0
          fixed latency: 48.22 ms
          card: 0 <bluez_card.30_22_00_00_8F_67>
          module: 22
          properties:
                  bluetooth.protocol = "a2dp_sink"
                  device.description = "porbox wireless"
                  device.string = "30:22:00:00:8F:67"
                  device.api = "bluez"
                  device.class = "sound"
                  device.bus = "bluetooth"
                  device.form_factor = "hands-free"
                  bluez.path = "/org/bluez/hci0/dev_30_22_00_00_8F_67"
                  bluez.class = "0x240408"
                  bluez.alias = "porbox wireless"
                  device.icon_name = "audio-handsfree-bluetooth"
                  device.intended_roles = "phone"
          ports:
                  handsfree-output: Handsfree (priority 0, latency offset 0 usec, available: unknown)
                          properties:

          active port: <handsfree-output>
  1 source(s) available.
    * index: 1
          name: <bluez_sink.30_22_00_00_8F_67.a2dp_sink.monitor>
          driver: <module-bluez5-device.c>
          flags: DECIBEL_VOLUME LATENCY
          state: SUSPENDED
          suspend cause: IDLE
          priority: 1030
          volume: front-left: 65536 / 100% / 0.00 dB,   front-right: 65536 / 100% / 0.00 dB
                  balance 0.00
          base volume: 65536 / 100% / 0.00 dB
          volume steps: 65537
          muted: no
          current latency: 0.00 ms
          max rewind: 0 KiB
          sample spec: s16le 2ch 44100Hz
          channel map: front-left,front-right
                      Stereo
          used by: 0
          linked by: 0
          fixed latency: 48.22 ms
          monitor_of: 1
          card: 0 <bluez_card.30_22_00_00_8F_67>
          module: 22
          properties:
                  device.description = "Monitor of porbox wireless"
                  device.class = "monitor"
                  device.string = "30:22:00:00:8F:67"
                  device.api = "bluez"
                  device.bus = "bluetooth"
                  device.form_factor = "hands-free"
                  bluez.path = "/org/bluez/hci0/dev_30_22_00_00_8F_67"
                  bluez.class = "0x240408"
                  bluez.alias = "porbox wireless"
                  device.icon_name = "audio-handsfree-bluetooth"
                  device.intended_roles = "phone"
  2 client(s) logged in.
      index: 0
          driver: <module-systemd-login.c>
          owner module: 17
          properties:
                  application.name = "Login Session 2"
                  systemd-login.session = "2"
      index: 28
          driver: <cli.c>
          owner module: 23
          properties:
                  application.name = "UNIX socket client"
  1 card(s) available.
      index: 0
          name: <bluez_card.30_22_00_00_8F_67>
          driver: <module-bluez5-device.c>
          owner module: 22
          properties:
                  device.description = "porbox wireless"
                  device.string = "30:22:00:00:8F:67"
                  device.api = "bluez"
                  device.class = "sound"
                  device.bus = "bluetooth"
                  device.form_factor = "hands-free"
                  bluez.path = "/org/bluez/hci0/dev_30_22_00_00_8F_67"
                  bluez.class = "0x240408"
                  bluez.alias = "porbox wireless"
                  device.icon_name = "audio-handsfree-bluetooth"
                  device.intended_roles = "phone"
          profiles:
                  headset_head_unit: Headset Head Unit (HSP/HFP) (priority 20, available: unknown)
                  a2dp_sink: High Fidelity Playback (A2DP Sink) (priority 10, available: unknown)
                  off: Off (priority 0, available: yes)
          active profile: <a2dp_sink>
          sinks:
                  bluez_sink.30_22_00_00_8F_67.a2dp_sink/#1: porbox wireless
          sources:
                  bluez_sink.30_22_00_00_8F_67.a2dp_sink.monitor/#1: Monitor of porbox wireless
          ports:
                  handsfree-output: Handsfree (priority 0, latency offset 0 usec, available: unknown)
                          properties:

                  handsfree-input: Handsfree (priority 0, latency offset 0 usec, available: unknown)
                          properties:

  0 sink input(s) available.
  0 source output(s) available.
  0 cache entrie(s) available.
  ```
  
  查看状态

  ```sh
  pacmd stat
  ```

  出现

  ```sh
  Memory blocks currently allocated: 1, size: 63.9 KiB.
  Memory blocks allocated during the whole lifetime: 121, size: 2.6 MiB.
  Memory blocks imported from other processes: 0, size: 0 B.
  Memory blocks exported to other processes: 0, size: 0 B.
  Total sample cache size: 0 B.
  Default sample spec: s16le 2ch 44100Hz
  Default channel map: front-left,front-right
  Default sink name: bluez_sink.30_22_00_00_8F_67.a2dp_sink
  Default source name: bluez_sink.30_22_00_00_8F_67.a2dp_sink.monitor
  Memory blocks of type POOL: 1 allocated/75 accumulated.
  Memory blocks of type POOL_EXTERNAL: 0 allocated/0 accumulated.
  Memory blocks of type APPENDED: 0 allocated/0 accumulated.
  Memory blocks of type USER: 0 allocated/0 accumulated.
  Memory blocks of type FIXED: 0 allocated/0 accumulated.
  Memory blocks of type IMPORTED: 0 allocated/46 accumulated.
  ```
  
  
##   测试蓝牙耳机

此时先查看连接
```sh
pactl list sinks
```
出现
```sh
Sink #1
        State: SUSPENDED
        Name: bluez_sink.30_22_00_00_8F_67.a2dp_sink
        Description: porbox wireless
        Driver: module-bluez5-device.c
        Sample Specification: s16le 2ch 44100Hz
        Channel Map: front-left,front-right
        Owner Module: 22
        Mute: no
        Volume: front-left: 65536 / 100% / 0.00 dB,   front-right: 65536 / 100% / 0.00 dB
                balance 0.00
        Base Volume: 65536 / 100% / 0.00 dB
        Monitor Source: bluez_sink.30_22_00_00_8F_67.a2dp_sink.monitor
        Latency: 0 usec, configured 0 usec
        Flags: HARDWARE DECIBEL_VOLUME LATENCY
        Properties:
                bluetooth.protocol = "a2dp_sink"
                device.description = "porbox wireless"
                device.string = "30:22:00:00:8F:67"
                device.api = "bluez"
                device.class = "sound"
                device.bus = "bluetooth"
                device.form_factor = "hands-free"
                bluez.path = "/org/bluez/hci0/dev_30_22_00_00_8F_67"
                bluez.class = "0x240408"
                bluez.alias = "porbox wireless"
                device.icon_name = "audio-handsfree-bluetooth"
                device.intended_roles = "phone"
        Ports:
                handsfree-output: Handsfree (priority: 0)
        Active Port: handsfree-output
        Formats:
                pcm
```
查看声卡

```sh
pactl list cards
```

出现

```sh
Card #0
        Name: bluez_card.30_22_00_00_8F_67
        Driver: module-bluez5-device.c
        Owner Module: 22
        Properties:
                device.description = "porbox wireless"
                device.string = "30:22:00:00:8F:67"
                device.api = "bluez"
                device.class = "sound"
                device.bus = "bluetooth"
                device.form_factor = "hands-free"
                bluez.path = "/org/bluez/hci0/dev_30_22_00_00_8F_67"
                bluez.class = "0x240408"
                bluez.alias = "porbox wireless"
                device.icon_name = "audio-handsfree-bluetooth"
                device.intended_roles = "phone"
        Profiles:
                headset_head_unit: Headset Head Unit (HSP/HFP) (sinks: 1, sources: 1, priority: 20, available: yes)
                a2dp_sink: High Fidelity Playback (A2DP Sink) (sinks: 1, sources: 0, priority: 10, available: yes)
                off: Off (sinks: 0, sources: 0, priority: 0, available: yes)
        Active Profile: a2dp_sink
        Ports:
                handsfree-output: Handsfree (priority: 0, latency offset: 0 usec)
                        Part of profile(s): headset_head_unit, a2dp_sink
                handsfree-input: Handsfree (priority: 0, latency offset: 0 usec)
                        Part of profile(s): headset_head_unit
```

现在使用以下命令即可

```sh
aplay xxx.wav
```