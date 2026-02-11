---
title: OrangePI蓝牙：开启蓝牙
date: 2018-05-29 16:53:42
tags: 
-   Linux
-   树莓派
-   蓝牙
categories: 
-   边缘计算 
---

我使用的板子是OrangePI zero Plus2，基于全志H5。我在上面安装好了armbian系统。我现在要使用蓝牙功能，对接我的蓝牙耳机。

<!--more-->


## 安装Bluez
linux上的蓝牙官方协议即为Bluez。所以需要先安装Bluez：
```sh
root@H5:~# apt-get install bluez bluez-tools
```
安装完成后可以观察到系统上多出了几个工具：
```sh
root@H5:~# hci
hciattach  hciconfig  hcitool 
```
使用工具：
```sh
root@H5:~# hciconfig 
root@H5:~# 
```
发现并没有蓝牙设备。

## 开启蓝牙设备

接下来我在网络上寻找解决方法。找到armbian论坛中的这个[帖子](https://forum.armbian.com/topic/5700-opz2-h5-onboard-bluetooth-support/?tab=comments#comment-53891)，其中thc013网友描述到：

enable overlay uart 1 and add param_uart1_rtscts=1 to  armbianenv.txt  and adjust /etc/init.d/ap6212-bluetooth so it looks like this
```sh
# Start patching
        rfkill unblock all  
        echo "0" > /sys/class/rfkill/rfkill0/state
        echo "1" > /sys/class/rfkill/rfkill0/state
        echo " " > /dev/$PORT
        devmem2 0x1f00060 b 1
        echo 10 > /sys/class/gpio/export
        echo out > /sys/class/gpio/gpio10/direction
        echo 0 > /sys/class/gpio/gpio10/value
        echo 1 > /sys/class/gpio/gpio10/value
        sleep 0.1
        hciattach /dev/$PORT bcm43xx 115200 flow bdaddr 
        $MAC_OPTIONS
        hciconfig hci0 up
```
and reboot
oh you might need to install devmem2

当然我要尝试使用这个方法了。

1.  **首先使能覆盖串口1**
    根据描述这个应该是在*armbianEnv.txt*中，我当即去寻找这个文件,这文件应该是在启动分区中的，所以我想将启动分区mount后去修改。
    ```sh
    root@H5:~# mount /dev/mmcblk2boot1 /mnt
    mount: /dev/mmcblk2boot1 is write-protected, mounting read-only
    mount: wrong fs type, bad option, bad superblock on /dev/mmcblk2boot1,
       missing codepage or helper program, or other error

       In some cases useful info is found in syslog - try
       dmesg | tail or so.
    ```
    然后我发现这个armbian系统和一般的系统不太一样。boot相关文件不需要挂载直接可以查看：
    ```sh
    root@H5:~# cd /boot/
    root@H5:/boot# ls
    armbianEnv.txt                  dtb-4.16.0-sunxi64
    armbian_first_run.txt.template  Image
    boot.bmp                        initrd.img-4.16.0-sunxi64
    boot.cmd                        System.map-4.16.0-sunxi64
    boot-desktop.png                uInitrd
    boot.scr                        uInitrd-4.16.0-sunxi64
    config-4.16.0-sunxi64           vmlinuz-4.16.0-sunxi64
    dtb
    root@H5:/boot# vi armbianEnv.txt
    ```
    在overlays后添加上了uart1，以及又添加了一行param_uart1_rtscts=1

2.  **调整/etc/init.d/ap6212-bluetooth**
    编辑这个文件
    ```sh
    root@H5:~# vi /etc/init.d/ap6212-bluetooth
    ```
    根据上面的帖子去修改这个文件。

3.  **安装devmem2**
    这是一个内核调试实用的工具，需要安装的话使用这个[链接](http://launchpadlibrarian.net/153545374/devmem2_0.0-0ubuntu1_arm64.deb)
    将软件包传入板子内安装：
    ```sh
    root@H5:~# dpkg -i devmem2_0.0-0ubuntu1_arm64.deb
    Selecting previously unselected package devmem2.
    (Reading database ... 31834 files and directories currently installed.)
    Preparing to unpack devmem2_0.0-0ubuntu1_arm64.deb ...
    Unpacking devmem2 (0.0-0ubuntu1) ...
    Setting up devmem2 (0.0-0ubuntu1) ...
    Processing triggers for man-db (2.7.6.1-2) ...
    ```
    安装完成重启测试蓝牙功能。

4.  **失败**
    我发现这个方法并不可行，我现在对这个方法有所了解了。他其实是使用这个方式开启蓝牙：
    ```sh
    root@H5:~# /etc/init.d/ap6212-bluetooth start
    [....] Starting ap6212-bluetooth (via systemctl): ap6212-bluetooth.serviceWarning: ap6212-bluetooth.service changed on disk. Run 'systemctl daemon-reload' to reload units.

    Job for ap6212-bluetooth.service failed because the control process exited with error code.
    See "systemctl status ap6212-bluetooth.service" and "journalctl -xe" for details.
    failed!

    root@H5:~# systemctl status ap6212-bluetooth.service
    ● ap6212-bluetooth.service - LSB: Patch firmware for ap6212 adapter
    Loaded: loaded (/etc/init.d/ap6212-bluetooth; generated; vendor preset: enabled)
    Active: failed (Result: exit-code) since Sun 2018-05-27 06:20:39 UTC; 2min 16s ago
        Docs: man:systemd-sysv-generator(8)
    Process: 2164 ExecStart=/etc/init.d/ap6212-bluetooth start (code=exited, status=1/FAILURE)

    May 27 06:20:29 H5 ap6212-bluetooth[2164]: Value at address 0x1F00060 (0xffff97239060): 0x1
    May 27 06:20:29 H5 ap6212-bluetooth[2164]: Written 0x1; readback 0x1
    May 27 06:20:29 H5 ap6212-bluetooth[2164]: sh: echo: I/O error
    May 27 06:20:39 H5 ap6212-bluetooth[2164]: Initialization timed out.
    May 27 06:20:39 H5 ap6212-bluetooth[2164]: bcm43xx_init
    May 27 06:20:39 H5 ap6212-bluetooth[2164]: Can't get device info: No such device
    May 27 06:20:39 H5 systemd[1]: ap6212-bluetooth.service: Control process exited, code=exited status=1
    May 27 06:20:39 H5 systemd[1]: Failed to start LSB: Patch firmware for ap6212 adapter.
    May 27 06:20:39 H5 systemd[1]: ap6212-bluetooth.service: Unit entered failed state.
    May 27 06:20:39 H5 systemd[1]: ap6212-bluetooth.service: Failed with result 'exit-code'.
    Warning: ap6212-bluetooth.service changed on disk. Run 'systemctl daemon-reload' to reload units.
    ```

# 尝试别的解决方法

我找了一天，刷了十几个镜像，这次使用Orange PI官方的镜像：ubuntu_xenial_zeroplus2_H5_V0_3.img。
我首先安装上了这个镜像，进入了root账户。

1.  **升级软件包**
    这里升级软件包时出现了一些错误，使用我这个命令将其修复。
    ```sh
    root@Orangepi:~# apt-get update
    Hit:1 http://ports.ubuntu.com xenial InRelease
    Hit:2 http://ports.ubuntu.com xenial-updates InRelease
    Hit:3 http://ports.ubuntu.com xenial-security InRelease
    Hit:4 http://ports.ubuntu.com xenial-backports InRelease
    Hit:5 http://ports.ubuntu.com/ubuntu-ports xenial-proposed InRelease
    Reading package lists... Error!
    E: Unable to parse package file /var/lib/dpkg/status (1)
    W: You may want to run apt-get update to correct these problems
    E: The package cache file is corrupted
    root@Orangepi:~# cp /var/lib/dpkg/status-old /var/lib/dpkg/status
    root@Orangepi:~# aptitude update
    Hit http://ports.ubuntu.com xenial InRelease
    Hit http://ports.ubuntu.com xenial-updates InRelease
    Hit http://ports.ubuntu.com xenial-security InRelease
    Hit http://ports.ubuntu.com xenial-backports InRelease
    Hit http://ports.ubuntu.com/ubuntu-ports xenial-proposed InRelease
    ```

2.  **安装bulez**
    ```sh
    root@Orangepi:~# apt-get install bluez
    Reading package lists... Done
    Building dependency tree
    Reading state information... Done
    The following NEW packages will be installed:
    bluez
    0 upgraded, 1 newly installed, 0 to remove and 393 not upgraded.
    Need to get 713 kB of archives.
    After this operation, 3902 kB of additional disk space will be used.
    Get:1 http://ports.ubuntu.com/ubuntu-ports xenial-proposed/main arm64 bluez arm64 5.37-0ubuntu5.2 [713 kB]
    Fetched 713 kB in 5s (121 kB/s)
    Selecting previously unselected package bluez.
    (Reading database ... 125410 files and directories currently installed.)
    Preparing to unpack .../bluez_5.37-0ubuntu5.2_arm64.deb ...
    Unpacking bluez (5.37-0ubuntu5.2) ...
    Processing triggers for dbus (1.10.6-1ubuntu3.1) ...
    Processing triggers for man-db (2.7.5-1) ...
    Processing triggers for systemd (229-4ubuntu12) ...
    Setting up bluez (5.37-0ubuntu5.2) ...
    Processing triggers for dbus (1.10.6-1ubuntu3.1) ...
    Processing triggers for systemd (229-4ubuntu12) ...
    ```
    安装好之后查看蓝牙设备，当然不出我所料是没有蓝牙设备的，因为我在这个文件系统的`/etc/init.d/`下没有看到和蓝牙启动相关的脚本。
    ```sh
    root@Orangepi:~# hciconfig hci0
    Can't get device info: No such device
    ```

3.  **启动蓝牙**
    我根据之前的启动蓝牙方法，尝试了如下命令：
    ```sh
    rfkill unblock all
    echo "0" > /sys/class/rfkill/rfkill0/state
    echo "1" > /sys/class/rfkill/rfkill0/state
    echo " " > /dev/ttyS1
    sleep 0.1
    hciattach /dev/ttyS1 bcm43xx 115200 flow bdaddr 43:29:B1:55:01:01
    ```
    执行结果(这里是前面的命令都执行了一遍)：
    ```sh
    root@Orangepi:~# hciattach /dev/ttyS1 bcm43xx 115200 flow bdaddr 43:29:B1:55:01:01
    bcm43xx_init
    Patch not found, continue anyway
    Set BDADDR UART: 43:29:B1:55:01:01
    Set Controller UART speed to 115200 bit/s
    Device setup complete
    root@Orangepi:~# hciconfig
    hci0:   Type: BR/EDR  Bus: UART
        BD Address: 43:29:B1:55:01:01  ACL MTU: 1021:8  SCO MTU: 64:1
        UP RUNNING PSCAN
        RX bytes:689 acl:0 sco:0 events:42 errors:0
        TX bytes:2730 acl:0 sco:0 commands:42 errors:0
    root@Orangepi:~# hcitool scan
    Scanning ...
        22:22:B8:95:DE:DC       X800+
    ```

#   总结

经过以上尝试，我觉得他们官方的镜像应该都可以。

1.  我觉得这个ubuntu xenial太过臃肿，我想用ubuntu server版，但是没有尝试，因为刷镜像太过烦恼。。。


2.  我之前的那个方法不行，应该是因为我使用的armbian基于4.16内核的镜像。但是我这个板子他不支持armbian 4.14内核的镜像，会无限重启。后来我下载了Armbian源代码，自行编译出Armbian_5.45_Orangepizeroplus2-h5_Debian_stretch_next_4.14.44.img这个镜像。这个也值得一试，毕竟armbian是定制的镜像，效率会高很多，发热量也很小。


3.  当然我觉得也可以在armbian上直接使用Orange Pi官方的wifi驱动，因为蓝牙不能用大概率是因为内核驱动的问题。

#   后记
我试了Armbian_5.45_Orangepizeroplus2-h5_Debian_stretch_next_4.14.44.img的镜像，然后进行了以下两步：

1.  **使能uart1**
    ```sh
    root@H5:/# vi /boot/armbianEnv.txt
    ```
    修改如下：
    ```git
    verbosity=1
    console=both
    overlay_prefix=sun50i-h5
    + overlays=usbhost2 usbhost3 uart1
    rootdev=UUID=73662019-22c8-4039-a605-680b57d0fb1c
    rootfstype=ext4
    + param_uart1_rtscts=1
    usbstoragequirks=0x2537:0x1066:u,0x2537:0x1068:u
    ```

2.  **修改蓝牙服务**
    ```sh
    root@H5:/# vi /etc/init.d/ap6212-bluetooth
    ```
    修改如下
    ```git
    # Start patching
    rfkill unblock all
    echo "0" > /sys/class/rfkill/rfkill0/state
    echo "1" > /sys/class/rfkill/rfkill0/state
    echo " " > /dev/$PORT
    +devmem2 0x1f00060 b 1
    #on orangepi win following command never ends on first try... force to run with a timeout...
    -timeout 5s echo " " > /dev/$PORT
    -if [ $? != 0 ]; then
    -        #timed out... retry
    -        echo " " > /dev/$PORT
    -fi
    hciattach /dev/$PORT bcm43xx 115200 flow bdaddr $MAC_OPTIONS
    hciconfig hci0 up
    ```

3.  **启动蓝牙**
    ```sh
    root@H5:/# /etc/init.d/ap6212-bluetooth start
    [....] Starting ap6212-bluetooth (via systemctl): ap6212-bluetooth.serviceWarning: ap6212-bluetooth.ser.
    . ok
    root@H5:/# hciconfig
    hci0:   Type: Primary  Bus: UART
            BD Address: 43:29:B1:55:01:01  ACL MTU: 1021:8  SCO MTU: 64:1
            DOWN
            RX bytes:696 acl:0 sco:0 events:42 errors:0
            TX bytes:2214 acl:0 sco:0 commands:42 errors:0
    ```
    启动成功