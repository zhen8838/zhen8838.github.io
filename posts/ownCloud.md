---
title: ubuntu18.04安装owncloud
date: 2018-06-20 17:06:18
tags:
-   Linux
categories:
-   工具使用
---

最近实验室闲置了一台主机，就将这个主机搭建成一个公有云给大家使用。
系统版本是ubuntu server。

<!--more-->


#   安装Apache
首先安装apache以供owncloud使用：
```sh
sudo apt install apache2
```
需要禁用apache目录列表
```sh
sudo a2dismod autoindex
```
开启额外模块
```sh
sudo a2enmod rewrite
sudo a2enmod headers
sudo a2enmod env
sudo a2enmod dir
sudo a2enmod mime
```
重启apache
```sh
sudo systemctl restart apache2
```
# 安装MariaDB Server

```sh
sudo apt-get install mariadb-server mariadb-client
```
安装完成后添加密码
```sh
sudo mysql_secure_installation
```
执行后设置密码、移除匿名用户、不允许root登录、删除测试数据库。
接下来登录MariaDB并创建数据库
```sh
sudo mysql -u root -p
```
下面的命令是建立数据库，其中的用户名和密码需要替换成自己的。
```sh
CREATE DATABASE owncloud;
CREATE USER 'oc_user'@'localhost' IDENTIFIED BY 'PASSWORD';
GRANT ALL ON owncloud.* TO 'oc_user'@'localhost' IDENTIFIED BY 'PASSWORD' WITH GRANT OPTION; 
FLUSH PRIVILEGES;
EXIT;
```
# 安装php
现在owncloud只支持php7.1：
```sh
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:ondrej/php
sudo apt update
sudo apt install php7.1
```
接着安装php模块：
```sh
sudo apt-get install php7.1-cli php7.1-common php7.1-mbstring php7.1-gd php7.1-intl php7.1-xml php7.1-mysql php7.1-zip php7.1-curl php7.1-xmlrpc
```
安装完成后配置一下：
```sh
sudo vi /etc/php/7.1/apache2/php.ini
```
修改以下内容
```sh
file_uploads = On
allow_url_fopen = On
memory_limit = 256M
upload_max_file_size = 100M
```
重启apache
```sh
sudo systemctl restart apache2
```
# 下载owncloud
```sh
cd /tmp 
wget https://download.owncloud.org/community/owncloud-10.0.3.zip
```
解压并移动文件：
```su
unzip owncloud-10.0.3.zip
sudo mv owncloud /var/www/html/owncloud/
```
# 设置目录和权限
为了保证owncloud工作，需要设置操作权限：
```sh
sudo chown -R www-data:www-data /var/www/html/owncloud/
sudo chmod -R 755 /var/www/html/owncloud/
```
# 完成安装
需要登录到对应的网址(http://ipadress/owncloud)进行最后的设置：
输入你想要的账户名与密码，以及配置数据库的名字。
![](ownClound/finsh.png)

# 完成后的界面

![](ownClound/end.png)