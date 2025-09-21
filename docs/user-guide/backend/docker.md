---
sidebar_position: 2
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Docker 部署

:::info 关于安装方式的说明
目前有两种安装方式，如果您是小白，推荐您[使用Docker DeskTop](#使用docker-desktop)。如果您对Docker比较熟悉，也可以[使用终端命令](#使用终端命令)。
:::

:::note
我们在 docker hub 成立了[openllmvtuber团队](https://hub.docker.com/orgs/openllmvtuber/members)，目前docker镜像由[@Harry_Y](https://github.com/Harry-Yu-Shuhang)维护。如果docker镜像有问题，可以联系他(邮箱: yushuhang@163.com)。
:::

## 使用Docker Desktop

### 下载 Docker Desktop

- 访问 [Docker Desktop 下载页面](https://www.docker.com/products/docker-desktop)。
- 点击 "Download Docker Desktop"
![alt text](./docker_img/download_docker_desktop.png)
- 根据电脑类型选择对应版本

:::tip 如何选择适用于自己电脑的版本
- **Mac Apple Silicon** → 新款 Mac (M1/M2/M3 芯片)  
- **Mac Intel Chip** → 老款 Mac (Intel 处理器)  
- **Windows AMD64** → 大部分 Windows 电脑 (Intel/AMD 64 位)  
- **Windows ARM64** → 少见，仅 ARM 架构 Windows (如 Surface Pro X)  
:::

如果您还不确定，可以按如下方法查看系统设置：

<Tabs groupId="operating-systems">
  <TabItem value="windows" label="Windows">

1. 右键 **此电脑 → 属性**
2. 在 **系统类型** 一栏查看：  
   - “基于 x64 的处理器” → 选择 **Windows AMD64**  
   - “基于 ARM 的处理器” → 选择 **Windows ARM64**

  </TabItem>
  <TabItem value="macos" label="macOS">

1. 点击屏幕左上角苹果图标 → **关于本机**  
2. 在“芯片”一栏查看：  
   - 显示 **Apple M1/M2/M3** → 选择 **Mac Apple Silicon**  
   - 显示 **Intel** → 选择 **Mac Intel Chip**

  </TabItem>
</Tabs>

### 启动 Docker Desktop

- 双击下载好的安装包，按照提示完成安装。
- 安装完成后，点击 Docker Desktop 图标启动 Docker。
- 首次启动时，Docker Desktop 会要求您登录 Docker Hub 账号。如果您没有账号，需要先注册一个。

## 使用终端命令