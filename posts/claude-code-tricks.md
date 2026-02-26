---
title: Vibe Coding 使用经验
mathjax: true
toc: true
categories:
  - Vibe Coding
date: 2026-02-25 16:52:30
tags:
-   Claude Code
-   Copilot
-   踩坑经验
---

`Vibe Coding`绝对是未来必不可少的工具。所以现在还是应该花时间去掌握工具，所以最近在使用`Claude Code`以及`Copilot`过程中，做一些总结。

<!--more-->

# `Claude Code`通知提醒

我比较喜欢在vscode中使用`Claude Code`，然后让多个session自己运行，我可以节省时间干别的事情，但是如何在需要交互的时候提醒我呢？这里还是需要一番配置。


1. 首先安装`Terminal Notification`插件，然后开启vscode的OSC通知功能，在`settings.json`中添加：

```json
{
  "terminal.integrated.enableVisualBell": true,
  "terminal.integrated.focusAfterRun": "terminal",
}
```

2. 添加一个监听脚本到`~/notify_osc_listener.sh`

```bash
#!/bin/bash
# 在普通 VSCode 终端中运行此脚本，接收 Claude Code 的通知并输出 OSC 777
PIPE="/tmp/claude-notify-pipe"

# 创建命名管道
rm -f "$PIPE"
mkfifo "$PIPE"

cleanup() {
    rm -f "$PIPE"
    echo "监听已停止"
    exit 0
}
trap cleanup EXIT INT TERM

echo "正在监听 Claude Code 通知... (Ctrl+C 停止)"

while true; do
    if IFS='|' read -r event message < "$PIPE"; then
        printf "\e]777;notify;%s;%s\a" "$event" "$message"
    fi
done
```

并执行他

```bash
bash ~/notify_osc_listener.sh
```


3. 添加一个hook脚本到`~/.claude/hooks/notify_osc.sh`

```bash
#!/bin/bash
# Claude Code hook: 将通知写入命名管道，由监听脚本输出 OSC 777
EVENT="$1"
MESSAGE="$2"
PIPE="/tmp/claude-notify-pipe"

if [ -p "$PIPE" ]; then
    # 用 timeout 避免没有监听者时阻塞
    timeout 1 bash -c "echo '${EVENT}|${MESSAGE}' > '${PIPE}'" 2>/dev/null &
fi
```

4. 在`~/.claude/settings.json` 设置hooks 

```json
{
"hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/hooks/notify_osc.sh 'Stop' 'Claude 响应完成，等待输入'"
          }
        ]
      }
    ],
  }
}
```

5. 再使用`Claude Code`，当他完成响应时，你就会收到vscode的通知了。