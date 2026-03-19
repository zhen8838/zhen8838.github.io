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

2. 添加一个脚本到`~/.claude/hooks/ghostty-notify.sh`

```bash
#!/bin/bash
# Sends a bell + OSC 777 notification to the parent Ghostty terminal.
# Walks up the process tree to find the TTY since hooks have no controlling terminal.

MSG="${1:-Notification}"

# Walk up process tree to find an ancestor with a real TTY
PID=$$
TTY=""
while [ "$PID" != "1" ] && [ -n "$PID" ]; do
  T=$(ps -o tty= -p "$PID" 2>/dev/null | tr -d ' ')
  if [ -n "$T" ] && [ "$T" != "??" ] && [ -e "/dev/$T" ]; then
    TTY="/dev/$T"
    break
  fi
  PID=$(ps -o ppid= -p "$PID" 2>/dev/null | tr -d ' ')
done

if [ -z "$TTY" ]; then
  exit 0
fi

# Send OSC 777 desktop notification + bell to the Ghostty terminal
printf '\033]777;notify;Claude Code;%s\007' "$MSG" > "$TTY"
printf '\a' > "$TTY"

exit 0
```

3. 在`~/.claude/settings.json` 设置hooks 

```json
{
"hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/hooks/ghostty-notify.**sh** 'Stop' 'Claude 响应完成，等待输入'"
          }
        ]
      }
    ],
  }
}
```

4. 再使用`Claude Code`，当他完成响应时，你就会收到vscode的通知了。

# 多任务分配监控

使用[vibekanban](https://www.vibekanban.com)，用看板的形式来管理多个`Claude Code`的session，非常简单。

# Agent Teams

最近也是使用多Agent来完成一些任务，借助了[slock.ai](https://slock.ai)平台，但是实践下来发现，虽然多Agent可以分工合作，但是在一些需要连续交互的任务中，会存在噪音(Chaos)，并且Agent的Context也会有耗尽的问题，所以想要长期稳定运转是一个比较大的挑战。

# RLCR工作流

目前两周尝试下来，最稳定的工作流还是RLCR (Ralph Loop with Code Review)，最简化的方案就是分Developer和Reviewer两个角色，给定一个spec明确的任务，Developer负责实现，Reviewer负责代码审查和反馈，直到满足需求为止。这个流程虽然简单，但是在实践中非常有效，可以保证代码质量和任务完成度。然后需要定期的复盘一些问题和经验，来不断打磨和优化这个流程。