---
title: vscode Python Docstring Generator
mathjax: true
toc: true
categories:
  - 工具使用
date: 2020-05-03 15:01:16
tags:
- Vscode
- Python
---

`Python Docstring Generator`这个插件我用了挺久了,不过他这里的提供的样式风格和`vscode`自动提示的风格是不太匹配的.之前改了一个模板我又找不到了,这里记录一下.


<!--more-->

主要还是`vscode`渲染注释的时候会把下划线转义,而且不会自动识别换行,说明他应该不是`markdown`格式的,所以用下面这个模板替换`~/.vscode/extensions/njpwerner.autodocstring-0.5.1/out/docstring/templates/google.mustache`,之后自动注释看起来就会清爽许多.

```
{{! Google Docstring Template }}
{{summaryPlaceholder}}

{{extendedSummaryPlaceholder}}

{{#parametersExist}}
Args:
{{#args}}
    `{{var}}` ({{typePlaceholder}}): {{descriptionPlaceholder}}
    
{{/args}}
{{#kwargs}}
    `{{var}}` ({{typePlaceholder}}, optional): {{descriptionPlaceholder}}. Defaults to {{&default}}.
    
{{/kwargs}}
{{/parametersExist}}

{{#exceptionsExist}}
Raises:
{{#exceptions}}
    {{type}}: {{descriptionPlaceholder}}
{{/exceptions}}
{{/exceptionsExist}}

{{#returnsExist}}
Returns:
{{#returns}}
    {{typePlaceholder}}: {{descriptionPlaceholder}}
{{/returns}}
{{/returnsExist}}

{{#yieldsExist}}
Yields:
{{#yields}}
    {{typePlaceholder}}: {{descriptionPlaceholder}}
{{/yields}}
{{/yieldsExist}}
```