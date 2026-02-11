---
title: hugging face llama使用
mathjax: true
toc: true
categories:
  - 深度学习
date: 2023-12-26 13:48:12
tags:
- 大语言模型 
- llama
- 踩坑经验
---

记录一下使用hugging face llama推理时遇到的问题.

<!--more-->

首先使用如下代码进行推理:

```python
from transformers import AutoConfig, AutoModel
from transformers import LlamaModel, LlamaConfig, LlamaTokenizerFast, LlamaForCausalLM
from transformers import AutoTokenizer
from torchsummary import summary
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger('transformers')
logger.setLevel(logging.DEBUG)

tokenizer = LlamaTokenizerFast.from_pretrained("/root/workspace/llama_test/llama-tokenizer")
prompt = "My name is Mariama, my favorite"
inputs = tokenizer(prompt, return_tensors="pt")
print(inputs)

config = AutoConfig.from_pretrained("/data/llama-65b-hf/") # 
config.torch_dtype = "float32"
config.use_cache = False
print(config)

model = LlamaForCausalLM.from_pretrained("/data/llama-65b-hf/", config=config)

print("model init!")
generate_ids = model.generate(inputs.input_ids, max_new_tokens=32)
print(generate_ids)
```

这里得到的输入为`tensor([[    1,  1619,  1024,   338,  1085,  2829, 29874, 29892,   590, 25448]])`, attention_mask为`tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])`

# generate过程

1. 加载模型的`generation.json`
    ```
    GenerationConfig {
      "bos_token_id": 0,
      "eos_token_id": 1,
      "max_new_tokens": 32,
      "pad_token_id": -1,
      "use_cache": false
    }
    ```

2. 配置最大长度

    `generation_config.max_length = generation_config.max_new_tokens + input_ids_length`目前是32+10 = 42

3. greedy_search
    
    根据输出策略, 进入`greedy_search`进行推理.

     1.  循环推理
     2.  准备输入
        ```
        'input_ids':
        tensor([[    1,  1619,  1024,   338,  1085,  2829, 29874, 29892,   590, 25448]])
        'position_ids':
        tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        'past_key_values':
        None
        'use_cache':
        False
        'attention_mask':
        tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        ```
     3. 进入forward
     4. decode layers.
        输出hidden_states为`torch.Size([1, 10, 8192])`
     5. 执行lm head, 得到logits为`torch.Size([1, 10, 32000])`
     6. 取logits最后一个`torch.Size([1, 32000])`求最大概率
        这里我得到的是`color`.
        而整个输出得到是`# name is Ktha and and I friends color`. 本来以为是输入`<s> xxx` 会得到`xxxy`这样, 然后取`y`作为一个输出. 现在看来其实前面部分也会被脑补一些.
