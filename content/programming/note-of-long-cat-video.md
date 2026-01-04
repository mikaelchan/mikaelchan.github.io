+++
date = '2026-01-04T16:18:00+08:00'
draft = false
title = 'LongCat-Video 笔记'
categories = ['Programming']
tags = ['LLM']
math = true
toc = true
+++
LongCat-Video 模型是美团开源的一款长上下文视频生成模型。本文为该模型代码的阅读笔记。
<!--more-->
[LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) 的架构如下：![LongCat-Video Architecture](/images/long-cat-video.png)
在正式深入DiT架构前，我们先看一下它是如何处理prompt，生成输入的{{<sidenote>}}工程开启了`DP`和`CP`。
```mermaid
graph TD
    subgraph Node_1[服务器 1 NVLink域]
        direction TB
        G0[GPU 0] --- G1[GPU 1] --- G2[...] --- G7[GPU 7]
        Note1[这8张卡组成 CP Group A<br>负责处理: Batch 1 的完整画面]
    end

    subgraph Node_2[服务器 2 NVLink域]
        direction TB
        G8[GPU 8] --- G9[GPU 9] --- G10[...] --- G15[GPU 15]
        Note2[这8张卡组成 CP Group B<br>负责处理: Batch 2 的完整画面]
    end

    Node_1 <== InfiniBand (Gradient Sync) ==> Node_2
    
    style Node_1 fill:#e1f5fe,stroke:#01579b
    style Node_2 fill:#e1f5fe,stroke:#01579b
    style Note1 fill:#fff,stroke:none
    style Note2 fill:#fff,stroke:none
```
{{</sidenote>}}。
## 文本Prompt 编码
