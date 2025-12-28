+++
date = '2025-09-13T17:53:48+08:00'
draft = false
title = 'CUDA学习——Softmax算子'
categories = ['Programming']
tags = ['GPU', 'Operator']
math = true
toc = true

+++
## 引言

在深度学习和机器学习领域，`Softmax`函数是一个至关重要的组成部分，在`Transformer`中{{<sidenote>}}将原始注意力分数转化为输入标记的概率分布。当然现在融合在FlashAttention中{{</sidenote>}}扮演着关键角色。正好我最近在学习`CUDA`编程，并尝试实现一个高效的`Softmax`算子。本文将分享我在实现过程中遇到的挑战、解决方案以及性能优化的经验。

<!--more-->
