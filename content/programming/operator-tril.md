+++
date = '2025-09-11T09:41:53+08:00'
draft = false
title = '那些年我写过的算子-Tril'
categories = ['Programming']
tags = ['Operator']
math = true
toc = true
+++
本系列文章记录了我从工作到现在经手的算子实现，分享一些经验和思考。`Tril`算子{{<sidenote>}}下三角矩阵，参见[torch.tril官方文档](https://docs.pytorch.org/docs/stable/generated/torch.tril.html){{</sidenote>}}是我在工作中实现的第一个算子，也比较简单，适合入门。
<!--more-->
## 算子定义
`Tril`算子是一个矩阵操作算子，输入一个二维矩阵(或者更高维的张量)和一个可选的对角线偏移量（`diagonal`），输出一个下三角矩阵(或者更高维张量的每个二维切片的下三角矩阵)。所谓下三角矩阵是指矩阵中位于对角线及其下方的元素保留，主对角线以上的元素置零。
![Tril算子实例](/images/tril.svg) {{<sidenote>}}图中m=8，n=10，空白处为0{{</sidenote>}}
接到这个算子时，正值公司在Ascend910B上部署`LLama`模型，`Tril`算子是其中的一个基础算子。这个算子在`LLama`模型中用于生成注意力掩码矩阵，确保每个位置只能关注其前面的位置信息。
## 算子GPU实现
拿到算子需求后，我们往往会参考已有的CUDA实现，这个算子在pytorch的`aten/src/ATen/native/cuda/TriangularOps.cu`中。{{<sidenote>}}该实现在2024年有过一次优化，曾有一度，我实现的性能比pytorch的还好{{</sidenote>}}
```cpp
IndexType running_index;
#pragma unroll
for (IndexType i = dims - 3; i >= 0; --i) {
    running_index = linear_idx % self_info.sizes[i];
    linear_idx /= self_info.sizes[i];
    self_offset += running_index * self_info.strides[i];
    result_offset += running_index * result_info.strides[i];
}
bool mask = upper ? (col - row >= k) : (col - row <= k);
result_info.data[result_offset] = mask ? self_info.data[self_offset] : scalar_t(0);
```
上面这段代码是`Tril`算子的核心逻辑，主要思路是通过计算每个线程处理的元素在输入张量中的位置，然后根据行列索引和对角线偏移量来决定是否保留该元素。对于`SIMT`架构点GPU来说，这种逐元素处理的方式非常适合并行计算，但是到了`SIMD`点NPU上，这种方式会带来严重的`scalar`计算，导致性能大幅下降。
## 算子NPU实现
NPU的`SIMD`架构要求我们尽量使用向量化操作，减少分支和标量计算。针对`Tril`算子，我们首先会想到退化的场景。
### 退化场景
当对角线在矩阵上方时，整个矩阵都在下三角区域内，输出矩阵等于输入矩阵，无需任何计算；当对角线在矩阵下方时，整个矩阵都在上三角区域内，输出矩阵全为零，也无需任何计算。
![Tril算子退化场景](/images/tril-degenerate.svg)
也就是$$y_{ij}=\begin{cases}0 &\text{if } k \le -m \\\\ x_{ij} &\text{if } k \ge n-1 \end{cases}$$
### 小矩阵场景
当矩阵元素个数比较小时{{<sidenote>}}这种场景很少见，优化优先级不高{{</sidenote>}}，