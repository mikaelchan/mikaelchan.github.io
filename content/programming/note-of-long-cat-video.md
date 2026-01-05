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
在正式深入DiT架构前，我们先看一下它是如何处理prompt，生成输入的{{<sidenote>}}工程开启了`DP`和`CP`。![dp-cp-diagram](/images/dp-cp-diagram.svg)
{{</sidenote>}}。
## 文本Prompt 编码
LongCat-Video 使用了T5 模型对文本Prompt 进行编码。核心的编码策略的代码如下：
```python
if context_parallel_util.get_cp_rank() == 0:
    # 只有主节点 (Rank 0) 执行文本编码
    (prompt_embeds, ...) = self.encode_prompt(...)
    
    if context_parallel_util.get_cp_size() > 1:
        # 主节点把结果广播 (Broadcast) 给所有其他节点
        context_parallel_util.cp_broadcast(prompt_embeds)
        # ... 广播 mask 和 negative prompt ...
        
elif context_parallel_util.get_cp_size() > 1:
    # 其他从节点 (Rank > 0)
    # 1. 先申请一块空显存 (buffer)，大小必须和 Rank 0 生成的一模一样
    prompt_embeds = torch.zeros(..., device=device)
    
    # 2. 接收广播数据 (填充 buffer)
    context_parallel_util.cp_broadcast(prompt_embeds)
    # ...
```
注意：这里是CP域下的计算逻辑，DP域下无须共享Prompt。为什么要先主节点计算再广播？因为文本编码器的计算量相对较小，但参数量较大、所以采用这种方式节省显存。
## Latent 噪声初始化
生成随机噪声的代码如下：
```python
# 生成随机噪声 (Gaussian Noise)
latents = torch.randn(shape, generator=generator, ...)

if context_parallel_util.get_cp_size() > 1:
    # 再次广播 Latents
    context_parallel_util.cp_broadcast(latents)
```
在 Context Parallel (CP) 中，视频的 $H \times W$ 被切分到了不同的 GPU 上。理论上，每个 GPU 只需要生成属于自己的那一小块噪声即可。但代码里选择生成完整的 Latents 然后广播，为什么？试想如果每个 GPU 独立生成，必须小心控制随机种子，确保边界处的噪声是连续的，否则拼起来的图像会有明显的接缝。而生成一个完整的 Latents（即使是高分辨率视频）的计算量在 GPU 上几乎可以忽略不计。直接在 Rank 0 生成完整的，然后广播给所有人，能保证所有卡拿到的初始噪声绝对一致，消除了随机性带来的 Bug。

当输入prompt仅为文本时，Latents 初始化如下：
```python
if latents is None:
    # 计算 Latent 形状
    num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
    shape = (...)
    # 生成高斯白噪声 N(0, I)
    latents = torch.randn(shape, ...)
```
当输入prompt包含图像时，Latents 初始化如下：
```python
if image is not None:
    # 1. 编码图片：把 RGB 图片通过 VAE Encoder 变成 Latent
    latent = retrieve_latents(self.vae.encode(encoded_input), gen)
    
    # 2. 归一化：将 Latent 分布拉回标准正态分布（这是 Diffusion 的假设前提）
    cond_latents = self.normalize_latents(cond_latents)
    
    # 3. 替换 (In-painting 逻辑)
    # 将生成的随机噪声的前几帧，强制替换为我们编码好的图片 Latent
    num_cond_latents = ...
    latents[:, :, :num_cond_latents] = cond_latents
```
这本质上是一个 In-painting（补全） 任务。模型拿到的输入是：[清晰的首帧 Latent] + [全是噪声的后续帧]。模型的任务是：保留首帧，根据首帧的内容，去噪还原后续的噪声。

## 视频预处理
```python
self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
self.t_embedder = TimestepEmbedder(t_embed_dim=adaln_tembed_dim,frequency_embedding_size=frequency_embedding_size)
self.y_embedder = CaptionEmbedder(
    in_channels=caption_channels,
    hidden_size=hidden_size,
)
```
输入处理由三个部分组成：
1. `x_embedder`：对视频的 Latent 进行 Patch Embedding。
2. `t_embedder`：对时间步长进行 Embedding。
3. `y_embedder`：对文本 Prompt 进行 Embedding。
