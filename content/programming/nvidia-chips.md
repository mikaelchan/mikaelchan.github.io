+++
date = '2025-09-13T16:11:33+08:00'
draft = false
title = 'Nvidia GPU速查'
categories = ['Programming']
tags = ["Chip", "GPU"]
+++
本文记录了一些Nvidia GPU的基本信息，方便查阅。
<!--more-->
## GPU型号
+ Volta SM70: 第一代Tensor Core，V100
+ Turing SM75: 第二代Tensor Core，RTX30
+ Ampere SM80: 第三代Tensor Core，A100
+ Hopper SM90: 第四代Tensor Core，H100
+ Blackwell SM100: 第五代Tensor Core，B200
## 主要参数对比

|参数|V100|A100|H100|
|----|----|----|----|
|计算能力（Compute Capability）|7.0|8.0|9.0|
|PTX版本|6+|7+|8+|
|CUDA版本|9-10|11+|12+|
|每个Warp的线程数（Threads / Warp）|32|32|32|
|每个SM的最大Warp数（Max Warps / SM）|64|64|64|
|每个SM的最大线程数（Max Threads / SM）|2048|2048|2048|
|每个SM的最大线程块数（Max Thread Blocks (CTAs) / SM）|32|32|32|
|每个线程块集群的最大线程块数（Max Thread Blocks / Thread Block Cl.）|NA|NA|16|
|每个SM的最大32位寄存器数（Max 32-bit Registers / SM）|65536|65536|65536|
|每个线程块的最大寄存器数（Max Registers / Thread Block (CTA)）|65536|65536|65536|
|每个线程的最大寄存器数（Max Registers / Thread）|255|255|255|
|每个线程块的最大线程数（Max Thread Block Size (# of threads)）|1024|1024|1024|
|SM寄存器与FP32核心的比例（Ratio of SM Registers to FP32 Cores）|1024|1024|512|
|每个SM的共享内存大小（Shared Memory Size / SM）|≤ 96 KB|≤ 164 KB|≤ 228 KB|
|张量核心代数（Tensor Core Generation）|1st|3rd|4th|

## 一些命令
```bash
# 查看GPU信息
nvidia-smi
# 编译到PTX
nvcc -arch=sm_90a -Xptxas -v -lineinfo -ptx -o xxx.ptx xxx.cu
# 编译到cubin
nvcc -arch=sm_90a -Xptxas -v -lineinfo -cubin -o xxx.cubin xxx.cu
# 查看cubin信息
cuobjdump -sass xxx.cubin
```