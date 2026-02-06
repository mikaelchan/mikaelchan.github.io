+++
date = '2026-02-07T13:03:30+08:00'
draft = false
title = 'CUDA学习——Transpose算子'
categories = ['Programming']
tags = ['GPU', 'Operator']
math = true
toc = true
+++
矩阵转置操作虽然在算法上非常基础——不需要任何浮点算术运算 (FLOPs)——但它代表了并行架构内存子系统的关键压力测试。 在GPGPU 的背景下，该操作的性能几乎完全取决于内存层次结构的效率，特别是流式多处理器 (SM) 加载/存储单元 (LSU)、片上共享内存 (SRAM) 和片外动态随机存取存储器 (DRAM) 之间的交互。

本文在A100(SXM4-80GB)上做了四组实验{{<sidenote>}}矩阵为4096x4096xfp32, 吞吐从218.67GB/s 提升至1796.98GB/s。{{</sidenote>}}：朴素全局内存方法、共享内存分块方法、利用 Padding（填充）优化的共享内存方法、利用 swizzling优化共享内存和对角线访问DRAM的方法，来学习CUDA上的优化技巧。
<!--more-->
## 理论框架和硬件架构
为了完全阐明观察到的性能差异背后的机制，必须对 GPU 的底层硬件架构建立严格的理解。转置算子的性能直接取决于软件访问模式如何映射到这些硬件单元的物理约束。
### 内存层次结构与延迟隐藏
GPU 计算的根本挑战在于“内存墙 (Memory Wall)”。虽然我是用的 A100 的计算吞吐量标注 19.5TFLOPS{{<sidenote>}}fp32 cuda core{{</sidenote>}}，但内存带宽仅在 2 TB/s。对于像矩阵转置这样的算子，它执行纯粹的数据排列$A_{ij}​→B_{ji}​$ 而没有数据重用或算术规约，其运算强度正好是每访问 1 字节传输 1 字节（假设完美缓存），甚至更差。
本文会涉及到3个不同层级的内存：
1. 全局内存 (DRAM): 最大但最慢的内存空间，延迟在 400–800 个时钟周期范围内。访问由 L2 缓存和内存控制器分区管理。

2. L2 缓存: 所有 SM 共享的统一缓存。它是几乎所有 DRAM 事务的一致性点和缓冲区。

3. L1 缓存 / 共享内存: 位于每个 SM 内部的快速片上 SRAM 块。L1 和共享内存通常是可配置或统一的。共享内存由程序员手动管理，提供低延迟（约 20–30 周期）和高带宽，前提是访问模式要避免冲突 。
### 加载/存储单元（LSU）与合并访存（Coalescing）
流式多处理器 (`SM`) 以 32 个线程为一组（称为 "Warps" / 线程束）执行指令。当一个 Warp 遇到内存指令（例如 `LDG` 或 `STG`）时，请求由加载/存储单元 (`LSU`) 处理。

LSU 的效率取决于 合并访问 (Coalescing)。SM 和 L2 缓存之间的内存接口以 32 字节的“扇区 (Sector)”为单位操作。当一个 Warp 的 32 个线程发出内存地址时，LSU 会分析这些地址的空间局部性。

+ 合并访问: 如果 Warp 发出的地址落在连续且对齐的区域内（例如，由 4 个 32 字节扇区组成的单个 128 字节缓存行），LSU 可以用最少数量的事务（通常是 4 个 32 字节事务）为整个 Warp 服务。{{<sidenote>}}这与锁步执行也有直接关系，但讨论这个还要考虑分支发散，为了省事略过。{{</sidenote>}}

+ 非合并访问: 如果地址是分散的（跨步的），LSU 必须为请求的每个不同扇区发出单独的事务。这种现象称为“扇区发散 (Sector Divergence)”，它会倍增内存总线上的流量并可能使 LSU 的内部队列饱和 。

### 加载/存储队列（LSQ）与停顿机制（stall）
LSU 包含有限数量的槽位用于跟踪挂起的内存请求，称为加载/存储队列 (LSQ) 或未命中状态保持寄存器 (MSHRs)。 当一个 Warp 发出导致许多单独事务（非合并）的内存指令时，它会消耗不成比例的大量 LSQ 条目。如果 LSQ 填满，Warp 调度器就无法从该 SM 上的任何 Warp 发出进一步的内存指令，直到挂起的请求完成。这造成了“结构性停顿 (Structural Stall)”或“内存节流”，有效地暂停了执行流水线并暴露了内存子系统的全部延迟 。

## 朴素写法
铺垫了这么多，是时候看一下代码了。
```c++
__global__ void transpose_naive(float *odata, const float *idata, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx_in  = y * width + x;
        int idx_out = x * height + y;
        odata[idx_out] = idata[idx_in];
    }
}
```
线程从全局内存读取并直接写入全局内存中的转置位置，没有中间缓冲，性能218.67GB/s。从读写和写入来看发生了什么：
### 读取：合并效率
在读取阶段，算子计算 `idx_in = y * width + x`。 Warp 内的线程（threadIdx.x 从 0 到 31 变化）拥有连续的 `x` 值:
+ 线程 0 读取 `idata[base + 0]`
+ 线程 1 读取 `idata[base + 1]`
+ ...
+ 线程 31 读取 `idata[base + 31]`

由于数据是 float（4 字节），Warp 请求的内存跨度是 32 线程×4 字节=128 字节。 这个 128 字节块是完美连续且对齐的。GPU 内存控制器通过 4 个事务（每个 32 字节）或单个 128 字节burst传输来服务此请求。读取带宽效率接近 100%。从 DRAM 取出的每个字节都被线程消耗，因此这一部分是高度优化的。

### 写入：非合并瓶颈
在写入阶段：`odata[idx_out] =...` 其中 `idx_out = x * height + y`。 这里，x 仍然随线程 ID（0 到 31）变化，但在输出矩阵中（因为我们正在转置），x 变成了行索引。连续线程之间的跨度是 height (4096)。
+ 线程 0 写入地址 A。
+ 线程 1 写入地址 A+4096 floats=A+16,384 字节。
+ 线程 2 写入地址 A+32,768 字节。

这种访问模式是病态的**非合并** `(Uncoalesced)`。地址跨度高达 16 KB。

### 事务膨胀与总线浪费
GPU 内存系统无法在没有开销的情况下向 DRAM 写入单个 4 字节字。内存事务的最小粒度是 32 字节扇区。 为了服务线程 0 的写入请求（4 字节），内存控制器必须处理一个完整的 32 字节扇区。
+ 数据传输: 传输 32 字节。
+ 有用数据: 4 字节。
+ 总线利用率: 4/32=12.5%。

因为 Warp 中的 32 个线程寻址 32 个不同的、非连续的扇区，LSU 被迫为单条 Warp 指令生成 **32 个独立的内存事务**。 这种事务的大规模膨胀解释了低带宽。虽然理论上的“有用”带宽是 218 GB/s，但硬件实际移动数据的速率可能接近 218×8=1744 GB/s。

### LSQ 饱和与流水线停顿

除了带宽浪费，非合并写入还在 SM 中造成了结构性冒险。LSQ 的深度是有限的（例如，每 SM 32-64 个条目）。一条合并的 Warp 指令通常消耗约 4 个条目。在这版实现中，一条 Warp 指令消耗 32 个条目。 这会瞬间填满 LSQ。Warp 调度器尝试发出下一条指令，但因为队列已满而收到来自 LSU 的“未就绪”信号。这迫使调度器停顿该 Warp，并且很可能停顿该 SM 上所有其他等待内存资源的 Warp。 这种 LSQ 停顿 阻止了 GPU 隐藏内存延迟。SM 处于空闲状态，等待低效事务积压被排空到 L2/DRAM。这种延迟暴露是性能低下的主要硬件原因 。

## 分块共享内存写法
为了让写入事务合并，需要在算子内做转置，将读取和写入解耦，而解耦意味着引入中间量——shared memory。这里我们将矩阵分块（tiling），分块的大小由warp线程大小决定：32*32。
```c++
__global__ void transpose_shared(...) {
    __shared__ float tile[32][32]; // 32x32

    // 1. 从全局内存合并读取 -> 存入共享内存 (行主序)
    int idx_in = y * width + x;
    tile[threadIdx.y][threadIdx.x] = idata[idx_in];
    __syncthreads();

    // 2. 从共享内存读取 (列主序) -> 合并写入全局内存
    x = blockIdx.y * TILE_DIM + threadIdx.x; // 转置后的块索引
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    int idx_out = y * height + x;
    odata[idx_out] = tile[threadIdx.x][threadIdx.y];
}
```
性能：626.93GB/s。
### 解耦
这里的根本优化是确保 **所有对全局内存的访问都是合并的**。
1. 读取阶段: 线程从 idata 逐行读取 32×32 的块。具有连续 ID 的线程读取连续地址。**合并访问**。
2. 缓冲: 数据存储在片上共享内存中。
3. 写入阶段: 线程将块逐行写入 odata（在目标的坐标系中）。因为我们交换了块索引（blockIdx.y 映射到 x），线程写入全局内存中的连续地址。**合并访问**。

通过确保全局读取和写入都是合并的，我们实现了 100% 的总线利用率。然而，瓶颈从全局内存转移到了**共享内存**。
### 共享内存Bank
共享内存并非单一的内存块，而是被划分为 32 个大小相等的内存模块，称为 **Bank (存储体)** 。
+ Bank 宽度: 4 字节 (32 位)。
+ 映射: 连续的 32 位字映射到连续的 Bank。
    - 地址 0 → Bank 0
    - 地址 4 → Bank 1
    - ...
    - 地址 124 → Bank 31
    - 地址 128 → Bank 0

**带宽规则**: 如果 32 个同时访问的目标是 32 个不同的 Bank（或者如果所有访问都针对完全相同的地址，即广播），共享内存硬件可以在每个周期服务 32 个访问。如果 Warp 中的多个线程请求的地址映射到 同一个 Bank（且地址不同），就会发生 **Bank 冲突** (Bank Conflict) 。

### Bank冲突
在写入阶段，线程从共享内存读取以写入全局内存： `val = tile[threadIdx.x][threadIdx.y]`;

这里，Warp 是由 threadIdx.x 从 0 到 31 变化的线程组成的，而 threadIdx.y 是常数（假设对于循环的第一次迭代或第一个 Warp，threadIdx.y = 0）。
+ 线程 0 读取 tile。
+ 线程 1 读取 tile[1]。
+ 线程 2 读取 tile[2]。
数组 `tile` 声明为 float tile。这意味着行 $i$ 和行 $i+1$ 之间的跨度正好是 32 个 float（128 字节）。

让我们计算每个线程请求的 Bank ID： $tile[i]$ 的线性索引是 $i \times 32$。
$$ Bank ID=(i\times 32)(mod32)=0 $$

对于 Warp 中的 每一个 线程 $i$ (0 到 31)，请求的地址都映射到 **Bank 0**。

+ 线程 0 请求 Bank 0 (地址 0)。
+ 线程 1 请求 Bank 0 (地址 32)。
+ 线程 2 请求 Bank 0 (地址 64)。

这是一个 32 路 Bank 冲突，共享内存性能的最坏情况。

### 硬件串行化开销
当硬件检测到 Bank 冲突时，它无法并行服务这些请求。内存控制器将 Warp 的请求拆分为串行的“波前 (Wavefronts)”或“重播 (Replays)”。 对于 32 路冲突，硬件必须将请求串行化为 32 个单独的事务。
+ 周期 1: 服务线程 0。
+ 周期 2: 服务线程 1。
+ ...
+ 周期 32: 服务线程 31。

实际上，共享内存的带宽降低到了其峰值能力的 1/32。虽然共享内存天生非常快，但将其吞吐量降低 97% 使其成为了显著的瓶颈。

## Padding 优化共享内存写法
引入一个简单的结构更改来消除bank冲突：
```c++
__global__ void transpose_shared_padding(...) {
    __shared__ float tile[32][33]; // Padding
    //... 逻辑保持不变...
}
```
性能1178.81GB/s。
### Padding逻辑
通过声明`tile[32][33]`，我们在共享内存布局中插入了一个“哑”列。逻辑 Tile 仍然是 32×32，但物理内存中行之间的跨度从 32 个 float 变为 33 个 float。

让我们重新检查导致上面bank冲突的访问模式 tile[threadIdx.x]（列访问）。 tile[i] 的线性索引现在是 $i\times 33$。

计算 Bank 映射：
$$Bank ID=(i×33)(mod32)=i(mod32)$$
+ 线程 0 (i=0) 访问 Bank 0。
+ 线程 1 (i=1) 访问 Bank 1。
+ 线程 2 (i=2) 访问 Bank 2。
+ ...
+ 线程 31 (i=31) 访问 Bank 31。

### 无冲突执行

由于跨度为 33，访问完美地错开了 Bank。Warp 中的每个线程都针对一个 **唯一的 Bank**。 共享内存硬件可以在 单个周期 内服务所有 32 个请求，从而创建了一个无冲突的内存事务，Replay 开销降至 0%。 共享内存不再是瓶颈。算子现在严格受限于全局内存带宽。

## XOR swizzling优化共享内存
虽然上面的Padding方式实现了无冲突事务，但浪费了一列空间。为此，我们引入Swizzling技术，这是一种通过改变逻辑索引与物理地址映射关系来避免冲突的技术，其核心在于“置乱”访问模式，使得逻辑上垂直的一列数据在物理上分布于不同的 Bank 中。

### swizzling 映射
依然我们有一个 $32 \times 32$ 的共享内存 Tile。
+ 逻辑坐标：$(row, col)$
+ 线性索引：$idx = row \times 32 + col$
+ Bank 映射（无 Swizzle）：$Bank = col \pmod{32}$。显然，固定 $col$ 变化 $row$ 时，Bank 不变，导致冲突。

引入 XOR Swizzling 后，我们重新定义物理地址的映射规则。常见的 Swizzle 模式是：$$col_{swizzled} = col \oplus row$$
或者更准确地应用在 1D 索引上：$$\text{Physical Address} = (row \times 32) + (col \oplus row)$$

冲突分析：当我们按列访问（即转置操作）时，我们固定 $col$，遍历 $row$：
+ $row = 0 \implies \text{Addr} = 0 + (col \oplus 0) \implies Bank = col$
+ $row = 1 \implies \text{Addr} = 32 + (col \oplus 1) \implies Bank = col \oplus 1$
+ $row = 2 \implies \text{Addr} = 64 + (col \oplus 2) \implies Bank = col \oplus 2$
+ ...

可以看到，随着 $row$ 的变化，$(col \oplus row)$ 的结果也在不断变化（遍历 $0 \dots 31$ 的排列）。因此，逻辑上的同一列数据被均匀地“散射”到了所有 32 个 Bank 中。

## 对角线重排列

尽管上面的性能已经非常好了，但矩阵转置可能会受到一种称为 Partition Camping (分区拥塞) 现象的影响，特别是在具有 2 的幂次维度的矩阵上（我们的case是4096×4096）。
### Partition Camping 现象
GPU 全局内存被划分为多个分区（通常为 6、8 或 12 个，取决于总线宽度），每个分区由一个内存控制器控制。物理地址通过涉及地址低位的哈希函数映射到分区。 对于宽度为 4096 的矩阵，行之间的跨度正好是 4096×4=16384 字节。 如果分区交错大小是（例如）256 字节，那么相隔大 2 的幂次的地址通常会映射到 同一个分区。

在标准转置中，块以行主序启动 (Block (0,0), (0,1), (0,2)...)。
+ 读取: 块读取行。这没问题。
+ 写入: 块写入列。
    - Block (0,0) 写入第 0 列。
    - Block (0,1) 写入第 32 列。
    - Block (0,2) 写入第 64 列。

如果哈希函数将这些列地址映射到同一个分区，多个线程块将竞争 同一个 DRAM 分区 (Camping)，而其他分区处于空闲状态。于是内存控制器级别串行化了 DRAM 访问，显著降低了有效带宽 。这就是为什么上面的性能只有0.58x的理论极限带宽。
### 对角线块重排列解决方案
为了防止这种情况，我们必须确保并发执行的块访问不同的分区。我们可以通过改变`blockIdx` 到矩阵 Tile 的映射来实现这一点。我们不按笛卡尔顺序处理 Tile，而是按**对角线顺序**处理。
```cpp
// 对角线块重排序
__global__ void transpose_diagonal(...) {
    // 坐标变换
    int blockIdx_y = blockIdx.x;
    int blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    // 使用 blockIdx_x 和 blockIdx_y 进行 Tile 计算
    int x = blockIdx_x * TILE_DIM + threadIdx.x;
    int y = blockIdx_y * TILE_DIM + threadIdx.y;
    //... 其余内核代码...
}
```
机制: 这种变换确保随着 blockIdx.x 的增加（GPU 调度器线性增加），内核扫描矩阵中对角相邻的 Tile。对角线 Tile 具有不同的行和列索引，从而将内存请求均匀地分散到 所有 DRAM 分区。在这个case上，吞吐为1796.98GB/s。

## 未来可期
因为我手上没有Hopper架构的芯片，无法做剩下的实验。在Hoopper上可以利用[TMA](https://research.colfax-intl.com/tutorial-hopper-tma/_，直接描述搬运的swizzling 和转置加载模式， 由专门的DMA引擎完成加载。