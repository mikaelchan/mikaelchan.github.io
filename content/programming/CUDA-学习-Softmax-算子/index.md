+++
date = '2025-02-07T13:03:30+08:00'
title = 'CUDA学习——Softmax算子'
tags = ['GPU', 'Operator']
series=["CUDA学习"]
series_order=1
+++
在深度学习中的核心地位在现代深度学习架构中，`Softmax` 函数扮演着至关重要的角色。作为将实数域输出映射到概率分布的关键组件，它广泛应用于多分类任务的输出层以及 `Transformer` 架构中的自注意力（`Self-Attention`）机制。虽然在计算复杂度上不及矩阵乘法，但在 Transformer 的注意力计算中，它频繁出现且涉及非线性的指数运算和全局归一化，往往成为内存带宽的瓶颈（Memory Bound）。
本文从数学定义的直接翻译（Naive Softmax）出发，剖析其数值溢出风险；进而引入 Safe Softmax 解决稳定性问题；通过分析内存带宽瓶颈，引入共享内存（Shared Memory）和 Warp 级原语（Warp Shuffle）进行优化；最终推导并实现 Online Softmax 算法，展示如何通过数学变换将多轮内存读写融合为单次扫描（Single Pass），从而显著降低显存带宽压力。
<!--more-->
{{<katex>}}
## 1. 数学原理与数值稳定性分析
### 1.1 Softmax 的定义与物理意义
`Softmax` 函数将一个 \(N\) 维的实数向量 \(\mathbf{x} = [x_1, x_2, \dots, x_N]\) 映射为一个 \(N\) 维的概率分布向量 \(\mathbf{y} = [y_1, y_2, \dots, y_N]\)。其标准数学定义如下：$$y_i = \text{Softmax}(\mathbf{x})_i = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}, \quad \text{for } i = 1, \dots, N$$
该变换具有两个关键性质：
1. 非负性：\(y_i > 0\)，因为指数函数 \(e^x\) 恒为正。
2. 归一化：\(\sum_{i=1}^N y_i = 1\)，这使得输出可以被解释为概率分布。

在 Transformer 模型中，输入向量 \(\mathbf{x}\) 通常是查询向量（Query）与键向量（Key）的点积结果（Score），Softmax 决定了模型对上下文中不同 Token 的关注程度（Attention Weight）。

### 1.2 浮点数表示与数值溢出
对于深度学习常用的单精度浮点数,指数函数 \(e^x\)的增长速度极快。
> [!INFO] 单精度浮点数（FP32,IEEE 754 标准），其指数位为 8 位，尾数位为 23 位。FP32 能表示的最大正有限数（MAX_FLOAT）约为 \(3.4028 \times 10^{38}\)

当 \(x\) 增大时，\(e^x\) 迅速逼近 FP32 的上限。
具体而言：$$e^{88} \approx 1.65 \times 10^{38}$$$$e^{89} \approx 4.49 \times 10^{38} > \text{MAX\_FLOAT}$$
这意味着，只要输入向量 \(\mathbf{x}\) 中存在任何一个元素 \(x_i \ge 89\)，计算 \(e^{x_i}\) 就会导致**上溢（Overflow）**，结果变为正无穷大（`inf`）。在随后的归一化步骤中，分母也会变成 `inf`。如果分子也是 `inf`，则会出现 `inf / inf`，结果变为 `NaN`。`NaN` 具有传染性，会破坏整个神经网络的后续计算，导致训练发散或推理错误。

### 1.3 Safe Softmax 的推导
为了解决上溢问题，业界通用的做法是利用 Softmax 的**平移不变性**。观察 Softmax 公式，对于任意标量 \(M\)，有：$$\frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}} = \frac{e^{x_i} \cdot e^{-M}}{\sum_{j=1}^{N} e^{x_j} \cdot e^{-M}} = \frac{e^{x_i - M}}{\sum_{j=1}^{N} e^{x_j - M}}$$
为了保证指数函数的指数部分非正，通常取输入向量的最大值作为 \(M\)，即 \(M = \max(\mathbf{x})\)。令 \(x'_i = x_i - \max(\mathbf{x})\)，则对于所有 $i$，有 \(x'_i \le 0\)。因此，\(e^{x'_i} \in (0, 1]\)。通过这种变换，我们彻底消除了上溢的风险。虽然可能出现**下溢（Underflow）**，即当 \(x_i\) 远小于 \(M\) 时，\(e^{x_i - M}\) 接近 0，但这在 Softmax 中是可以接受的（对应概率极小），不会导致 `NaN` 错误。这种数值稳定的形式被称为 **Safe Softmax**。然而，Safe Softmax 引入了新的计算依赖：
1. Pass 1: 遍历 \(\mathbf{x}\)，找到最大值 \(M\)。
2. Pass 2: 遍历 \(\mathbf{x}\)，计算 \(S = \sum e^{x_i - M}\)。
3. Pass 3: 遍历 \(\mathbf{x}\)，计算最终输出 \(y_i = e^{x_i - M} / S\)。

这种三趟扫描（3-Pass）的模式大大增加了对显存的访问次数，成为后续优化的重点。

## 2. CUDA 硬件架构

在深入代码之前，我们必须建立对 NVIDIA GPU 硬件架构的深刻理解。Softmax 的优化本质上是对 **GPU 存储层级和线程调度机制的适配**。
### 2.1 线程层次结构与执行模型
CUDA 的执行模型将线程组织为三级结构：
+ Grid（网格）：对应整个 Kernel 的启动。
+ Block（线程块）：被调度到流多处理器（Streaming Multiprocessor, SM）上执行的基本单元。
+ Warp（线程束）：硬件执行的最小单元，通常包含 32 个线程。Warp 内的线程遵循单指令多线程（SIMT）模式，即同时执行相同的指令但处理不同的数据。
**
Warp Divergence（分支发散）：**如果 Warp 内的线程在 `if-else` 分支中进入不同的路径，硬件必须串行执行这些路径，导致性能下降。在 Softmax 优化中，我们需要尽量避免 Warp 内的分支发散。

### 2.2 存储层级与访存模式
Softmax 是典型的 Memory Bound（访存受限） 算子。其计算密度（Compute Intensity，即每字节数据进行的浮点运算次数）较低。因此，**最大化有效显存带宽利用率**是优化的关键。
|存储类型|作用域|延迟 (时钟周期)|大小|特性|
|---|---|---|---|---|
|Global Memory|Grid|~400-800|显存容量 (GB级)|带宽最高，但延迟最大。需合并访问 (Coalescing)。|
|L2 Cache|Grid|~200|~40-100 MB|芯片级缓存，缓解 Global Memory 压力。|
|Shared|MemoryBlock|~20-50|~48-164 KB/SM|用户可编程的高速缓存 (L1)，用于 Block 内通信。|
|Registers|Thread|~1|~64K/SM|速度最快，每个线程私有。数量有限。|

**合并访问（Memory Coalescing）：** 当 Warp 内的 32 个线程访问 Global Memory 时，如果它们访问的地址是连续且对齐的（例如线程 \(k\) 访问地址 \(A + k \times 4\)），硬件可以将这 32 个请求合并为极少数（通常是 1 到 4 个）内存事务（Transaction）。如果访问杂乱无章，将导致大量细粒度的内存事务，严重浪费带宽。

### 2.3 并行规约（Reduction）模式
Softmax 计算涉及两个关键的规约操作：寻找最大值（Max Reduction）和求和（Sum Reduction）。在 GPU 上，高效的规约通常采用树状结构（Tree Reduction），时间复杂度为 \(O(\log N)\)。


## 3. 朴素 Softmax
我们首先考察最直观的实现。假设输入矩阵大小为 \(R \times C\)，我们需要对每一行（Row）进行 Softmax 处理。为了简化讨论，假设 \(C\) 较小（例如 \(C \le 1024\)），适合一个 Thread Block 处理一行。

### 3.1 算法流程
我们先直接翻译数学公式，不考虑 Safe Softmax 的数值稳定性，逻辑如下：
1. 每个 Block 处理一行。
2. Block 内的线程合作计算该行的指数之和 \(\sum e^{x_j}\)。
3. 每个线程计算 \(e^{x_i}\) 并除以和。

### 4.2 代码实现分析
```C++
__global__ void softmax_naive_kernel(float* input, float* output, int R, int C) {
    // 每一个 Block 处理输入矩阵的一行
    int row_idx = blockIdx.x;
    if (row_idx >= R) return;

    // 当前线程在行内的索引
    int tid = threadIdx.x;
    
    // 指向当前行数据的指针
    float* input_row = input + row_idx * C;
    float* output_row = output + row_idx * C;

    // 步骤 1: 计算指数之和 (Sum Reduction)
    float sum_exp = 0.0f;
    if (tid == 0) {
        for (int i = 0; i < C; ++i) {
            sum_exp += expf(input_row[i]);
        }
    }

    // 线程同步，确保 tid=0 算完了 sum_exp
    // 需要使用 Shared Memory 来广播 sum
    __shared__ float s_sum_exp;
    if (tid == 0) {
        s_sum_exp = sum_exp;
    }
    __syncthreads();

    // 步骤 2: 计算输出
    // 假设 C <= blockDim.x，每个线程处理一个元素
    if (tid < C) {
        output_row[tid] = expf(input_row[tid]) / s_sum_exp;
    }
}
```

上述代码存在三个致命缺陷，使其在实际应用中完全不可用：
1. **数值不稳定性：**直接计算 `expf(input_row[i])` 极易导致上溢，产生 `NaN`。
2. **极度低效的规约：** 代码中仅由 `tid=0` 的线程串行计算 `Sum`，其余线程空闲。这违背了 GPU 并行计算的初衷，性能会随 \(C\) 的增加线性下降。
3. **访存模式不佳：** 在 `tid=0` 的循环中，该线程跳跃式读取内存，无法利用 Warp 级的内存合并。
为了解决这些问题，我们必须引入 Safe Softmax 逻辑，并使用 **并行规约** 算法。

## 4. Safe Softmax 与并行规约(Shared Memory)
为了构建工业级可用的 Softmax，我们引入 Safe Softmax 逻辑（减去最大值），并利用 Shared Memory 进行树状规约。

### 4.1 算法重构
新的流程变为三个逻辑步骤（3-Pass）：
1. **Find Max:** 并行规约找出当前行的最大值 \(M\)。
2. **Compute Sum:** 计算 （\sum e^{x_i - M}\)。
3, **Normalize:** 计算 \(y_i = e^{x_i - M} / \sum\)。

### 4.2 树状规约 (Tree Reduction)
树状规约是 GPU 编程中的经典模式。其基本思想是：
1. 第一轮，线程 \(i\) 将数据与线程 \(i + \text{stride}\) 的数据合并；
2. 第二轮，`stride` 减半，直到 `stride` 为 \(0\)。

### 4.3 基于 Shared Memory 的实现
```c++
template<int BLOCK_SIZE>
__global__ void softmax_safe_v1_kernel(float* input, float* output, int R, int C) {
    int row_idx = blockIdx.x;
    if (row_idx >= R) return;
    int tid = threadIdx.x;
    
    // 加载数据：处理 C > BLOCK_SIZE 的情况（Grid-Stride Loop）
    // 为了简化，假设 C <= BLOCK_SIZE 且每个线程处理一个元素
    float val = (tid < C)? input[row_idx * C + tid] : -INFINITY;

    // --- Pass 1: 寻找最大值 ---
    __shared__ float s_data;
    s_data[tid] = val;
    __syncthreads();

    // 树状规约
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            // 比较并更新
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = s_data; // 全局最大值位于索引 0
    __syncthreads(); // 广播 max_val 必须的同步

    // --- Pass 2: 计算指数和 ---
    // 重新计算局部 exp 值
    // 注意：原始输入 val 仍在寄存器中，不需要重新读取 Global Memory
    float exp_val = (tid < C)? expf(val - max_val) : 0.0f;
    s_data[tid] = exp_val;
    __syncthreads();

    // 再次树状规约求和
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    float sum_exp = s_data;
    __syncthreads();

    // --- Pass 3: 归一化输出 ---
    if (tid < C) {
        output[row_idx * C + tid] = exp_val / sum_exp;
    }
}
```
尽管该版本解决了数值稳定性问题并引入了并行性，但在高性能计算专家的眼中，它仍有显著的优化空间：
1. Shared Memory Bank ConflictShared Memory 被划分为 32 个 Bank。如果一个 Warp 内的多个线程同时访问同一个 Bank 的不同地址，会发生 Bank Conflict，导致访问串行化。在树状规约的某些步长（stride）下，容易触发此问题。
2. 频繁的同步开销`__syncthreads()` 是一个 **Block** 级的屏障（`Barrier`）。当调用它时，SM 必须暂停该 Block 内所有 Warp 的执行，直到所有 Warp 到达该点。在上述代码中，我们在规约的每一层循环中都调用了 `__syncthreads()`，极大地破坏了流水线的并行度。
3. Warp Divergence随着 stride 的减小，参与计算的线程越来越少（`tid < stride`）。例如，当`stride=1` 时，只有 `tid=0` 在工作，其他 31 个线程（在一个 Warp 内）必须等待。这导致了严重的算力浪费。

## 5. Warp Shuffle 与寄存器优化
为了消除 Shared Memory 的开销和不必要的同步，我们可以利用`Warp Shuffle Instructions`。这些指令允许 Warp 内的线程直接交换寄存器数据，无需经过 Shared Memory，且无需 `__syncthreads()`（因为 Warp 内是隐式同步的）。

### 5.1 Warp Shuffle 原语
最常用的指令是 `__shfl_xor_sync(mask, var, laneMask)`，它允许线程与其 ID 进行异或运算后的目标线程交换数据。
```c++
__device__ inline float warpReduceMax(float val) {
    // 0xffffffff 表示参与通信的所有 32 个线程
    // 进行 5 轮蝴蝶交换（Butterfly Reduction），每轮 stride 减半
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}
```
> [!INFO] 其他warp shuffle 原语：
> 1. `T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize)` (直接广播/索引)
> 2. `T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize)` (向下平移， Tree Reduction)
> 3. `T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize)` (向上平移，prefix Sum)

### 5.2 Block Reduce：跨 Warp 通信
由于 Warp Shuffle 只能在 32 个线程内通信，对于大于 32 的 Block，我们需要一种机制来聚合各个 Warp 的结果。策略如下：
1. 每个 Warp 先执行 warpReduce。
2. 每个 Warp 的第一个线程（Lane 0）将结果写入 Shared Memory 的一个小缓冲区（大小为 `BLOCK_SIZE / 32`）。
3. Block 同步 ·__syncthreads()·。
4. 由第一个 Warp 读取 Shared Memory 中的结果，并再次执行 warpReduce。

这种方法将 Shared Memory 的使用量减少到了极小（仅几十个 float），且大部分计算在寄存器中全速运行。

## 6. Online Softmax —— 算法层面的突破
在上述所有 Safe Softmax 的实现中，我们都无法回避一个根本的结构性问题：多趟扫描（`Multi-Pass`）。我们必须遍历所有数据求出 \(M\)（Max），才能开始计算指数和 \(S\)。这意味着数据需要被读取或处理多次。虽然寄存器缓存可以减少 Global Memory 访问，但在处理长序列（\(C\) 很大）时，寄存器会溢出，导致必须多次读取 Global Memory。

Online Softmax 算法的出现打破了这一限制。它源于 [Welford 算法](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)在方差计算中的思想：我们可以一边遍历数据，一边动态更新统计量。

### 6.1 Online Softmax 的数学推导
我们的目标是在一次遍历中同时维护“当前的最大值”和“当前的指数和”。设处理到第 \(k\) 个元素时的局部最大值为 \(m_k\)，局部指数和为 \(d_k\)。定义：
$$
\begin{aligned}
m_k &= \max(x_1, \dots, x_k) \\\\
d_k &= \sum_{i=1}^{k} e^{x_i - m_k}
\end{aligned}
$$
当引入新元素 \(x_{k+1}\) 时，我们需要推导 \(m_{k+1}\) 和 \(d_{k+1}\) 的递推公式。
**更新最大值：**
$$m_{k+1} = \max(m_k, x_{k+1})$$
**更新指数和：**
$$
d_{k+1} = \sum_{i=1}^{k+1}e^{x_i - m_{k+1}} = e^{x_{k+1} - m_{k+1}} + \sum_{i=1}^{k} e^{x_i - m_{k+1}}
$$
用指数性质 \(e^{a-b} = e^{a-c} \cdot e^{c-b}\)，我们将求和项中的 \(m_{k+1}\) 替换为 \(m_k\)：$$
\sum_{i=1}^{k} e^{x_i - m_{k+1}} = \sum_{i=1}^{k} e^{x_i - m_k + m_k - m_{k+1}} = \left( \sum_{i=1}^{k} e^{x_i - m_k} \right) \cdot e^{m_k - m_{k+1}} = d_k \cdot e^{m_k - m_{k+1}}
$$
代入原式，得到核心递推公式：
$$
d_{k+1} = d_k \cdot e^{m_k - m_{k+1}} + e^{x_{k+1} - m_{k+1}}
$$
这个公式的物理意义非常优美：当发现一个新的最大值时（\(m_{k+1} > m_k\)），我们需要对之前累积的和 \(d_k\) 进行“衰减”（乘以 \(e^{m_k - m_{k+1}}\)，这是一个小于 1 的数），以适应新的基准；如果新值不是最大值（\(m_{k+1} = m_k\)），则衰减系数为 1，直接累加即可。
> [!INFO] Online Softmax 允许我们在流式读取数据的过程中，实时更新 `Max` 和 `Sum`。从而方便：
> 1. **算子融合：** `FlashAttention` 的核心技术。在计算 \(Q \times K^T\) 生成 Attention Score 时，Score 是在片上（On-Chip）计算出来的。使用 Online Softmax，我们可以直接处理这些 Score 并丢弃它们，而不需要将其写入 Global Memory 再读回来找 Max。
> 2. **降低带宽：** 对于标准 Softmax，虽然 Online 算法本身计算完 Stats 后仍需再次遍历数据生成 Output，但如果配合寄存器分块（Tiling），可以将输入数据保留在寄存器中，实现真正的 Single Pass（读一次输入，写一次输出）。

## 7. Online Softmax 的 CUDA 实现
我们来实现一个基于 Warp Shuffle 的 Online Softmax。这里的难点在于如何合并两个线程（或两个 Warp）持有的 (max, sum) 状态。

### 7.1 状态合并逻辑
假设线程 `A` 拥有 \((m_A, d_A)\)，线程 B 拥有 \((m_B, d_B)\)。合并后的状态 \((m_{AB}, d_{AB})\) 计算如下：
$$
\begin{aligned}
m_{AB} &= \max(m_A, m_B) \\
d_{AB} &= d_A \cdot e^{m_A - m_{AB}} + d_B \cdot e^{m_B - m_{AB}}
\end{aligned}
$$
这个逻辑与串行递推是同构的。
### 7.2 CUDA 代码实现
```c++
struct OnlineStats {
    float m; // Max
    float d; // Denominator
};
// 合并两个状态的设备函数
__device__ inline OnlineStats combine_stats(OnlineStats a, OnlineStats b) {
    OnlineStats out;
    out.m = fmaxf(a.m, b.m);
    // 避免计算 exp(负无穷)，增加鲁棒性
    float factor_a = (a.m == -INFINITY)? 0.0f : expf(a.m - out.m);
    float factor_b = (b.m == -INFINITY)? 0.0f : expf(b.m - out.m);
    out.d = a.d * factor_a + b.d * factor_b;
    return out;
}

// Warp 级在线规约
__device__ inline OnlineStats warpReduceOnline(OnlineStats val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        // 获取目标线程的 m 和 d
        float other_m = __shfl_xor_sync(0xffffffff, val.m, offset);
        float other_d = __shfl_xor_sync(0xffffffff, val.d, offset);
        
        OnlineStats other = {other_m, other_d};
        // 合并状态
        val = combine_stats(val, other);
    }
    return val;
}

__global__ void softmax_online_kernel(float* input, float* output, int R, int C) {
    int row_idx = blockIdx.x;
    if (row_idx >= R) return;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    // --- 步骤 1: 线程本地规约 (Thread-Local Reduction) ---
    // 每个线程处理多个元素 (Grid-Stride)
    OnlineStats local_stats = {-INFINITY, 0.0f};
    
    // 假设我们使用 float4 向量化加载来优化带宽（此处省略 float4 转换细节，展示逻辑）
    for (int i = tid; i < C; i += blockDim.x) {
        float val = input[row_idx * C + i];
        // 增量更新
        float new_m = fmaxf(local_stats.m, val);
        float factor = expf(local_stats.m - new_m);
        local_stats.d = local_stats.d * factor + expf(val - new_m);
        local_stats.m = new_m;
    }

    // --- 步骤 2: Warp 内部规约 ---
    local_stats = warpReduceOnline(local_stats);

    // --- 步骤 3: Block 内部规约 (跨 Warp) ---
    static __shared__ float s_m;
    static __shared__ float s_d;
    
    // 每个 Warp 的 Lane 0 将结果写入 Shared Memory
    if (lane == 0) {
        s_m[wid] = local_stats.m;
        s_d[wid] = local_stats.d;
    }
    __syncthreads();

    // 由第一个 Warp 读取并聚合
    OnlineStats block_stats = {-INFINITY, 0.0f};
    if (tid < 32) { 
         int num_warps = (blockDim.x + 31) / 32;
         if (tid < num_warps) {
             block_stats = {s_m[tid], s_d[tid]};
         }
         // 再次 Warp 规约得到全 Block 结果
         block_stats = warpReduceOnline(block_stats);
    }
    
    // 广播最终结果
    if (tid == 0) {
        s_m = block_stats.m;
        s_d = block_stats.d;
    }
    __syncthreads();
    
    float global_max = s_m;
    float global_sum = s_d;

    // --- 步骤 4: 计算输出并写回 ---
    // 此时 global_max 和 global_sum 是相对于整行的全局值
    for (int i = tid; i < C; i += blockDim.x) {
        float val = input[row_idx * C + i];
        output[row_idx * C + i] = expf(val - global_max) / global_sum;
    }
}
```
上面的代码实现了：
1. **数值稳定性保证：** 在 combine_stats 中，我们计算 `expf(a.m - out.m)`。由于 out.m 是最大值，`a.m - out.m` 必然 \(\le 0\)，保证了指数运算永远不会上溢，继承了 Safe Softmax 的优点。
2. **并行性：** 不同于 Safe Softmax 需要等待整个 Block 找到 Max 后才能计算 `Exp` `Sum`，Online Softmax 的每个线程、每个 Warp 都在独立地计算局部的 Max 和 Sum。这种“分而治之”的策略极大地提高了流水线效率。
3. **扩展性：** 该算法不仅适用于 Softmax，还是实现 LayerNorm、RMSNorm 等归一化算子的通用范式。

## 8. 其他优化：向量化与循环展开
除了算法逻辑，针对 GPU 硬件特性的微观优化同样决定成败。
### 8.1 向量化加载
NVIDIA GPU 的内存控制器通常以 32 字节为粒度进行传输。L2 Cache 行大小通常为 128 字节。如果我们使用 `float`（4 字节）逐个读取，会导致大量细粒度的内存指令，降低指令发射效率。使用 `LD.E.128` 指令（对应 CUDA C++ 中的 float4 类型）可以一次加载 16 字节（4 个 float）。
+ 指令数减少：指令数量减少为原来的 1/4，减轻了 Warp 调度器的压力。
+ 带宽利用率：更容易形成饱满的内存事务。
```c++
// 向量化读取
float4* in_vec = reinterpret_cast<float4*>(input_row);
// 此时 tid 需要调整， stride 变为 blockDim.x
float4 val_vec = in_vec[tid]; 
// 分别处理 val_vec.x, y, z, w
```
### 8.2 循环展开
结合向量化，我们可以使用 `#pragma unroll` 指令。他允许编译器将循环体复制多份，利用指令级并行（ILP）来掩盖内存读取的延迟。当一个 float4 在等待内存数据返回时，算术单元可以处理上一个 float4 的计算。

## 9. Roofline 模型分析
为了量化优化的效果，我们使用 Roofline 模型进行理论分析。Softmax 的计算主要是 Exp, Add, Div。
+ **FLOPs:** 每个元素约 3-5 次 FLOPs。
+ **Bytes:** 读 4 字节，写 4 字节。
+ **算术强度 (Arithmetic Intensity):** \(I \approx 5 / 8 = 0.625\) FLOPs/Byte。
A100 GPU 的理论 FP32 算力约为 19.5 TFLOPS，带宽 1555 GB/s。拐点约为 12.5 FLOPs/Byte。由于 \(0.625 \ll 12.5\)，Softmax 处于极端的 Memory Bound 区域。这意味着，任何减少 Global Memory 访问次数的优化（如 Online Softmax 结合寄存器缓存），都将带来线性的性能提升。
## 10.总结
从上面的实现中可以看到，最优的算法往往不是数学上运算量最小的，而是与硬件存储层级最匹配的。Online Softmax 虽然增加了计算量（每次都要乘衰减系数），但它换取了宝贵的显存带宽。其次，从 Block 级同步下沉到 Warp 级通信，是现代 CUDA 优化的核心趋势。最后，在 FP8 甚至 FP4 低精度推理时代，Softmax 的数值稳定性将面临新的挑战，Online Softmax 的动态调整机制将显得尤为重要。