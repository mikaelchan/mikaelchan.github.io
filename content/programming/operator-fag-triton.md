+++
date = '2026-02-09T09:54:49+08:00'
draft = false
title = 'Flash attention 反向在triton上的实现'
categories = ['Programming']
tags = ['operator', 'flash-attention']
math = true
toc = true
+++
最近在做Flash attention 反向{{<sidenote>}}吐槽下大家都叫fag，但是我一直很反感这个缩写。{{</sidenote>}}在Ascend triton上的实现， 受限与Triton的表达能力（内存层次和流水线），走了不少弯路。以下是实现0.94x AscendC实现的记录。
<!--more-->
### 1. 算法回顾
FA 反向公式如下：
$$
S = QK^\top \\\\
dP = dOV^\top \\\\
P = \exp(S \cdot scale-L) \\\\
\Delta = \sum{dO \cdot O} \\\\
dS = P\cdot (dP - \Delta) \\\\
dQ = dS K \cdot scale \\\\
dK = dS^\top Q \cdot scale \\\\
dV = P ^\top dO \cdot scale
$$
其中， Q，O, dO 的shape 为`[N, H, S1, D]`, K, V的shape为`[N, H, S2, D]`，$scale=1/\sqrt{D}$。
![反向实例](/images/flash-attention-grad.svg)
将输入分块后{{<sidenote>}}在sequence维度上{{</sidenote>}}，S1 单块BM, S2单块BN, 单次计算下各个小块shape如下：
+ **Q, O, dO**: BM * D
+ **K, V**: BN * D
+ **S, dP, P**, dS: BM * BN
+ **$\Delta$, L**: BM
+ **dQ**：BM*D
+ **dK, dV**： BN*D

在Ascend平台上，上图中的Step1, Step3运行在cube单元上，step2 运行在vector单元上。

### 2. Ascend 计算流
在开始分析计算流前，有必要看一下Ascend NPU硬件架构。
![NPU架构](/images/npu-arch.svg)
请注意：
1. L0C 计算好的matmul结果可以直接传输到UB上(`FIXPIPE`)
2. vector 计算好的结果可以直接传输到L1上(`MTE3`)
以上的两条通路减少了利用GM作为中转存储的需求，提高了计算效率，因此性能相较于从前比较线性，跟计算块数成正比。

单块的计算流如下：
![计算流](/images/flash-attention-grad-compute-flow.svg)
各部分标记如下：
+ C1: 计算S, 计算量： $BM\times D\times BN\times 2$, 2为Fp16
+ C2: 计算dP, 计算量： $BM\times D\times BN\times 2$
+ C3: 计算dV, 计算量： $BM\times D\times BN\times 2$
+ C4: 计算dQ, 计算量： $BM\times D\times BN\times 2$
+ C5: 计算dK, 计算量： $BM\times D\times BN\times 2$
+ V1: 计算$\Delta$, 计算量： $HALF\\_BM\times D\times 4 \times 6$, 2cast+乘法+规约加法（计算量大概3倍乘法）, 4为fp32
+ V2: 计算P，计算量: $HALF\\_BM \times BN \times 4 \times 4$, 乘法+加法+exp+castL
+ V3: 计算dS，计算量: $HALF\\_BM \times BN \times 4 \times 4$, 乘法+加法 
数据依赖如下：
+ C1，V1 -> V2 -> C3
+ C2, V2 -> V3 -> C4, C5

我们可以认为C1, C2, V1是生产者，V2, V3 是中间消费者，C3, C4, C5是最终消费者。
从计算单元上看，C1, C2, C3, C4, C5 依次在cube单元上计算，V1, V2, V3 依次在vector单元上计算。
Cube的算力和vector的算力比大概是4:1，并且1个cube核对应2个vector核。vector代码段执行会因为多指令执行有折损，实测下来当`D=128`时，C1和V2, V3的时间大致相当。
数据S在C1上产生，通过`FIXPIPE`传输到UB上，V2上消费后产生的P通过`MTE3`传输到L1上，C3上消费。dP在C2上产生，通过`FIXPIPE`传输到UB上，V3上消费后产生的dS通过`MTE3`传输到L1上，C4, C5上消费。

多个块的计算流如下：
![多块计算流](/images/flash-attention-grad-multi-block-compute-flow.svg)
### 3. CV并行掩盖
为了让多轮计算的cube和vector代码段并行执行，消除掉C2 和C3之间的气泡，我们让C1 和C2 提前计算一轮，记`C_{iter}_{op}`表示第iter轮的op操作，那么计算顺序如下：
1. 第1轮：计算C11, C12, V11
2. 第2轮：计算C21, C22, V21, V12, V13
3. 第3轮：计算C13, C14, C15, V22, V23, C31, C32, V31
4. 第4轮：计算C23, C24, C25, V32, V33, C41, C42, V41
5. 以此类推

![多块计算流CV并行掩盖](/images/flash-attention-grad-preload-compute-flow.svg)
此时，UB上同时存在两个块的S和dP, 内存增加一倍。
伪代码如下：
```python
# preload
q, k, do = load_block(0)
s, dp = cube_produce_block(q, k, o, do)
for i in range(1, num_blocks):
    # produce next block
    q_next, k_next, o_next, do_next = load_block(i)
    s_next, dp_next = cube_produce_block(q_next, k_next, o_next, do_next)
    # consume current block
    p, ds = vector_consume_block(s, dp)
    q, k, do = load_block(i-1)
    dq, dk, dv = cube_consume_block(q, k, do, p, ds)
    # swap
    s, dp = s_next, dp_next
# consume last block
p, ds = vector_consume_block(s, dp)
q, k, do = load_block(num_blocks-1)
dq, dk, dv = cube_consume_block(q, k, do, p, ds)
```

### 4. L1 缓存复用
注意到，C3, C4, C5 需要使用Q, K, dO 三个输入，而这三个输入在计算C1和C2时已经加载到L1上了，如果L1空间允许，可以将这三个输入保留在L1上，复用到C3, C4, C5中，减少GM到L1的传输。此时，L1上同时存在两个块的Q, K, dO, 内存增加一倍。代码变成了：
```python
# preload
q, k, do, s, dp = cube_produce_block(0)
for i in range(1, num_blocks):
    # produce next block
    q_next, k_next, do_next, s_next, dp_next = cube_produce_block(i)
    # consume current block
    p, ds = vector_consume_block(s, dp)
    dq, dk, dv = cube_consume_block(q, k, do, p, ds)
    store(dq, dk, dv)
    # swap
    q, k, do, s, dp = q_next, k_next, do_next, s_next, dp_next
# consume last block
p, ds = vector_consume_block(s, dp)
dq, dk, dv = cube_consume_block(q, k, do, p, ds)
```
这里已经超出了`triton-ascend`的表达能力了：在lowering到内存分配阶段时，编译器将q_next, k_next, do_next 的内存分配覆盖到了q, k, do上，导致计算流又编程了完全串行的形式。对应的mlir示意如下：
```mlir
scf.for %arg30 = %c0_i32 to %35 step %c1_i32 iter_args(%arg31= %28,....) ->(tensor<128x128xf16>, tensor<128x128xf16>, tensor<128x128xf16>, i32, i32): i32 {
    ....
    scf.yield %67, %68, %70, %56, %58: tensor<128x128xf16>, tensor<128x128xf16>, tensor<128x128xf16>, i32, i32
}
```
`scf.for`的iter_args部分，`q`, `k`, `do` 为`tensor<128x128xf16>`类型的输入，如果同时存在两份，那么必然`scf.for`的iter_args部分需要定义六个`tensor<128x128xf16>`类型的输入，因为我们只显式地在for循环外load了一份数据，在for循环内load的下一块数据会被编译器认为是对前一块数据的覆盖。为了解决这个问题，我们需要告诉编译器：for循环内同时存在两份数据——于是，我们将for循环以2为步长展开：
```python
# preload
curr_q, curr_k, curr_do, curr_s, curr_dp = cube_produce_block(0)
for i in range(0, num_blocks, 2):
    # produce next block
    next_q, next_k, next_do, next_s, next_dp = cube_produce_block(i+1)
    # consume current block
    p, ds = vector_consume_block(curr_s, curr_dp)
    dq, dk, dv = cube_consume_block(curr_q, curr_k, curr_do, p, ds)
    store(dq, dk, dv)
    # swap
    next_next_q, next_next_k, next_next_do, next_next_s, next_next_dp = cube_produce_block(i+2)
    # consume next block
    p, ds = vector_consume_block(next_s, next_dp)
    dq, dk, dv = cube_consume_block(next_q, next_k, next_do, p, ds)
    store(dq, dk, dv)
    # swap
    curr_q, curr_k, curr_do, curr_s, curr_dp = next_next_q, next_next_k, next_next_do, next_next_s, next_next_dp
# consume last block if odd
p, ds = vector_consume_block(curr_s, curr_dp)
dq, dk, dv = cube_consume_block(curr_q, curr_k, curr_do, p, ds)
```
这份代码已经可以达成复用目的了，随之而来的是我们必须对当前块id进行合法性判断，以避免搬运越界。代码不可避免地长成了这样：
```python
# preload
curr_q, curr_k, curr_do, curr_s, curr_dp = cube_produce_block(0)
for i in range(0, num_blocks, 2):
    # produce next block
    if valid(i):
        next_q, next_k, next_do, next_s, next_dp = cube_produce_block(i+1)
    # consume current block
    p, ds = vector_consume_block(curr_s, curr_dp)
    dq, dk, dv = cube_consume_block(curr_q, curr_k, curr_do, p, ds)
    store(dq, dk, dv)
    # swap
    if valid(i+1):
        next_next_q, next_next_k, next_next_do, next_next_s, next_next_dp = cube_produce_block(i+2)
    # consume next block
    if valid(i):
        p, ds = vector_consume_block(next_s, next_dp)
        dq, dk, dv = cube_consume_block(next_q, next_k, next_do, p, ds)
        store(dq, dk, dv)
    # swap
    if valid(i+1):
        curr_q, curr_k, curr_do, curr_s, curr_dp = next_next_q, next_next_k, next_next_do, next_next_s, next_next_dp
# consume last block if odd
p, ds = vector_consume_block(curr_s, curr_dp)
dq, dk, dv = cube_consume_block(curr_q, curr_k, curr_do, p, ds)
```
然而这样的代码是无法编译通过的，因为`next_q, next_k, ...`等变量在`if valid(i)`语句块外是不可见的。为了解决这个问题，我们需要换一种思路，即将load和compute分开，如果不合法我们就load地址为0的数据，可以参与计算，不store回去即可。最终代码如下：
```python
curr_q, curr_k, curr_do, curr_s, curr_dp = cube_produce_block(0)
for i in range(0, num_blocks, 2):
    # produce next block
    next_q, next_k, next_do, next_s, next_dp = cube_produce_block(i+1)
    # consume current block
    p, ds = vector_consume_block(curr_s, curr_dp)
    dq, dk, dv = cube_consume_block(curr_q, curr_k, curr_do, p, ds)
    if valid(i):
        store(dq, dk, dv)
    next_next_q, next_next_k, next_next_do, next_next_s, next_next_dp = cube_produce_block(i+2)
    p, ds = vector_consume_block(next_s, next_dp)
    dq, dk, dv = cube_consume_block(next_q, next_k, next_do, p, ds)
    if valid(i+1):
        store(dq, dk, dv)
    curr_q, curr_k, curr_do, curr_s, curr_dp = next_next_q, next_next_k, next_next_do, next_next_s, next_next_dp
# consume last block if odd
p, ds = vector_consume_block(curr_s, curr_dp)
dq, dk, dv = cube_consume_block(curr_q, curr_k, curr_do, p, ds)
if valid(num_blocks-1):
    store(dq, dk, dv)
```

### 5. L0C 复用
当算出结果`dQ, dK, dV`搬到GM上时，实际是通过`atomic_add`的方式累加回去。注意到，有3个`FIXPIPE`，最终会使得算子的性能受`FIXPIPE`的带宽限制。注意到：
1. 对于同一个S1块，输出的`dQ`对应同一块地址，可以在算完matmul后不搬出去，作为下次matmul的累加输入。
2. 对于同一个S2块，输出的`dK, dV`对应同一块地址，可以在算完matmul后不搬出去，作为下次matmul的累加输入。
![L0C-reuse](/images/flash-attention-grad-l0c-reuse.svg)

为了达成这个目的，我们需要设置计算块的顺序（显然优先固定S2块，因为此时dK, dV可以复用L0C，减少两个`FIXPIPE`）:
![L0C-reuse-task-schedule](/images/flash-attention-grad-task-schedule.svg)

每个核沿着S1 方向处理连续多块，每次store出dQ, 在更换S2方向索引时，再将dK, dV store出。

### 6. 负载均衡
我们可以根据每个分块的索引计算出该块对应的S1, S2方向的索引：
$$
s2\\_idx = block\\_id / num\\_blocks\\_S1 \\\\
s1\\_idx = block\\_id \\% num\\_blocks\\_S1
$$
当`causal=True`时，我们需要跳过不需要计算的块，我们需要求解一元二次方程:
$$
s2\\_idx ^ 2 -(2 * num\\_blocks\\_S1 -1) * s2\\_idx - 2(num\\_blocks\\_S1 - 1 - block\\_idx) = 0
$$
它对应的根为：
$$
s2\\_idx = \\frac{(2 * num\\_blocks\\_S1 -1) - \\sqrt{(2 * num\\_blocks\\_S1 -1)^2 + 8(num\\_blocks\\_S1 - 1 - block\\_idx)}}{2}
$$
在这里，sqrt导致编译保存，最终我们通过牛顿迭代法求解这个方程{{<sidenote>}}为什么10 次迭代，值得讨论{{</sidenote>}}：
```python
term1 = (NUM_BLOCKS_M * 2 - 1) * (NUM_BLOCKS_M * 2 - 1)
term2 = 8 * (NUM_BLOCKS_M - 1 - flat_idx_in_batch)
delta = term1 + term2
val = delta
guess = val
guess = (guess + val // tl.maximum(guess, 1)) // 2
guess = (guess + val // tl.maximum(guess, 1)) // 2
guess = (guess + val // tl.maximum(guess, 1)) // 2
guess = (guess + val // tl.maximum(guess, 1)) // 2
guess = (guess + val // tl.maximum(guess, 1)) // 2
guess = (guess + val // tl.maximum(guess, 1)) // 2
guess = (guess + val // tl.maximum(guess, 1)) // 2
guess = (guess + val // tl.maximum(guess, 1)) // 2
guess = (guess + val // tl.maximum(guess, 1)) // 2
guess = (guess + val // tl.maximum(guess, 1)) // 2
numerator = (NUM_BLOCKS_M * 2 - 1) - guess
s2_idx = (numerator + 1) // 2
s1_idx = flat_idx_in_batch - ((NUM_BLOCKS_M * 2 - 1 - s2_idx) * s2_idx) // 2 
```

### 7. 分形
Matmul计算要求分形为`NZ`格式，在Vecto单元上计算出结果为`ND`格式，需要手动将其转换为`NZ`格式。P的结果为`[HALF_BLOCK_M, BLOCK_N]`， 它对应的`NZ`格式为`[BLOCK_N/16, HALF_BLOCK_M/16, 16, 16]`，转换代码如下：
```python
p_nz = tl.permute(p.reshape(HALF_BLOCK_M/16, 16, BLOCK_N/16, 16), (2, 0, 3, 1))
```
但是这段代码对应的vector耗时过长，无奈之下，写成寄存器编程的方式：
```python
@triton.jit
def _softmax_vf(
    s, 
    l,
    sm_scale,
    HALF_BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    dtype: tl.constexpr
):
    HALF_BLOCK_N = BLOCK_N // 2
    p_nz = bl.allocate_local_buffer(dtype, [BLOCK_N // 16, HALF_BLOCK_M, 16], al.address_space.UB)
    p_nz = bl.to_tensor(p_nz)
    
    with al.Scope(vector_mode="simd", outline=True, no_inline=True):
        for loop in tl.static_range(HALF_BLOCK_M):
            s_loop_0 = tl.extract_slice(s, [loop, 0], [1, HALF_BLOCK_N], [1, 1])
            s_loop_1 = tl.extract_slice(s, [loop, HALF_BLOCK_N], [1, HALF_BLOCK_N], [1, 1])
            l_loop = tl.extract_slice(l, [loop], [1], [1])
            p_loop_0 = tl.math.exp(s_loop_0 * sm_scale - l_loop)
            p_loop_1 = tl.math.exp(s_loop_1 * sm_scale - l_loop)
            p_loop_0 = p_loop_0.reshape(HALF_BLOCK_N//16, 1, 16)
            p_loop_0_cast = p_loop_0.to(dtype)
            p_loop_1 = p_loop_1.reshape(HALF_BLOCK_N//16, 1, 16)
            p_loop_1_cast = p_loop_1.to(dtype)

            p_nz = tl.insert_slice(p_nz, p_loop_0_cast, [0, loop, 0], [HALF_BLOCK_N//16, 1, 16], [1, 1, 1])
            p_nz = tl.insert_slice(p_nz, p_loop_1_cast, [HALF_BLOCK_N//16, loop, 0], [HALF_BLOCK_N//16, 1, 16], [1, 1, 1])
    return p_nz
```
编译器会识别这种模式，编译成`vsstb`指令，连续搬运固定间隔的16个数到`UB`上。

### 8. 总结
通过以上优化，我们最终实现了接近AscendC实现的性能，实测结果0.94xAscendC实现。另一方面，由于Triton的表达能力有限，我们不得不通过代码展开等方式来规避编译器的限制，导致代码臃肿且难以维护。希望未来Triton能够支持更复杂的内存分配策略，从而简化代码实现。