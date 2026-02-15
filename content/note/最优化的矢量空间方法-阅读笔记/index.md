+++
date = '2024-12-28T16:33:00+08:00'
title = '《最优化的矢量空间方法》阅读笔记'
tags = ['Optimization', 'Math']
+++
本文记录了对《Optimization by Vector Space Methods》的阅读笔记。
<!--more-->
{{<katex>}}
## 1. 范数
范数是定义在矢量空间\(X\)上的函数\(\|\cdot\|:X\to\mathbb{R}\)，满足以下性质：
1. \(\|x\|\ge 0\)，且\(\|x\|=0\)当且仅当\(x=0\)；
2. \(\|\alpha x\|=|\alpha|\|x\|\)，其中\(\alpha\in\mathbb{R}\)；
3. \(\|x+y\|\le\|x\|+\|y\|\)。

## 2. 泛函的极值和紧性
魏尔斯特拉斯定理（Weierstrass Theorem）：定义在赋范线性空间\(X\)上的紧子集\(K\)上的上半连续泛函\(f\)，在\(K\)上可达到其最大值。

证明定义在\(C[0, 1]\) 中的单位球面不是紧集。
> [!INFO] 范数定义如下：$$\|x\|=\max_{t \in [0,1]}{|x(t)|}$$

定义泛函\(f(x)=\int_{0}^{1/2} x(t)dt - \int_{1/2}^{1}x(t)dt\),我们先确定 \(|f(x)|\) 的上界。对于任意 \(x \in S\)（即 \(\|x\|_\infty = 1\)），有:
$$
\begin{aligned}
|f(x)| &= \left| \int_{0}^{1/2} x(t)dt - \int_{1/2}^{1} x(t)dt \right| \\
&\le \int_{0}^{1/2} |x(t)|dt + \int_{1/2}^{1} |x(t)|dt \\
&= \int_{0}^{1} |x(t)|dt \\
&\le \sup_{t \in [0,1]} |x(t)| \cdot 1 = 1
\end{aligned}
$$
所以 \(f(x)\) 的值被限制在 \(1\) 以内。接下来，我们来证明最大值无法取到。
如果：$$\int_{0}^{1/2} x_0(t)dt - \int_{1/2}^{1} x_0(t)dt = 1$$且我们已知 \(|x_0(t)| \le 1\)，那么：$$\int_{0}^{1/2} x_0(t)dt \le \frac{1}{2} \quad \text{且} \quad -\int_{1/2}^{1} x_0(t)dt \le \frac{1}{2}$$两个小于等于 \(1/2\) 的数相加要等于 1，说明它们必须同时取等号：\(\int_{0}^{1/2} x_0(t)dt = \frac{1}{2}\)由于 \(x_0(t) \le 1\) 且是连续函数，这要求在区间 \([0, 1/2]\) 上恒有 \(x_0(t) = 1\)。特别是，在端点处 \(x_0(1/2) = 1\)。\(-\int_{1/2}^{1} x_0(t)dt = \frac{1}{2} \implies \int_{1/2}^{1} x_0(t)dt = -\frac{1}{2}\)。同理，由于 \(x_0(t) \ge -1\) 且是连续函数，这要求在区间 \([1/2, 1]\) 上恒有 \(x_0(t) = -1\)。特别是，在端点处 \(x_0(1/2) = -1\)。矛盾出现：函数 \(x_0\) 在 \(t=1/2\) 处必须同时等于 \(1\) 和 \(-1\)。$$1 = x_0(1/2) = -1$$这对于连续函数是不可能的（事实上，能够取到最大值的函数是阶跃函数，属于 \(L^\infty\) 空间但不属于 \(C[0, 1]\)）。

根据魏尔斯特拉斯定理的逆否命题：如果一个定义域上的连续函数无法取到最值，则该定义域不是紧集。因此，\(C[0, 1]\) 中的单位球面不是紧集。

## 3. 内积空间
定义了内积的线性矢量空间称为内积空间。内积定义在\(X\times X\)上且满足：
1. \(\langle x, y \rangle = \overline{\langle y, x \rangle}\)，共轭对称性；
2. \(\langle ax + by, z \rangle = a\langle x, z \rangle + b\langle y, z \rangle\)，线性性；
3. \(\langle x, x \rangle \ge 0\)，且\(\langle x, x \rangle = 0\)当且仅当\(x=0\)。

内积空间中的范数定义为：$$\|x\|=\sqrt{\langle x, x \rangle}$$。

## 4. 希尔伯特空间
希尔伯特空间是完备的内积空间。空间\(E^n\)，\(l_2\)，\(L_2\)都是希尔伯特空间。

## 5. 投影定理
**正交**：在内积空间中，对于矢量\(x\)和\(y\)，如果\(\langle x, y \rangle = 0\)，则称\(x\)和\(y\)正交，记作\(x\perp y\)。如果对于集合\(M\)中的所有矢量\(y\)，都有\(\langle x, y \rangle = 0\)，则称\(x\)是\(M\)的正交，记作\(x\perp M\)。

设\(X\)是一个内积空间，\(M\)是\(X\)的子空间，\(x\) 是\(X\)中的任一矢量。若有\(m_0\in M\)，使得对于所有的\(m\in M\)，都有\(\|x-m_0\|\le\|x-m\|\)，则\(m_0\)是唯一的。而且\(m_0 \in M\) 是\(M\)中唯一的极小化矢量的充要条件是差矢量 \(x-m_0 \perp M\)。

**投影定理**：
设\(H\)是希尔伯特空间，\(M\)是\(H\)的**闭子空间**。对任一矢量\(x \in H\)，都有唯一的矢量\(m_0 \in M\)，使得\(\|x-m_0\|\le\|x-m\|\)，并且\(m_0 \in M\)是唯一的极小化矢量的充要条件是\(m_0 \perp M\)。
> [!INFO] 这里闭子空间保证存在性。

![投影定理](projection.svg "投影定理")
 {style="width:50%;"}

## 6. 格拉姆矩阵
假定\(y_1, y_2, \dots, y_n\)是希尔伯特空间\(H\)中的一组矢量，用这些矢量生成\(H\)中的一个有限维（闭）子空间$M$。任意给定一个矢量\(x \in H\)，求一个矢量 \(\hat{x}\)，它是\(M\)中最接近\(x\)的矢量。如果将\(\hat{x}\)表示为\(y_1, y_2, \dots, y_n\)的线性组合，即\(\hat{x}=\sum_{i=1}^{n}\alpha_iy_i\)。这个逼近问题等价于找到\(\alpha_1, \alpha_2, \dots, \alpha_n\)，使得\(\|x- \hat{x}\|\)最小。
根据投影定理，\(\hat{x}\)是\(M\)中最接近\(x\)的矢量，当且仅当\(x-\hat{x}\perp y_i\)，即\(\langle x-\hat{x}, y_i\rangle = 0\)，\(i=1,2,\dots,n\)。写成下面方程组：
$$
\begin{cases}
\langle y_1, y_1\rangle\alpha_1 + \langle y_1, y_2\rangle\alpha_2 + \dots + \langle y_1, y_n\rangle\alpha_n = \langle x, y_1\rangle \\
\langle y_2, y_1\rangle\alpha_1 + \langle y_2, y_2\rangle\alpha_2 + \dots + \langle y_2, y_n\rangle\alpha_n = \langle x, y_2\rangle \\
\vdots \\
\langle y_n, y_1\rangle\alpha_1 + \langle y_n, y_2\rangle\alpha_2 + \dots + \langle y_n, y_n\rangle\alpha_n = \langle x, y_n\rangle
\end{cases}
$$
这个以\(\alpha_1, \alpha_2, \dots, \alpha_n\)为未知数的线性方程组，叫做极小化问题的法方程。

对于矢量\(y_1, y_2, \dots, y_n\)的\(n\times n\)矩阵
$$
G(y_1, y_2, \dots, y_n) = \begin{pmatrix}
\langle y_1, y_1\rangle & \langle y_1, y_2\rangle & \dots & \langle y_1, y_n\rangle \\
\langle y_2, y_1\rangle & \langle y_2, y_2\rangle & \dots & \langle y_2, y_n\rangle \\
\vdots & \vdots & \ddots & \vdots \\
\langle y_n, y_1\rangle & \langle y_n, y_2\rangle & \dots & \langle y_n, y_n\rangle
\end{pmatrix}
$$
称为\(y_1, y_2, \dots, y_n\)的格拉姆矩阵。
> [!INFO] \(G(y_1, y_2, \dots, y_n)\)是半正定的。

## 7. Gram-Schmidt 正交化过程

在 \(L^2[-1,1]\) 空间中，内积定义为：
$$
\langle f, g \rangle = \int_{-1}^{1} f(t)g(t) \, dt
$$
我们设原始基底为 \(x_0 = 1, x_1 = t, x_2 = t^2\)。我们的目标是求出正交序列 \(u_0, u_1, u_2\)。

第一步：计算 \(u_0\) (对应 \(n=0\))

Gram-Schmidt 的第一步直接取第一个向量：
$$
u_0 = x_0 = 1
$$
为了后续计算方便，我们要算出它的模长平方（自内积）：
$$
\langle u_0, u_0 \rangle = \int_{-1}^{1} 1 \cdot 1 \, dt = [t]_{-1}^{1} = 2
$$

第二步：计算 \(u_1\) (对应 \(n=1\))
我们要从 \(x_1 = t\) 中减去它在 \(u_0\) 上的投影分量。公式为：$$
u_1 = x_1 - \frac{\langle x_1, u_0 \rangle}{\langle u_0, u_0 \rangle} u_0
$$
1.计算分子（内积）：
> [!INFO] 奇函数在对称区间的积分为 0

$$
\langle x_1, u_0 \rangle = \int_{-1}^{1} t \cdot 1 \, dt = \left[ \frac{t^2}{2} \right]_{-1}^{1} = 0
$$
2. 代入公式：
$$
u_1 = t - \frac{0}{2} \cdot 1 = t
$$
3.计算 \(u_1\) 的模长平方（为下一步做准备）：
$$
\langle u_1, u_1 \rangle = \int_{-1}^{1} t \cdot t \, dt = \int_{-1}^{1} t^2 \, dt = \left[ \frac{t^3}{3} \right]_{-1}^{1} = \frac{1}{3} - (-\frac{1}{3}) = \frac{2}{3}
$$

第三步：计算 \(u_2\) (对应 \(n=2\))

我们要从 \(x_2 = t^2\) 中减去它在 \(u_0\) 和 \(u_1\) 上的投影分量。公式为：
$$
u_2 = x_2 - \frac{\langle x_2, u_0 \rangle}{\langle u_0, u_0 \rangle} u_0 - \frac{\langle x_2, u_1 \rangle}{\langle u_1, u_1 \rangle} u_1
$$
1.计算第一个投影分量（对 \(u_0\)）：$$\langle x_2, u_0 \rangle = \int_{-1}^{1} t^2 \cdot 1 \, dt = \frac{2}{3}$$
系数为：\(\frac{2/3}{2} = \frac{1}{3}\)

2.计算第二个投影分量（对 \(u_1\)）：
$$\langle x_2, u_1 \rangle = \int_{-1}^{1} t^2 \cdot t \, dt = \int_{-1}^{1} t^3 \, dt = 0
$$
系数为：\(\frac{0}{2/3} = 0\)

3.代入公式：
$$
u_2 = t^2 - \frac{1}{3} \cdot u_0 - 0 \cdot u_1 = t^2 - \frac{1}{3}
$$

第四步：标准化（转换为标准勒让德多项式）

到现在为止，我们得到的正交多项式序列是：\(u_0 = 1,u_1 = t,u_2 = t^2 - \frac{1}{3}\)

这组多项式虽然是正交的，但它们还不是教科书上标准的勒让德多项式 (Legendre Polynomials, \(P_n\))。标准定义要求每个多项式在 \(t=1\) 处的取值为\(1\)（即 \(P_n(1)=1\)）。我们需要对上面的结果进行缩放。对于 \(n=0\):\(u_0(1) = 1\)，符合标准。$$P_0(t) = 1$$对于 \(n=1\):\(u_1(1) = 1\)，符合标准。$$P_1(t) = t$$对于 \(n=2\):我们计算 \(u_2(1) = 1^2 - \frac{1}{3} = \frac{2}{3}\)。为了让它变成 \(1\)，我们需要乘以倒数 \(\frac{3}{2}\)。$$P_2(t) = \frac{3}{2} \left( t^2 - \frac{1}{3} \right) = \frac{1}{2}(3t^2 - 1)$$
## 8. 对偶空间
赋范空间 \(X\) 的对偶空间通常记为 \(X^*\)（或 \(X'\)），它由 \(X\) 上所有有界线性泛函组成。在这个对偶空间 \(X^*\) 上，范数 \(\|\cdot\|_{X^*}\) 定义为算子范数。具体定义方式有以下几种等价形式。对于任意 \(f \in X^*\)：

1. 标准定义（单位球上的上确界）这是最常用的定义方式。\(f\) 的范数是它在 \(X\) 的闭单位球上的最大绝对值（上确界）：$$\|f\| = \sup_{\|x\| \le 1} |f(x)|$$
2. 等价定义以下几种表达方式在数学上是完全等价的
+ 单位球面上的上确界
> [!INFO] 仅当 \(X \neq \{0\}\) 时适用
> $$\|f\| = \sup_{\|x\| = 1} |f(x)|$$
+ 比值的上确界（直观理解为“最大放大倍率”）：$$\|f\| = \sup_{x \neq 0} \frac{|f(x)|}{\|x\|}$$
+ 最佳Lipschitz常数（最小上界）：$$\|f\| = \inf \{ M \ge 0 : |f(x)| \le M\|x\|, \forall x \in X \}$$
