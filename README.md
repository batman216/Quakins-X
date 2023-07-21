# QUAKINS .README

<aside>
🔥 **QUA**ntum **KIN**etic **S**olver is designed for calculating a Vlasov/Wigner-Poisson/Maxwell system. The multi-GPU version (the ultimate verion, QUAKINS-X) begins in ***April 2023***.

</aside>

## Physics & Algorithm

### Flux Balance 算法

Free stream 方程

$$
\frac{\partial f}{\partial t}+v\frac{\partial f}{\partial x}=0
$$

实际上就是 $f$ 沿 $x$ 平移：$f(x,t+\mathrm{d}t)=f(x-v\mathrm{d}t,t)$. 

![Free stream](QUAKINS%20README%207176d72b7dc34013af04046d18135526/free_stream.png)

Free stream

如果函数 $f$ 是用粒子描述的（拉格朗日观点），让每个粒子按速度  $v$ 运动就行了，如果是流体网格描述（欧拉观点） ，则稍微复杂一点点. 最简单的方法是**直接演化 $f$ 的傅里叶分量**

$$
f_k(t+\mathrm{d}t)=f_k(t)\mathrm{e}^{-ikv\mathrm{d}t}
$$

下面介绍 **flux balance 流体方法**. 若$t$ 时刻 $f_i$ 是已知的，$\mathrm{d}t$ 时间后，在以 $x_i$ 为中心，宽 $\mathrm{d}x$ 的微元范围内，速度大于零时，流体从右边界的流出去一部分，左边界流进来一部分，速度小于零时反之，记 $x$ 增大方向流动为正，记每点的流量为 $\Phi_i$ , 则有

$$
f_i(t+\mathrm{d}t)=f_i(t)+\Phi_{i-1/2}(t,\mathrm{d}t)-\Phi_{i+1/2}(t,\mathrm{d}t)
$$

所以求解过程归结到整个计算区域的流函数 $\Phi_{i+1/2}$ 上，如下图所示，我们对每个网格计算右边界的 $\Phi_{i+1/2}$，第 $i$ 个网格对应的 $\Phi_{i+1/2}$ 等于红线标记区间对 $f$ 积分（绿色阴影部分），QUAKINS中采用三点插值来计算这一段积分，用到了如图三个红×标记的点，注意到速度大于零和小于零时插值点取得不一样

![demo_fbm.png](QUAKINS%20README%207176d72b7dc34013af04046d18135526/demo_fbm.png)

当速度很大时，$v\mathrm{d}t$也许会比一个网格或者几个网格还长，如下图所示，此时 $x_{i+1/2}-v\mathrm{d}t$ 点掉在了第 $I$ 个网格，如下两图中分别有 $I=i-3$ ， $I=i+3$ ，此时只需要插值第 $I$ 个网格中的一小段积分，剩下的几个网格积分不需要插值，可以直接加起来 

![positive_v_fbm_multi.png](QUAKINS%20README%207176d72b7dc34013af04046d18135526/positive_v_fbm_multi.png)

![negative_v_fbm_multi.png](QUAKINS%20README%207176d72b7dc34013af04046d18135526/negative_v_fbm_multi.png)

$$
f_h(x)=f_i+(x-x_i)\frac{f_{i+1}-f_i}{\mathrm{d}x}
$$

$$
f_h(x)=f_i+\frac{1}{6\mathrm{d}x^2}\left[2(x-x_i)(x-x_{i-3/2})+(x-x_{i-1/2})(x-x_{i+1/2})\right](f_{i+1}-f_i)+\frac{1}{6\mathrm{d}x^2}\left[2(x-x_i)(x-x_{i+3/2})+(x-x_{i-1/2})(x-x_{i+1/2})\right](f_{i}-f_{i-1})
$$

综上有，当$v<0$时

$$
\Phi_{i+1/2}=-\sum_{a=i+1}^{I-1} f_a+\alpha_i\left[f_I-\frac{1}{6}\left(1-\alpha_i\right)\left(1+\alpha_i\right)\left(f_{I+1}-f_I\right)-\frac{1}{6}\left(2+\alpha_i\right)\left(1+\alpha_i\right)\left(f_{I}-f_{I-1}\right)\right]
$$

$v>0$时有

$$
\Phi_{i+1/2}=\sum_{a=I+1}^{i} f_a+\alpha_i\left[f_I+\frac{1}{6}\left(1-\alpha_i\right)\left(2-\alpha_i\right)\left(f_{I+1}-f_I\right)-\frac{1}{6}\left(1-\alpha_i\right)\left(1+\alpha_i\right)\left(f_{I}-f_{I-1}\right)\right]
$$

### 柱坐标下的二维泊松方程数组解

Poisson方程

$$
\mathcal{J}^{-1}\partial_a\mathcal{J}g^{ab}\partial_b\phi=-4\pi\rho
$$

柱坐标系下有 $g_{rr}=1,\,\,g_{\theta\theta}=r^2,\,\,g_{zz}=1,\,\,\mathcal{J}=r$, 于是

$$
\frac{1}{r}\frac{\partial}{\partial r}r\frac{\partial\phi}{\partial r}+\frac{1}{r^2}\frac{\partial^2\phi}{\partial\theta^2}+\frac{\partial^2\phi}{\partial z^2}=-4\pi\rho
$$

重新组织一下，假设极向对称

$$
r^2\frac{\partial^2\phi}{\partial r^2}+r\frac{\partial\phi}{\partial r}+\bcancel{\frac{\partial^2\phi}{\partial\theta^2}}+r^2\frac{\partial^2\phi}{\partial z^2}=-4\pi r^2\rho
$$

柱坐标的 $r$ 方向是不均匀的，不能用FFT求，因此只看$z$方向的傅里叶空间：

$$
\frac{\partial^2\phi_{k_z}}{\partial r^2}+\frac{1}{r}\frac{\partial\phi_{k_z}}{\partial r}-k_z^2\phi_{k_z}=-4\pi \rho_{k_z}
$$

离散化

$$
\frac{\phi^{k_z}_{i+1}-2\phi^{k_z}_{i}+\phi^{k_z}_{i-1}}{\Delta r^2}+\frac{\phi^{k_z}_{i+1}-\phi^{k_z}_{i-1}}{2r_i\Delta r}-k_z^2\phi^{k_z}_i=-4\pi \rho^{k_z}_i
$$

解得 $\boldsymbol{\phi}=\mathcal{A}^{-1}\boldsymbol{\rho}$，其中

$$
\mathcal A=-\begin{pmatrix}
\\
&\ddots & 1/\Delta r^2 +1/2r_{i-1}\Delta r& &\\ 
&1/\Delta r^2 -1/2r_i\Delta r & -2/\Delta r^2-k_z^2 & 1/\Delta r^2 +1/2r_i\Delta r&\\
& & 1/\Delta r^2 -1/2r_{i+1}\Delta r& \ddots&\\
\\
\end{pmatrix}\\=\mathrm{diag}\left(2/\Delta r^2+k_z^2\right)-\mathrm{diag}^{-1}\left( 1/\Delta r^2 +1/2r\Delta r\right)-\\\mathrm{diag}^{+1}\left( 1/\Delta r^2 -1/2r\Delta r\right)+\mathrm{b.\,c.\,terms}
$$

<aside>
💡 **虽然拉普拉斯算符是自伴（self-adjoint）算符，但矩阵 $\mathcal{A}$ 是非自伴的！** 很多求解线性方程组的算法是要求矩阵自伴的，比如Eigen的**`SimplicialLLT`** 之类的方法. 注意对实矩阵来说，自伴就是沿对角线对称.

</aside>

在 $r=0$ 处设置导数为零的诺伊曼边界条件，此时取第一个格点 $r_0=\Delta r/2$ ， 则有 $\phi_0=\phi_{-1}$，对应在矩阵 $\mathcal A$ 中只需要令 $\mathcal A_{00}=1/\Delta r^2+1/2r_0\Delta r+k_z^2$. 这样正好避免了 $r=0$ 处的奇异性（柱坐标下的 $r=0$ 点其实不奇异，两边乘以 $r$ 就行了，但是这样算起来麻烦一点）. 巧的是， $\mathcal A_{00}$ 和对角线上的所有 $\mathcal A_{ii}$ 都是相同的，所以实际上这个矩阵不需要做任何处理就自动满足 $\phi'|_{r=0}=0,\, \phi|_{r_\mathrm{max}}=0$ 的边界条件.

### 四维势的时间推进求解电磁波

$$

\square^2 A^\mu=4\pi J^\mu

$$

$$
\left(\partial^2_t-\nabla^2\right)\phi=4\pi\rho
$$

### 静电扰动与量子波动的耦合：Wigner方程

### 电磁场与量子波动的耦合：规范不变的电磁Wigner方程

## Implementation Details

### Quakins 计算流程图

![每张显卡中分配两块大小相同的内存用来存相空间分布函数，通过 reorder copy 在两块内存间来回拷贝调整四维矩阵的存储顺序，上图为四维相空间的一次空间推进](QUAKINS%20README%207176d72b7dc34013af04046d18135526/quakins_x_a_gpu_demo.png)

每张显卡中分配两块大小相同的内存用来存相空间分布函数，通过 reorder copy 在两块内存间来回拷贝调整四维矩阵的存储顺序，上图为四维相空间的一次空间推进

![以四张显卡为例，初始化数据是沿着其中一个空间维度并行的，因此进行空间推进前要先用NCCL并行交互，填满交互缓冲层，推进完成后将结果reduce积分成密度发送给cpu整合后做下一步解场准备](QUAKINS%20README%207176d72b7dc34013af04046d18135526/quakins_x_demo.png)

以四张显卡为例，初始化数据是沿着其中一个空间维度并行的，因此进行空间推进前要先用NCCL并行交互，填满交互缓冲层，推进完成后将结果reduce积分成密度发送给cpu整合后做下一步解场准备

### 基于 thrust 的 **Reorder copy** 实现以及速度测试

Reorder copy 是一个快速调整多维数组在连续内存中存储顺序的技术，其核心在于copy前后位置指标的关系. 这需要借助多维指标与一维指标的关系

$$

i_s =\frac{\mathrm{mod}\left(i,n_s\cdots n_0\right)}{n_{s-1}\cdots n_0}

$$

其中 $*i$* 和 $*i_s$* 分别为一维总指标和第 $*s$* 维的指标, $*s*$ 取 $0 ∼$ $*N − 1*$,  $n$ 为连续内存总数据量，$*n_i$* 为第 $*i$* 个维度的数据量，$*N$* 为数据维度，有 $n=\prod_{\alpha=1}^Nn_\alpha$. 按需将$\{i_s\}$的排列顺序置换为$*i_s'*$ 于是有

$$
⁍

$$

- Quakins 通过`thrust::scatter`实现显卡上的 reorder copy

![Reorder copy 在不同显卡上的计算耗时与数组长度的关系，结果显然是线性的，在4090上每增加一千四百万个格点耗时只增加约1毫秒 (ms)，拷贝速度是3080系列的3.5倍左右](QUAKINS%20README%207176d72b7dc34013af04046d18135526/reorder_copy_elapsed_time.png)

Reorder copy 在不同显卡上的计算耗时与数组长度的关系，结果显然是线性的，在4090上每增加一千四百万个格点耗时只增加约1毫秒 (ms)，拷贝速度是3080系列的3.5倍左右

### Flux Balance 自由扩散推进流体元测试

- 测试**跨网格推进**，看第一个维度的相空间，周期边条件的 $x_1$ 方向有一个 $\cos (2\pi x/L)$ 的扰动，自由扩散情况下，这个扰动会被相混掉，下图分别是dt=0.05跑80步，和dt=4跑一步的结果

![multi_step_fbm.png](QUAKINS%20README%207176d72b7dc34013af04046d18135526/multi_step_fbm.png)

![one_step_fbm.png](QUAKINS%20README%207176d72b7dc34013af04046d18135526/one_step_fbm.png)

- 可以看到，就自由输运方程而言，跨网格的推进方法理论上一步能走无穷远，实际最大推进步长只受限于 ghost 网格的数量
- 速度空间网格稀疏带来的离散相混问题

### NCCL并行交互测试

- Quakins 并行交互采用 **MPI+NCCL 协议**，每个MPI线程控制一张显卡，可以兼容跨节点模拟.
- 目前，双卡4090PC和六卡服务器的显卡都是插在在PCIe4.0**×**8上的，理论点对点峰值速度应该有16G/s，但是我测试出来都只有6G/s，不知道为什么
- 下图是**200×200×252×1272**的单精度四维相空间网格（128亿个float，占内存47.76G）下热电子自由扩散的模拟，用了六张4090，推进了500步，耗时409秒（约810ms一步）

![一团运动的热电子在自由扩散，这里没有去掉并行交互的缓冲层，也可以看到电子跨越了6张显卡](QUAKINS%20README%207176d72b7dc34013af04046d18135526/hot_electron_multi_gpu.gif)

一团运动的热电子在自由扩散，这里没有去掉并行交互的缓冲层，也可以看到电子跨越了6张显卡

![去掉缓冲层，结果看起来正常了](QUAKINS%20README%207176d72b7dc34013af04046d18135526/hot_electron_multi_gpu_no_buff.gif)

去掉缓冲层，结果看起来正常了

![并行效率测试，图中每次算例都保证每张卡的计算量相同，即6卡算例的计算量为2卡算例的三倍，总网格数为216*216*200*200*(卡数)](QUAKINS%20README%207176d72b7dc34013af04046d18135526/ncclfree_time.png)

并行效率测试，图中每次算例都保证每张卡的计算量相同，即6卡算例的计算量为2卡算例的三倍，总网格数为216*216*200*200*(卡数)

- 测试发现，增加显卡数量并不会明显提高卡间并行交互时间，且相空间推进的并行效率几乎为100%，符合预期，下图为6卡运行现场

![nvidia-smi.png](QUAKINS%20README%207176d72b7dc34013af04046d18135526/nvidia-smi.png)

### 反射边界条件的实现及验证

- 该算法的精髓在于一个自由翻转的**跨步分块迭代器 (strided chunk iterator)**，将 v=-v 处ghost网格长度范围内的值反向拷贝到 ghost 网格中，如下图所示. 与 free stream 类似，最外层速度空间是串行的.

![reflection_boundary_demo.png](QUAKINS%20README%207176d72b7dc34013af04046d18135526/reflection_boundary_demo.png)

![r方向反射边条件，z方向周期边条件，一团热电子在反弹，由于Poisson方程求解时$r=r_{\mathrm{max}}$ 设置了0边条件，因此电子运动到边界附近时的势场解是错的](QUAKINS%20README%207176d72b7dc34013af04046d18135526/reflecting_demo.gif)

r方向反射边条件，z方向周期边条件，一团热电子在反弹，由于Poisson方程求解时$r=r_{\mathrm{max}}$ 设置了0边条件，因此电子运动到边界附近时的势场解是错的

<aside>
💡 注意推进时 ghost 网格不要设置为0！否则当边界点上有流体时，从边界到 ghost 网格的突变将导致守恒流算法的插值产生误差. 因此不需要使用的 ghost 网格最好填充与边界点相等的值

</aside>

### 柱坐标 Poisson 方程求解过程及验证

- 目前，涉及到矩阵反除的Poisson方程的求解都要送给CPU算，矩阵反除用Eigen来算. 具体对于柱坐标，矩阵只有三条对角线的非零值，Quakins采用`Eigen::SparseLU` 求解.
- `fftPlanMany` 的高级输入输出格式可以自动实现 reorder copy, 不用自己再 copy 一次.
- 验证求解器的准确性，分离变量 $\phi(r,z)=\mathcal R(r)\mathcal Z(z)$，利用 $J_m(r)$ 为方程

$$
\left[\frac{1}{r}\frac{\partial}{\partial r}\left(r\frac{\partial}{\partial r}\right)+\frac{m^2}{r^2}\right]\mathcal R(r)=k_r^2\mathcal R(r)
$$

的本征函数，本征值 $k_r^2=(\mu_m^{(i)}/r_{\mathrm{max}})^2$，$\mu_m^{(i)}$ 为 $J_m(x)$ 的第 $i$ 个零点（因为右边界是零边界条件），于是有

$$
\frac{1}{r}\frac{\partial}{\partial r}\left(r\frac{\partial}{\partial r}\right)J_0(k_rr)=k_r^2J_0(k_rr),\quad\frac{1}{r}\frac{\partial}{\partial r}\left(r\frac{\partial}{\partial r}\right)J_0(r)=J_0(r)
$$

将密度 $\rho$  设置为 $CJ_0(k_rr)$，则解得的 $\phi(r)=CJ_0(k_rr)/k_r^2$ . 这里 $r=10$ 处设置了0边界条件，如果 $k_r$ 选的不是本征值是无法得到正确解的

![八个节点的零阶贝塞尔函数，数值解与理论符合得很好](QUAKINS%20README%207176d72b7dc34013af04046d18135526/quakins_poisson_test_8.png)

八个节点的零阶贝塞尔函数，数值解与理论符合得很好

当 $z$ 方向不均匀时，在其傅里叶空间有 

$$
\frac{1}{r}\frac{\partial}{\partial r}\left(r\frac{\partial}{\partial r}\right)J_0(k_rr)=\left(k_r^2+k_z^2\right)J_0(k_rr)
$$

### 速度空间的傅里叶谱推进方案及测试

- 傅里叶变换采用多维 `cufftR2C/cufftC2R` , R2C 的算法输入和输出数组长度不一样，对于一维FFT, 输入N个 `cufftReal`，输出 N/2+1个 `cufftComplex`, 因此输出数组比输入数组多一个 `cufftComplex` 或者两个 `cufftReal` . 对于多维FFT, 实际上只有最内层的维度是 R2C 的, 因为多维FFT是从内到外每个维度分别FFT, 第一层 R2C 后就只能 C2C 了.
- Quakins 默认速度空间内侧维度的最后两个元素是无效的，用来作为R2C的 padding 元素
- 给一团速度的电子一个恒定的力：

![有个$-z$方向的力](QUAKINS%20README%207176d72b7dc34013af04046d18135526/force_demo.gif)

有个$-z$方向的力

![有个$-r$方向的力](QUAKINS%20README%207176d72b7dc34013af04046d18135526/force_demo1.gif)

有个$-r$方向的力

### 速度空间 Wigner 势能项推进方案及测试

经典力学中，分布函数速度空间的每一个傅里叶分量的震荡频率都只与局域势场相关，且与速度的傅里叶对偶是线性相关的：

$$
f_{\boldsymbol\lambda}(\boldsymbol x,t+\mathrm{d}t)=f_{\boldsymbol\lambda}(\boldsymbol x,t)\exp\left[-\mathrm{i} \mathrm{d}t\boldsymbol\lambda\cdot\boldsymbol \nabla\phi(\boldsymbol x)\right]
$$

而量子力学中是非线性非局域的

$$
f_{\boldsymbol\lambda}(\boldsymbol x,t+\mathrm{d}t)=f_{\boldsymbol\lambda}(\boldsymbol x,t)\exp\left\{-\mathrm{i}\mathrm{d}t \left[\phi\left(\boldsymbol x+\frac{\boldsymbol\lambda}{2}\right)-\phi\left(\boldsymbol x-\frac{\boldsymbol\lambda}{2}\right)\right] \right\}
$$

这里的 $\lambda$ 视为分布函数的量子关联长度，因为从 Wigner 方程的推导可以看出速度（动量）原来就是关联长度的傅里叶对偶.

- **从物理上看，$\lambda$** 的实际分布宽度应该与热速度分布成反比：$\Delta\lambda_q=2\pi/v_\mathrm{the}$. 对于单粒子 Wigner 方程，这其实就是德布罗意波长?

$$
\Delta\lambda = 2\pi/v_\mathrm{max},\quad\lambda_\mathrm{max}=2\pi/\Delta v
$$

- 计算 Wigner 势能需要对势场插值，Quakins 采用 **CUDA 纹理插值（texture interpolation）**技术

## Maybes

- 也许 reorder copy 和相空间推进可以简化成一次矩阵操作
- 也许可以用粒子蒙特卡洛解Wigner方程
- 也许可以用delta-f方法解Wigner方程