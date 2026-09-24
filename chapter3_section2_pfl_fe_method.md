# 第三章 系统设计

## 3.2 个性化联邦学习与功能加密聚合方法

### 3.2.1 方法概述

针对非独立同分布数据下统一全局模型难以同时适应不同客户端，以及明文上传模型更新可能泄露
本地训练信息的问题，本文将个性化参数隔离与线性功能加密结合。方法的基本思路是：客户端仅
共享能够跨用户迁移的表示参数，将与本地分布直接相关的预测头保留在本地；对于需要上传的
共享参数更新，客户端先进行裁剪和定点编码，再通过 DMCFE-IP 接口加密。服务器只能利用与
样本权重向量绑定的功能密钥恢复共享更新的加权和，不能从协议规定的输出中直接获得任一客户
端的独立更新。

设共享表示参数为 $\boldsymbol{\theta}^{s}$，客户端 $C_i$ 的个性化参数为
$\boldsymbol{\theta}^{p}_i$。本文求解的个性化联邦优化目标为

\[
\min_{\boldsymbol{\theta}^{s},\{\boldsymbol{\theta}^{p}_i\}_{i=1}^{K}}
F\!\left(\boldsymbol{\theta}^{s},
\{\boldsymbol{\theta}^{p}_i\}_{i=1}^{K}\right)
=\sum_{i=1}^{K}p_i
F_i\!\left(\boldsymbol{\theta}^{s},\boldsymbol{\theta}^{p}_i\right),
\qquad
p_i=\frac{n_i}{\sum_{j=1}^{K}n_j}.
\]

其中，$F_i$ 表示客户端 $C_i$ 的本地经验风险，$n_i$ 表示其本地样本数。该目标保留一个跨
客户端学习的共享表示，同时允许每个客户端通过 $\boldsymbol{\theta}^{p}_i$ 适应自身数据
分布。整个方法由参数划分与初始化、个性化双阶段训练、共享更新编码、功能加密聚合以及掉线
恢复五个部分组成。

### 3.2.2 共享参数与个性化参数划分

在训练开始前，系统根据模型模块名称或参数选择规则，将参与训练的浮点参数键集合划分为共享参数键集合
$\mathcal{K}^{s}$ 和个性化参数键集合 $\mathcal{K}^{p}$，并满足

\[
\mathcal{K}^{s}\cap\mathcal{K}^{p}=\varnothing,
\qquad
\mathcal{K}^{s}\cup\mathcal{K}^{p}=\mathcal{K}.
\]

对于分类模型，特征提取层或隐藏表示层属于 $\mathcal{K}^{s}$，最后的预测层属于
$\mathcal{K}^{p}$。以本文的 MNIST 卷积模型为例，卷积层和第一全连接层构成共享表示，最后
的线性分类层作为客户端个性化头。服务器只维护 $\boldsymbol{\theta}^{s}$；客户端 $C_i$ 以
自身标识为索引，持久化保存 $\boldsymbol{\theta}^{p}_i$。当客户端在后续轮次再次被采样时，
系统加载该客户端上次训练后的个性化参数，而不是使用其他客户端的预测头。

这种显式划分同时限定了隐私保护的边界。只有 $\mathcal{K}^{s}$ 中的浮点参数更新进入向量化
和聚合流程；$\mathcal{K}^{p}$ 中的参数不构造上传消息，也不参与服务器平均。对于模型状态中
的整数缓冲区，仅当所有参与客户端的对应状态一致时才允许复制，从而避免对不适合定点加权的
离散状态执行无意义的平均。

### 3.2.3 个性化双阶段本地训练

在第 $t$ 轮，服务器从全部客户端中选择参与集合 $\mathcal{A}_t$，并广播共享参数
$\boldsymbol{\theta}^{s}_t$。客户端 $C_i\in\mathcal{A}_t$ 将其与本地保存的
$\boldsymbol{\theta}^{p}_{i,t}$ 组合，形成当前本地模型。为减小共享表示和个性化头同时更新
产生的相互干扰，本地优化被划分为两个连续阶段。

第一阶段固定共享参数，仅优化个性化参数。若个性化训练共执行 $E_p$ 个本地 epoch，则一次
梯度更新可表示为

\[
\boldsymbol{\theta}^{p}_{i}
\leftarrow
\boldsymbol{\theta}^{p}_{i}
-\eta_p\nabla_{\boldsymbol{\theta}^{p}_{i}}
F_i\!\left(\boldsymbol{\theta}^{s}_t,
\boldsymbol{\theta}^{p}_{i}\right),
\qquad
\boldsymbol{\theta}^{s}=\boldsymbol{\theta}^{s}_t,
\]

其中 $\eta_p$ 为个性化参数学习率。该阶段使预测头先适应客户端当前的数据分布。

第二阶段固定更新后的个性化参数，仅优化共享表示。设共享表示训练执行 $E_s$ 个本地 epoch，
则其梯度更新为

\[
\boldsymbol{\theta}^{s}_{i}
\leftarrow
\boldsymbol{\theta}^{s}_{i}
-\eta_s\nabla_{\boldsymbol{\theta}^{s}_{i}}
F_i\!\left(\boldsymbol{\theta}^{s}_{i},
\boldsymbol{\theta}^{p}_{i,t+1}\right),
\qquad
\boldsymbol{\theta}^{s}_{i}\big|_{0}=\boldsymbol{\theta}^{s}_t,
\]

其中 $\eta_s$ 为共享参数学习率。训练完成后，客户端将
$\boldsymbol{\theta}^{p}_{i,t+1}$ 保存到本地，并计算共享更新

\[
\Delta_{i,t}^{s}
=\widetilde{\boldsymbol{\theta}}^{s}_{i,t}
-\boldsymbol{\theta}^{s}_t.
\]

因此，服务器聚合的对象只包含客户端对共享表示的修正量。个性化头既不出现在
$\Delta_{i,t}^{s}$ 中，也不用于生成后续功能加密密文。

### 3.2.4 共享更新的裁剪与定点编码

DMCFE-IP 的明文空间为整数域，而神经网络更新通常为浮点张量。为建立二者之间的确定性映射，
客户端首先按照预先约定的参数键顺序，将共享更新展平为 $d$ 维向量

\[
\boldsymbol{\delta}_{i,t}
=\operatorname{vec}_{\mathcal{K}^{s}}
\left(\Delta_{i,t}^{s}\right)\in\mathbb{R}^{d}.
\]

固定参数顺序可以保证不同客户端对相同坐标具有一致解释。随后，客户端以裁剪阈值 $B>0$
限制每个坐标的数值范围：

\[
\bar{\delta}_{i,t,r}
=\max\{-B,\min\{B,\delta_{i,t,r}\}\},
\qquad r=1,2,\ldots,d.
\]

令定点缩放因子为 $q\in\mathbb{N}^{+}$，编码后的整数向量为

\[
z_{i,t,r}=\left\lfloor q\bar{\delta}_{i,t,r}\right\rceil,
\qquad
\boldsymbol{z}_{i,t}\in\mathbb{Z}^{d}.
\]

裁剪用于约束明文幅值和整数加权和的范围，定点缩放则控制数值精度。为避免整数溢出，系统在
加密前验证

\[
\max_{i,r}|z_{i,t,r}|
\sum_{i\in\mathcal{A}_t}w_{i,t}
<2^{63},
\]

其中 $w_{i,t}=n_i$ 为正整数样本权重。高维更新被划分为长度不超过 $L$ 的连续分块，并对每个
分块独立执行加权内积。分块不改变聚合结果，只用于限制中间张量的峰值内存。

定点编码引入的误差可以与裁剪误差分开分析。记理想明文加权平均为
$\Delta^{\mathrm{plain}}_t$，裁剪后的加权平均为 $\Delta^{\mathrm{clip}}_t$，协议恢复结果为
$\widehat{\Delta}^{s}_t$。则

\[
\left\|\widehat{\Delta}^{s}_t-
\Delta^{\mathrm{plain}}_t\right\|_{\infty}
\leq
\left\|\Delta^{\mathrm{clip}}_t-
\Delta^{\mathrm{plain}}_t\right\|_{\infty}
+\frac{1}{2q}.
\]

其中第一项来自坐标裁剪，第二项来自四舍五入。当没有坐标触发裁剪时，每个聚合坐标相对于
明文加权平均的误差不超过 $1/(2q)$。实验中同时记录裁剪比例、最大绝对误差和平均绝对误差，
以检验 $B$ 与 $q$ 的取值是否适合当前模型。

### 3.2.5 DMCFE-IP 加权聚合

本文使用去中心化多客户端功能加密的内积功能，将 FedAvg 所需的样本加权和表示为受限函数
输出。为便于描述，将 DMCFE-IP 方案抽象为 $\mathsf{Setup}$、$\mathsf{KeyGen}$、
$\mathsf{Enc}$、$\mathsf{ShareDKeyGen}$、$\mathsf{Combine}$ 和 $\mathsf{Dec}$ 六个算法。

在系统初始化阶段，可信初始化组件执行

\[
pp\leftarrow\mathsf{Setup}(1^{\lambda},K),
\]

其中 $\lambda$ 为安全参数，$pp$ 为公共参数。各客户端基于公共参数独立生成并保管自己的密钥
材料

\[
sk_i\leftarrow\mathsf{KeyGen}(pp,i).
\]

初始化组件只负责公共参数发布和客户端身份注册，不保存能够单独解密客户端更新的集中式主密钥。

在第 $t$ 轮，服务器公布参与者有序集合 $\mathcal{A}_t$、参数版本和分块编号，并将其编码为
标签 $\tau_{t,b}$。客户端 $C_i$ 对第 $b$ 个整数更新分块
$\boldsymbol{z}^{(b)}_{i,t}$ 执行

\[
\boldsymbol{c}^{(b)}_{i,t}
\leftarrow
\mathsf{Enc}\!\left(
pp,sk_i,\boldsymbol{z}^{(b)}_{i,t},\tau_{t,b}
\right).
\]

标签将密文限制在特定轮次、参与者集合、参数版本和坐标分块中，防止不同轮次或不同参数位置
的密文被直接混用。服务器需要计算的功能由权重向量
$\boldsymbol{w}_t=(w_{i,t})_{i\in\mathcal{A}_t}$ 确定。每个参与客户端针对该权重向量生成
部分功能密钥

\[
dk_{i,t}
\leftarrow
\mathsf{ShareDKeyGen}
\left(sk_i,\boldsymbol{w}_t,\mathcal{A}_t,\tau_t\right).
\]

服务器仅能在收集满足协议要求的部分密钥后执行

\[
dk_{\boldsymbol{w}_t}
\leftarrow
\mathsf{Combine}
\left(\{dk_{i,t}\}_{i\in\mathcal{A}_t}\right).
\]

服务器收集同一标签下的客户端密文，并逐分块执行功能解密：

\[
\boldsymbol{u}^{(b)}_t
=\mathsf{Dec}\!\left(
pp,dk_{\boldsymbol{w}_t},
\{\boldsymbol{c}^{(b)}_{i,t}\}_{i\in\mathcal{A}_t},
\tau_{t,b}
\right)
=\sum_{i\in\mathcal{A}_t}
w_{i,t}\boldsymbol{z}^{(b)}_{i,t}.
\]

将全部分块按原顺序连接后得到 $\boldsymbol{u}_t$。服务器仅对该函数输出进行反量化和归一化：

\[
\widehat{\boldsymbol{\delta}}^{s}_t
=\frac{\boldsymbol{u}_t}
{qW_t},
\qquad
W_t=\sum_{i\in\mathcal{A}_t}w_{i,t},
\]

并通过 $\operatorname{unvec}_{\mathcal{K}^{s}}$ 恢复共享参数字典。由于功能密钥只对应权重向量
$\boldsymbol{w}_t$ 所定义的线性函数，服务器的协议输出被限制为客户端编码更新的加权和。

样本权重不仅决定训练目标，也决定功能密钥对应的函数。为防止错误权重破坏聚合语义，客户端
对消息

\[
m_{i,t}=(t,\mathcal{A}_t,i,w_{i,t})
\]

生成数字签名 $\sigma_{i,t}$。服务器必须在请求或组合功能密钥前验证所有参与者的签名，并检查
$w_{i,t}>0$。轮次索引和完整参与者集合被纳入签名载荷，使旧轮次权重或不同参与集合下的有效
签名不能被直接复用。

### 3.2.6 面向客户端掉线的恢复过程

移动设备或边缘节点可能在一轮协议的不同阶段离线。为区分掉线时机，本文定义三个逐步收缩的
客户端集合：$\mathcal{U}_{2,t}$ 表示完成密钥共享的客户端，$\mathcal{U}_{3,t}$ 表示完成密文
上传的客户端，$\mathcal{U}_{4,t}$ 表示提交恢复响应的客户端，并满足

\[
\mathcal{U}_{4,t}\subseteq
\mathcal{U}_{3,t}\subseteq
\mathcal{U}_{2,t}\subseteq\mathcal{A}_t.
\]

在密钥共享阶段，每个客户端生成临时密钥材料和随机掩码种子，并采用阈值为 $\rho$ 的秘密共享
将恢复材料分发给其他参与者。客户端之间的成对掩码按照客户端标识确定相反符号，因此当双方
均上传密文时，成对掩码在聚合中相互抵消。

若客户端属于 $\mathcal{U}_{2,t}\setminus\mathcal{U}_{3,t}$，则其完成密钥共享但未上传密文，
称为早期掉线。此时，在线客户端密文中仍包含与该客户端对应的单边掩码。服务器从至少 $\rho$
个恢复响应中重构早期掉线客户端的临时密钥材料，计算并移除残留掩码。若客户端属于
$\mathcal{U}_{3,t}\setminus\mathcal{U}_{4,t}$，则其已上传密文但未提交恢复响应，称为晚期掉线。
服务器重构其随机掩码种子，并从部分功能密钥之和中消除对应掩码。

当 $|\mathcal{U}_{4,t}|<\rho$ 时，服务器无法获得足够的恢复份额，当前轮次必须中止。达到阈值
时，恢复后的功能密钥满足

\[
dk^{\mathrm{rec}}_{\boldsymbol{w}_t}
=dk_{\boldsymbol{w}_t},
\]

从而对 $\mathcal{U}_{3,t}$ 中实际上传密文的客户端得到正确的加权和。该过程只恢复完成聚合
所需的掩码或密钥组合，不改变个性化参数始终留在客户端的约束。

### 3.2.7 全局更新与方法流程

服务器获得共享更新后，按照原始参数顺序恢复各共享张量，并执行

\[
\boldsymbol{\theta}^{s}_{t+1}
=\boldsymbol{\theta}^{s}_{t}
+\operatorname{unvec}_{\mathcal{K}^{s}}
\left(\widehat{\boldsymbol{\delta}}^{s}_t\right).
\]

未被选择的客户端继续保存其上一轮个性化参数；被选择的客户端保存本轮更新后的个性化参数。
本文方法的完整流程概括如下。

```text
算法 3-1：基于功能加密的个性化联邦学习
输入：客户端数据集 {D_i}，通信轮数 T，裁剪阈值 B，缩放因子 q，恢复阈值 rho
输出：共享参数 theta_T^s 与各客户端个性化参数 {theta_i^p}

1:  初始化共享参数 theta_0^s、个性化参数 {theta_i,0^p} 和 DMCFE-IP 密钥材料
2:  for t = 0, 1, ..., T-1 do
3:      服务器采样参与集合 A_t 并广播 theta_t^s
4:      客户端签名并提交 (t, A_t, i, w_i,t)，服务器验证权重提案
5:      for each C_i in A_t in parallel do
6:          加载 theta_t^s 和客户端本地 theta_i,t^p
7:          固定 theta_t^s，训练个性化参数 E_p 个 epoch
8:          固定 theta_i,t+1^p，训练共享参数 E_s 个 epoch
9:          计算共享更新 Delta_i,t^s
10:         按 K^s 向量化、裁剪、定点编码并分块
11:         使用轮次与分块标签加密各更新分块
12:     end for
13:     若发生掉线且恢复响应不少于 rho，则恢复聚合所需掩码；否则中止本轮
14:     服务器使用权重功能密钥解密各分块的加权和
15:     反量化、归一化并恢复共享参数结构
16:     更新 theta_t+1^s，客户端保留各自 theta_i,t+1^p
17: end for
```

在功能加密方案满足正确性、签名验证通过且掉线恢复达到阈值的条件下，算法第 14 行输出
$\sum_i w_{i,t}\boldsymbol{z}_{i,t}$，因此第 15 行与对编码后更新执行明文样本加权平均等价。
个性化参数从未作为加密输入或聚合输入，故模型参数划分在整个流程中保持不变。

### 3.2.8 计算与通信复杂度

设第 $t$ 轮参与客户端数为 $m=|\mathcal{A}_t|$，共享更新维度为 $d$，分块长度为 $L$，则
分块数为 $c=\lceil d/L\rceil$。不计本地模型训练开销时，每个客户端完成向量化、裁剪和定点
编码需要 $O(d)$ 次基本运算。明文参考聚合需要 $O(md)$ 次整数乘加；真实功能加密的计算量还
取决于具体方案的群运算、模幂运算和密文打包方式，因此实验中应分别报告密钥生成、加密、功能
密钥组合与解密时间，而不能仅用渐近复杂度代替运行开销。

若每个密文能够打包 $s$ 个更新坐标，客户端模型更新的上行通信量可表示为

\[
\mathcal{B}_{\mathrm{update}}
=m\left\lceil\frac{d}{s}\right\rceil |ct|,
\]

其中 $|ct|$ 表示单个序列化密文的字节数；不进行打包时 $s=1$。采用成对密钥协商和阈值秘密
共享的恢复阶段还会产生 $O(m^2)$ 条密钥共享或恢复控制消息。该部分开销与模型维度无关，但会
随参与客户端数量二次增长。因此，实验评估需要同时报告模型密文载荷和恢复协议载荷。

分块处理将单次加权计算的工作区限制在当前分块。若密文能够按块流式接收，服务器除聚合结果
外的附加工作空间为 $O(mL)$；当前参考实现为便于误差对照，会先物化全部客户端定点向量，其
总体存储仍为 $O(md)$。因此，分块参数 $L$ 在当前实现中主要降低临时堆叠开销，并不等同于完整
的流式密码聚合。

### 3.2.9 实现说明与方法边界

当前原型提供两类后端。`chunked_reference` 用于高维共享状态的快速、完整坐标聚合与定点误差
测量；`toy_mcfe` 用于小规模的加密、功能密钥生成和解密接口正确性检查。完整的
$\mathcal{U}_2/\mathcal{U}_3/\mathcal{U}_4$ 掉线恢复流程在独立协议验证路径中实现。三者共同
验证参数划分、数值编码、线性功能输出、签名权重和阈值恢复等方法组件，但当前尚未组合为经过
安全审计的生产级 DMCFE-IP 后端。

因此，本节关于服务器仅获得授权函数输出的描述，以采用满足相应安全定义的 DMCFE-IP 实例为
前提。现有原型能够支持算法正确性、量化误差、裁剪比例、聚合时间和掉线恢复实验，但不能替代
针对具体密码方案的不可区分性证明、参数安全性分析或串谋安全证明。



