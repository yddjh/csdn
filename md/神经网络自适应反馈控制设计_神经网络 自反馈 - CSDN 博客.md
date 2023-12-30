> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/u010038790/article/details/114004873?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170391868716800180613476%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170391868716800180613476&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-4-114004873-null-null.142^v99^pc_search_result_base9&utm_term=%E8%87%AA%E9%80%82%E5%BA%94%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%8E%A7%E5%88%B6&spm=1018.2226.3001.4187)

#### 文章目录

*   [写在前面](#_2)
*   [问题描述](#_8)
*   [自适应神经网络控制器](#_23)
*   *   [RBF 神经网络](#RBF_31)
    *   [神经网络训练](#_60)
    *   [自适应律之一](#_86)
    *   [自适应律之二](#_98)

写在前面
----

自适应神经网络控制器将自适应控制与神经网络相结合，通过神经网络系统辨识的方法达到自适应控制的目的。

控制器针对系统模型参数未知的被控对象，基于系统的状态变量，通过神经网络辨识被控对象的未知参数，并将被控对象的模型信息提供给自适应反馈控制器。相对于传统的 [PID 控制](https://so.csdn.net/so/search?q=PID%E6%8E%A7%E5%88%B6&spm=1001.2101.3001.7020)器，自适应神经网络控制器有更强的鲁棒性。

问题描述
----

考虑一个含有未知函数的非线性系统，其状态方程可表示为  
{ x ˙ = f (x) + P u y = x ( 1 ) \left\{

$$\begin{aligned} \dot x&=f(x)+Pu\\ y&=x \end{aligned}$$

\right.\qquad (1)

{x˙y​=f(x)+Pu=x​(1)

其中

x , y ∈ R m x,y\in\mathbb R^m x,y∈Rm

为系统状态变量和输出变量，

u ∈ R m u\in\mathbb R^m u∈Rm

为系统的输入变量，

f (x) f(x) f(x)

为未知函数，

P ∈ R m × m P\in\mathbb R^{m\times m} P∈Rm×m

为未知正定常矩阵。

控制目标是使系统输出变量 y y y 跟踪参考轨迹 y d y_d yd​。

自适应神经网络控制器
----------

神经网络反馈自适应控制器的结构如下图所示 [1](#fn1)。

![](https://star2dust.github.io/post-images/1613827807560.PNG)

### RBF 神经网络

首先定义**跟踪误差** e e e  
e = y − y d (2) e=y-y_d\qquad (2) e=y−yd​(2)  
对 (2) 求导，得  
e ˙ = P ( P − 1 ( f (x) − y ˙ d ) + u ) ( 3 ) \dot e = P\left(P^{-1}(f(x)-\dot y_d)+u\right)\qquad (3) e˙=P(P−1(f(x)−y˙​d​)+u)(3)  
考虑 P − 1 ( f (x) − y ˙ d ) P^{-1}(f(x)-\dot y_d) P−1(f(x)−y˙​d​) 未知，需要辨识该部分模型。令 Z = [ x , y ˙ d ] Z=[x,\dot y_d] Z=[x,y˙​d​]，采用 **RBF 神经网络** (Radial basis function neural networks)逼近 (1) 中的未知函数：  
W ∗ T S (Z) + δ ( Z ) = P − 1 ( f ( x ) − y ˙ d ) ( 4 ) W^{*T}S(Z)+\delta(Z)=P^{-1}(f(x)-\dot y_d)\qquad (4) W∗TS(Z)+δ(Z)=P−1(f(x)−y˙​d​)(4)  
其中， S ∈ R m S\in\mathbb R^m S∈Rm 是高斯基函数， W ∗ ∈ R m × m W^*\in\mathbb R^{m\times m} W∗∈Rm×m 是理想的常数权值 ( ∥ W ∥ F ≤ W m \|W\|_F\leq W_{m} ∥W∥F​≤Wm​)， δ ∈ R m \delta\in\mathbb R^m δ∈Rm 是逼近误差 ( ∥ δ ∥ ≤ δ m \|\delta\|\leq \delta_m ∥δ∥≤δm​)。通常认为存在理想的权值使得逼近误差 δ \delta δ最小。

定义 W ∗ W^* W∗的估计值为 W ^ \hat W W^，**权值估计误差** W ~ = W ^ − W ∗ \tilde W=\hat W-W^* W~=W^−W∗。将 (4) 代入 (3) 得  
e ˙ = P ( W ∗ T S (Z) + δ ( Z ) + u ) ( 5 ) \dot e=P(W^{*T}S(Z)+\delta(Z)+u)\qquad (5) e˙=P(W∗TS(Z)+δ(Z)+u)(5)  
构造控制器为  
u = − k e − W ^ T S (Z) ( 6 ) u=-ke-\hat W^TS(Z)\qquad (6) u=−ke−W^TS(Z)(6)  
将 (6) 代入 (5) 得  
e ˙ = P [ − k e − W ~ T S (Z) + δ ( Z ) ] ( 7 ) \dot e=P[-ke-\tilde W^TS(Z)+\delta(Z)]\qquad (7) e˙=P[−ke−W~TS(Z)+δ(Z)](7)

### 神经网络训练

为了使权值 W ^ \hat W W^ 逼近理想权值 W ∗ W^* W∗，需要对 W ^ \hat W W^ 进行训练。同样为了确保控制系统全局稳定，基于李雅普诺夫稳定性判据计算 W ^ \hat W W^ 的自适应更新律。

将跟踪误差和权值估计误差作为李雅普诺夫函数的自变量，构建函数如下  
V = 1 2 e T P − 1 e + 1 2 tr ⁡ ( W ~ T Γ − 1 W ~ ) (8) V=\frac{1}{2}e^TP^{-1}e+\frac{1}{2}\operatorname{tr}(\tilde W^T\Gamma^{-1}\tilde W)\qquad (8) V=21​eTP−1e+21​tr(W~TΓ−1W~)(8)  
其中 Γ \Gamma Γ为正定对称矩阵。对 (8) 求时间导数可得  
V ˙ = − k e T e − S T (Z) W ~ e + tr ⁡ ( W ~ T Γ − 1 W ~ ˙ ) + δ ( Z ) e \dot V=-ke^Te-S^T(Z)\tilde We+\operatorname{tr}(\tilde W^T\Gamma^{-1}\dot{\tilde W})+\delta(Z)e V˙=−keTe−ST(Z)W~e+tr(W~TΓ−1W~˙)+δ(Z)e  
由于 W ~ = W ^ − W ∗ \tilde W=\hat W-W^* W~=W^−W∗，有 W ^ ˙ = W ~ ˙ \dot {\hat W}=\dot {\tilde W} W^˙=W~˙，从而有  
V ˙ = − k e T e − S T (Z) W ~ e + tr ⁡ ( W ~ T Γ − 1 W ^ ˙ ) + δ ( Z ) e \dot V=-ke^Te-S^T(Z)\tilde We+\operatorname{tr}(\tilde W^T\Gamma^{-1}\dot {\hat W})+\delta(Z)e V˙=−keTe−ST(Z)W~e+tr(W~TΓ−1W^˙)+δ(Z)e  
由于  
S (Z) W ~ e = tr ⁡ ( e S T ( Z ) W ~ ) S(Z)\tilde We=\operatorname{tr}(eS^T(Z)\tilde W) S(Z)W~e=tr(eST(Z)W~)

则  
V ˙ = − k e T e + tr ⁡ ( − e S T (Z) W ~ + W ^ ˙ T Γ − 1 W ~ ) + δ ( Z ) e ( 9 ) \dot V=-ke^Te+\operatorname{tr}(-eS^T(Z)\tilde W+\dot {\hat W}^T\Gamma^{-1}\tilde W)+\delta(Z)e\qquad (9) V˙=−keTe+tr(−eST(Z)W~+W^˙TΓ−1W~)+δ(Z)e(9)

### 自适应律之一

稳定性要求 V ˙ ≤ 0 \dot V\leq 0 V˙≤0，考虑 − S T (Z) W ~ e -S^T(Z)\tilde We −ST(Z)W~e 和 tr ⁡ ( W ~ T Γ − 1 W ^ ˙ ) \operatorname{tr}(\tilde W^T\Gamma^{-1}\dot {\hat W}) tr(W~TΓ−1W^˙) 相互抵消时，通过选择系数 k k k 满足 k ≥ δ m ∥ e ∥ k\geq\frac{\delta_m}{\|e\|} k≥∥e∥δm​​，使得下式成立：  
V ˙ = − k e T e + δ (Z) e ≤ − k ∥ e ∥ 2 + δ ∗ ∥ e ∥ = − ( k − δ ∗ ∥ e ∥ ) ∥ e ∥ 2 ≤ 0. \dot V=-ke^Te+\delta(Z)e\leq -k\|e\|^2+\delta^*\|e\|=-(k-\frac{\delta^*}{\|e\|})\|e\|^2\leq 0. V˙=−keTe+δ(Z)e≤−k∥e∥2+δ∗∥e∥=−(k−∥e∥δ∗​)∥e∥2≤0.  
此时神经网络的权值自适应更新律为  
W ^ ˙ = Γ S (Z) e ( 10 ) \dot {\hat W}=\Gamma S(Z)e\qquad (10) W^˙=ΓS(Z)e(10)  
相应 ∥ e ∥ \|e\| ∥e∥的收敛半径为 δ m / k \delta_m/k δm​/k。但该方法不能保证权值 W ~ = W ^ − W ∗ \tilde W=\hat W-W^* W~=W^−W∗的有界性，即无法实现**未知上界有界** (Unknown Upper Bound, UUB) 问题。

### 自适应律之二

取自适应律 [2](#fn2)[3](#fn3) 为  
W ^ ˙ = Γ S (Z) e − k 1 Γ W ^ ( 11 ) \dot {\hat W}=\Gamma S(Z)e-k_1\Gamma\hat W\qquad (11) W^˙=ΓS(Z)e−k1​ΓW^(11)  
将 (11) 代入 (9) 得  
V ˙ = − k e T e − k 1 tr ⁡ ( W ^ T W ~ ) + δ (Z) e \dot V=-ke^Te-k_1\operatorname{tr}(\hat W^T\tilde W)+\delta(Z)e V˙=−keTe−k1​tr(W^TW~)+δ(Z)e

为了比较真实的矩阵和估计的矩阵值之间的误差，或者说比较真实矩阵和估计矩阵之间的相似性，我们可以采用 **Frobenius 范数**。根据 F 范数的性质 [4](#fn4)[5](#fn5)，有  
2 tr ⁡ ( W ^ T W ~ ) = ∥ W ^ ∥ F 2 + ∥ W ~ ∥ F 2 − ∥ W ∗ ∥ F 2 ≥ ∥ W ~ ∥ F 2 − W m 2 tr ⁡ ( W ^ T W ~ ) ≥ ∥ W ~ ∥ F 2 − ∥ W ~ ∥ ∥ W ∗ ∥ F ≥ ∥ W ~ ∥ F 2 − ∥ W ~ ∥ F W m

$$\begin{aligned} 2\operatorname{tr}(\hat W^T\tilde W)&= \|\hat W\|_F^2+\|\tilde W\|_F^2-\|W^*\|_F^2\geq \|\tilde W\|_F^2-W_m^2\\ \operatorname{tr}(\hat W^T\tilde W)&\geq \|\tilde W\|_F^2-\|\tilde W\|\|W^*\|_F\geq \|\tilde W\|_F^2-\|\tilde W\|_FW_m \end{aligned}$$

2tr(W^TW~)tr(W^TW~)​=∥W^∥F2​+∥W~∥F2​−∥W∗∥F2​≥∥W~∥F2​−Wm2​≥∥W~∥F2​−∥W~∥∥W∗∥F​≥∥W~∥F2​−∥W~∥F​Wm​​

则

V ˙ ≤ − k e T e − k 1 ( ∥ W ~ ∥ F 2 − ∥ W ~ ∥ F W m ) + δ (Z) e ≤ − [ ∥ e ∥ ∥ W ~ ∥ F ] [ k 0 0 k 1 ] [ ∥ e ∥ ∥ W ~ ∥ F ] + [ δ m k 1 W m ] [ ∥ e ∥ ∥ W ~ ∥ F ] ( 12 ) 

$$\begin{aligned} \dot V&\leq -ke^Te-k_1(\|\tilde W\|_F^2-\|\tilde W\|_FW_m)+\delta(Z)e\\ &\leq-\begin{bmatrix}\|e\|&\|\tilde W\|_F\end{bmatrix}\begin{bmatrix}k&0\\ 0&k_1\end{bmatrix}\begin{bmatrix}\|e\|\\ \|\tilde W\|_F\end{bmatrix} +\begin{bmatrix}\delta_m&k_1W_m\end{bmatrix}\begin{bmatrix}\|e\|\\ \|\tilde W\|_F\end{bmatrix} \end{aligned}$$

\qquad (12) V˙​≤−keTe−k1​(∥W~∥F2​−∥W~∥F​Wm​)+δ(Z)e≤−[∥e∥​∥W~∥F​​][k0​0k1​​][∥e∥∥W~∥F​​]+[δm​​k1​Wm​​][∥e∥∥W~∥F​​]​(12)

令

z = [ ∥ e ∥ ∥ W ~ ∥ F ] T z=

$$\begin{bmatrix}\|e\|&\|\tilde W\|_F\end{bmatrix}$$

^T z=[∥e∥​∥W~∥F​​]T

。则 (12) 可以表示为

V ˙ ≤ − z T Q z + h z ≤ − λ ‾ (Q) ∥ z ∥ 2 + ∥ h ∥ ∥ z ∥ ( 13 ) \dot V\leq -z^TQz+hz\leq -\underline \lambda(Q)\|z\|^2+\|h\|\|z\|\qquad (13) V˙≤−zTQz+hz≤−λ​(Q)∥z∥2+∥h∥∥z∥(13)

令

{ R ‾ = min ⁡ ( λ ‾ ( P − 1 ) , λ ‾ ( Γ − 1 ) ) R ‾ = max ⁡ ( λ ‾ ( P − 1 ) , λ ‾ ( Γ − 1 ) ) \left\{

$$\begin{aligned} \underline R&=\min(\underline \lambda(P^{-1}),\underline \lambda(\Gamma^{-1}))\\ \overline R&=\max(\overline \lambda(P^{-1}),\overline \lambda(\Gamma^{-1})) \end{aligned}$$

\right. {R​R​=min(λ​(P−1),λ​(Γ−1))=max(λ(P−1),λ(Γ−1))​

则

1 2 R ‾ ∥ z ∥ 2 ≤ V ≤ 1 2 R ˉ ∥ z ∥ 2 (14) \frac{1}{2}\underline R\|z\|^2\leq V\leq \frac{1}{2}\bar R\|z\|^2\qquad (14) 21​R​∥z∥2≤V≤21​Rˉ∥z∥2(14)

结合 (13)、(14) 得到

V ˙ ≤ − α V + β V (15) \dot V\leq -\alpha V+\beta \sqrt{V}\qquad (15) V˙≤−αV+βV ​(15)

其中

α = 2 λ ‾ (Q) R ‾ , β = 2 ∥ h ∥ R ‾ \alpha = \frac{2\underline \lambda(Q)}{\overline R},\quad \beta=\frac{\sqrt{2\|h\|}}{\sqrt{\underline R}} α=R2λ​(Q)​,β=R​ ​2∥h∥ ​​

对 (15) 积分可得

V (t) ≤ V ( 0 ) e − α 2 t + β α ( 1 − e − α 2 t ) \sqrt{V(t)}\leq \sqrt{V(0)}e^{-\frac{\alpha}{2}t}+\frac{\beta}{\alpha}(1-e^{-\frac{\alpha}{2}t}) V(t) ​≤V(0) ​e−2α​t+αβ​(1−e−2α​t)

可保证

∥ e ∥ , ∥ W ~ ∥ F \|e\|,\|\tilde W\|_F ∥e∥,∥W~∥F​

均最终一致有界

[6](#fn6)

，且

lim ⁡ t → ∞ ∥ e ∥ ≤ ∥ h ∥ R ‾ λ ‾ (Q) R ‾ λ ‾ ( P − 1 ) \lim_{t\to\infty} \|e\|\leq \frac{\|h\|\overline R}{\underline \lambda(Q)\sqrt{\underline R \underline \lambda (P^{-1})}} t→∞lim​∥e∥≤λ​(Q)R​λ​(P−1) ​∥h∥R​

相比传统的 PID 控制器， 自适应控制器能修正自己的特性以适应对象和扰动的动态特性的变化 ， 因此对模型复杂多变被控对象，自适应神经网络控制算法具有更高的可靠性。

1.  神经网络控制 - 杨辰光 - HuangTL 的文章 - 知乎 https://zhuanlan.zhihu.com/p/346601192 [↩︎](#fnref1)
    
2.  刘金琨. 机器人控制系统的设计与 MATLAB 仿真. 北京：清华大学出版社，2008. [↩︎](#fnref2)
    
3.  Z. Peng, D. Wang, H. Zhang, and G. Sun, “Distributed neural network control for adaptive synchronization of uncertain dynamical multiagent systems,” IEEE Trans. Neural Networks Learn. Syst., vol. 25, no. 8, pp. 1508–1519, 2014. [↩︎](#fnref3)
    
4.  M. Chen, S. S. Ge, and B. V. E. How, “Robust adaptive neural network control for a class of uncertain MIMO nonlinear systems with input nonlinearities,” IEEE Trans. Neural Networks, vol. 21, no. 5, pp. 796–812, May 2010. [↩︎](#fnref4)
    
5.  StackExchange. Relation between Frobenius norm and trace. https://math.stackexchange.com/questions/1898839/relation-between-frobenius-norm-and-trace [↩︎](#fnref5)
    
6.  H. Khalil, Nonlinear Systems, 3rd ed. Englewood Cliffs, NJ, USA: Prentice-Hall, 2002 [↩︎](#fnref6)