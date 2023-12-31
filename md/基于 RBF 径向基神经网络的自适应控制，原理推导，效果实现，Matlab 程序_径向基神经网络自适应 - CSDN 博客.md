> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_36815313/article/details/127344582?utm_medium=distribuhttps://blog.csdn.net/weixin_36815313/article/details/127344582?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_utm_term~default-1-127344582-blog-129823956.235^v40^pc_relevant_3m_sort_dl_base1&spm=1001.2101.3001.4242.2&utm_relevant_index=4)

<table><thead><tr><th></th><th>径向基</th></tr></thead><tbody><tr><td>1</td><td><a href="https://blog.csdn.net/weixin_36815313/article/details/116176193">径向基 RBF(radial basis function)函数、RBF 神经网络、 反推 (back-stepping) 控制</a></td></tr><tr><td>2</td><td><a href="https://blog.csdn.net/weixin_36815313/article/details/127344582">基于 RBF 径向基神经网络的自适应控制，原理推导，效果实现，Matlab 程序</a></td></tr><tr><td>3</td><td><a href="https://zhaojichao.blog.csdn.net/article/details/127333011" rel="nofollow">使用 RBF 神经网络，结合参考模型，通过神经网络输出被控模型的控制器，实现其对参考模型的跟踪，Matlab 程序</a></td></tr></tbody></table>

#### 文章目录

*   [1. 基于 RBF 网络逼近的自适应控制](#1_RBF_9)
*   *   [1.1 问题描述](#11__13)
    *   [1.2 验证没有未知干扰项的控制器](#12__61)
    *   [1.3 RBF 网络原理](#13_RBF__189)
    *   [1.4 RBF 控制算法设计与分析](#14_RBF__218)
    *   *   [1.4.1 梯度下降法](#141__221)
        *   [1.4.2 稳定性理论设计法](#142__225)
    *   [1.4 验证有未知项和 RBF 的控制器](#14_RBF_264)
*   [2. RBF 神经网络自适应控制 matlab 仿真_RBF 神经网络及其在控制中的应用简介](#2_RBFmatlab_RBF_409)
*   *   [2.1 采用梯度下降法计算权值](#21__415)
    *   [2.2 依据稳定性理论设计权值](#22__417)

1. 基于 [RBF](https://so.csdn.net/so/search?q=RBF&spm=1001.2101.3001.7020) 网络逼近的自适应控制
-----------------------------------------------------------------------------------

### 1.1 问题描述

简单的运动系统动力学方程为：  
θ ¨ = f ( θ , θ ˙ ) + u (1) \ddot{\theta} = f(\theta, \dot{\theta}) + u \tag{1} θ¨=f(θ,θ˙)+u(1)

其中 θ \theta θ 为角度， u u u 为控制输入。

写成状态方程形式为：  
x ˙ 1 = x 2 x ˙ 2 = f (x) + u (2)

$$\begin{aligned} &\dot{x}_1 = x_2 \\ &\dot{x}_2 = f(x) + u \end{aligned}$$

\tag{2}

​x˙1​=x2​x˙2​=f(x)+u​(2)

其中 f (x) f(x) f(x) 为未知非线性函数。

未知指令为 x d x_d xd​，则误差及其变化率为：  
e = x 1 − x d e ˙ = x ˙ 1 − x ˙ d = x 2 − x ˙ d

$$\begin{aligned} e &= x_1 - x_d \\ \dot{e} &= \dot{x}_1 - \dot{x}_d \\ &=x_2 - \dot{x}_d \end{aligned}$$

ee˙​=x1​−xd​=x˙1​−x˙d​=x2​−x˙d​​

定义误差函数为  
s = c e + e ˙ ,     c > 0 (3) s=ce + \dot{e}, ~~~ c>0 \tag{3} s=ce+e˙,   c>0(3)

则  
s ˙ = c e ˙ + e ¨ = c e ˙ + x ˙ 2 − x ¨ d = c e ˙ + f (x) + u − x ¨ d (4)

$$\begin{aligned} \dot{s} &= c\dot{e}+\ddot{e}\\ &=c\dot{e}+\dot{x}_2-\ddot{x}_d \\ &=c\dot{e}+f(x)+u-\ddot{x}_d \end{aligned}$$

\tag{4}

s˙​=ce˙+e¨=ce˙+x˙2​−x¨d​=ce˙+f(x)+u−x¨d​​(4)

由式（3）可知，如果 s → 0 s\rightarrow 0 s→0，则 e → 0 e\rightarrow 0 e→0 且 e ˙ → 0 \dot{e}\rightarrow 0 e˙→0。

若对滑模控制有了解的，可以发现上述误差函数 (3) 的形式与滑模控制中的滑模面类似，可以看一下文章[【控制】滑模控制，滑模面的选择](https://blog.csdn.net/weixin_36815313/article/details/126751337)。这里还有滑模的解决方案，也就是趋近律的选择。

借助趋近律 s ˙ = − η  sgn (s) \dot{s} = -\eta ~\text{sgn}(s) s˙=−η sgn(s)，那么基于上式 (4) 可以得到  
s ˙ = c e ˙ + f (x) + u − x ¨ d = − η  sgn ( s ) u = − c e ˙ − f ( x ) + x ¨ d − η  sgn ( s ) (5)

$$\begin{aligned} \dot{s} &=c\dot{e}+f(x)+u-\ddot{x}_d = -\eta ~\text{sgn}(s) \\ u& = -c\dot{e} - f(x) + \ddot{x}_d -\eta ~\text{sgn}(s) \end{aligned}$$

\tag{5}

s˙u​=ce˙+f(x)+u−x¨d​=−η sgn(s)=−ce˙−f(x)+x¨d​−η sgn(s)​(5)

式 (5) 就是包含了未知非线性函数 f (x) f(x) f(x) 的系统控制器。

但是 f (x) f(x) f(x) 往往是未知的，我们没有一个具体的显式表达式。而这个控制器并不能让系统达到期望输入，根本原因就是 f (x) f(x) f(x) 的存在影响了系统。

### 1.2 验证没有未知干扰项的控制器

不过为了方便理解，我们先验证一下没有未知项 f (x) f(x) f(x) 干扰时的控制器。

也就是将模型式 (2) 简化为  
x ˙ 1 = x 2 x ˙ 2 = u (6)

$$\begin{aligned} &\dot{x}_1 = x_2 \\ &\dot{x}_2 = u \end{aligned}$$

\tag{6}

​x˙1​=x2​x˙2​=u​(6)

将控制器式 (5) 简化为

u = − c e ˙ + x ¨ d − η  sgn (s) (7)

$$\begin{aligned} u& = -c\dot{e} + \ddot{x}_d -\eta ~\text{sgn}(s) \end{aligned}$$

\tag{7}

u​=−ce˙+x¨d​−η sgn(s)​(7)

系统初始状态为 x 1 (0) = 2 x_1(0) = 2 x1​(0)=2， x 2 (0) = 0.2 x_2(0) = 0.2 x2​(0)=0.2。期望状态为 x d = sin ⁡ (t) x_d = \sin(t) xd​=sin(t)， x ˙ d = cos ⁡ (t) \dot{x}_d = \cos(t) x˙d​=cos(t)， x ¨ d = − sin ⁡ (t) \ddot{x}_d = -\sin(t) x¨d​=−sin(t)。参数假设为 c = 1 , η = 0.5 c = 1, \eta = 0.5 c=1,η=0.5。  
虽然我们把未知干扰项简化掉了，但是我们这里还是给出一个 f (x) = 10 x 1 x 2 f(x)=10 x_1 x_2 f(x)=10x1​x2​。画画图，看看效果。

首先是没有未知干扰项的仿真结果，如下图所示。

![](https://img-blog.csdnimg.cn/69b667a1ea104cdbbbb2a95ebd3e8d3a.png#pic_center)

![](https://img-blog.csdnimg.cn/c58bc2e8c08144489ab328494c5db245.png#pic_center)

关于控制器振荡问题的解决办法，参考[【控制】滑模控制，滑模面的选择](https://blog.csdn.net/weixin_36815313/article/details/126751337)。

紧接着给出有 f (x) f(x) f(x) 的仿真结果。

![](https://img-blog.csdnimg.cn/18b064203e0848be9894f556ac705e0a.png#pic_center)

![](https://img-blog.csdnimg.cn/2cd07e0d534e4cacaf78d35e73e2a858.png#pic_center)

最后再把程序给出。

```
% 
% Author: Z-JC
% Data: 2022-10-16
clear
clc

%%
% states
x_1(:,1) = 2;
x_2(:,1) = 0.2;

fx(:,1) = 10 * x_1(:,1) * x_2(:,1);

% Control inputs
u(:,1) = 0;

% Desired 
x_d(:,1) = sin(0);
ddot_x_d(:,1) = cos(0);
dot_x_d(:,1) = -sin(0);

% Parameters
c = 1;
eta = 0.5;

%% Time state
t(1,1) = 0;
tBegin = 0;
tFinal = 20;
dT = 0.05;
times = (tFinal-tBegin)/dT;

% Iterations
for i=1:times
    % Record time
    t(:,i+1) = t(:,i) + dT;
    
    % error
    x_d(:,i+1) = sin(t(:,i+1));
    e = x_1(:,i) - x_d(:,i+1);
    
    dot_x_d(:,i+1) = cos(t(:,i+1));
    dot_e = x_2(:,i) - dot_x_d(:,i+1);
    
    s = c*e + dot_e;
    
    % control input
    ddot_x_d(:,i+1) = -sin(t(:,i+1));
    u(:,i+1) = -c*dot_e + ddot_x_d(:,i+1) - eta * sign(s);
    
    % update states
    x_2(:,i+1) = x_2(:,i) + dT * ( 0*fx(:,i)+u(:,i+1) );
    x_1(:,i+1) = x_1(:,i) + dT * x_2(:,i+1);
    
    % update unknown
    fx(:,i+1) = 10 * x_1(:,i+1) * x_2(:,i+1);
end

%% Plot results
figure(1)
subplot(2,1,1)
plot(t,x_1, t,x_d, 'linewidth',1.5); hold on
legend('$x_{1}$', '$x_{d}$', 'interpreter','latex');
grid on

subplot(2,1,2)
plot(t,x_2, t,dot_x_d, 'linewidth',1.5); hold on
legend('$x_{2}$', '$\dot{x}_{d}$', 'interpreter', 'latex');
grid on

figure(2)
subplot(2,1,1)
plot(t,u, 'linewidth',1.5); hold on
legend('$u$', 'interpreter','latex');
grid on

subplot(2,1,2)
plot(t,fx, 'linewidth',1.5); hold on
legend('$f(x)$', 'interpreter','latex');
grid on

```

### 1.3 RBF 网络原理

对比上述结果也可以看到， f (x) f(x) f(x) 对系统影响特别大。这时候就需要 RBF 来发挥作用了。

由于 RBF 网络具有万能逼近特性，采用 RBF 神经网络逼近 f (x) f(x) f(x)，网络算法为：  
h j = exp ⁡ ( − ∥ x − c j ∥ 2 2 b j 2 ) (8) h_j = \exp(- \frac{\|x-c_j\|^2}{2b^2_j}) \tag{8} hj​=exp(−2bj2​∥x−cj​∥2​)(8)

f = W ∗ T h (x) + ε (9) f = W^{*\text{T}}h(x) + \varepsilon \tag{9} f=W∗Th(x)+ε(9)

其中， x x x 为网络的输入， j j j 为网络隐含层第 j j j 个节点， h = [ h j ] T h=[h_j]^\text{T} h=[hj​]T 为网络的高斯基函数输出， W ∗ W^* W∗ 为网络的理想权值， ε \varepsilon ε 为网络的逼近误差， ε ≤ ε N \varepsilon\le\varepsilon_N ε≤εN​。

这里我们使用系统的状态 x 1 , x 2 x_1, x_2 x1​,x2​ 作为神经网络的输入，那么输入节点就为 2。隐含层节点暂定为 5。输出层因为是对未知函数 f (x) f(x) f(x) 的估计，因此输出节点为 1。

由式 (2)，网络输入取状态变量 x = [ x 1 , x 2 ] T x=[x_1, x_2]^\text{T} x=[x1​,x2​]T，则网络输出为：  
f ^ (x) = W ^ T h ( x ) (10) \hat{f}(x) = \hat{W}^\text{T} h(x) \tag{10} f^​(x)=W^Th(x)(10)

未知函数 f (x) f(x) f(x) 的误差为  
f (x) − f ^ ( x ) = W ∗ T h ( x ) + ε − W ^ T h ( x ) = − W ~ T h ( x ) + ε (11)

$$\begin{aligned} f(x) - \hat{f}(x) &= W^{*\text{T}}h(x) + \varepsilon - \hat{W}^\text{T}h(x)\\ &=-\tilde{W}^\text{T} h(x) + \varepsilon \end{aligned}$$

\tag{11}

f(x)−f^​(x)​=W∗Th(x)+ε−W^Th(x)=−W~Th(x)+ε​(11)

接下来的任务就很清晰了，就是想办法让误差减少，这也等价于优化实际的权重 W ^ \hat{W} W^，让其无限逼近理想权重值 W ∗ W^* W∗。

### 1.4 RBF [控制算法](https://so.csdn.net/so/search?q=%E6%8E%A7%E5%88%B6%E7%AE%97%E6%B3%95&spm=1001.2101.3001.7020)设计与分析

更新 RBF 网络的权重值，这里介绍两种方法，分别是梯度下降法和稳定性理论设计法。

#### 1.4.1 梯度下降法

由于使用梯度下降法来更新权重的资料很多，这里不再赘述。

#### 1.4.2 稳定性理论设计法

接下来介绍基于 Lyapunov 稳定性理论的设计法。

定义 Lyapunov 函数为  
V = 1 2 s 2 + 1 2 γ W ~ T W ~ (12) V = \frac{1}{2}s^2+\frac{1}{2\gamma}\tilde{W}^\text{T} \tilde{W} \tag{12} V=21​s2+2γ1​W~TW~(12)

其中 γ > 0 , W ~ = W ^ − W ∗ \gamma>0, \tilde{W}=\hat{W}-W^* γ>0,W~=W^−W∗。

则  
V ˙ = s s ˙ + 1 γ W ~ T ( W ^ ˙ − W ∗ ˙ ) = s s ˙ + 1 γ W ~ T W ^ ˙ = s ( c e ˙ + f (x) + u − x ¨ d ) + 1 γ W ~ T W ^ ˙ (13)

$$\begin{aligned} \dot{V} &= s\dot{s} + \frac{1}{\gamma}\tilde{W}^\text{T} ( \dot{\hat{W}} - \dot{W^*} ) \\ &= s\dot{s} + \frac{1}{\gamma}\tilde{W}^\text{T} \dot{\hat{W}} \\ &=s(c\dot{e}+f(x)+u-\ddot{x}_d)+\frac{1}{\gamma}\tilde{W}^\text{T} \dot{\hat{W}} \end{aligned}$$

\tag{13}

V˙​=ss˙+γ1​W~T(W^˙−W∗˙)=ss˙+γ1​W~TW^˙=s(ce˙+f(x)+u−x¨d​)+γ1​W~TW^˙​(13)

设计控制律为  
u = − c e ˙ − f ^ (x) + x ¨ d − η  sgn ( s ) (14) u=-c\dot{e} - \hat{f}(x) + \ddot{x}_d - \eta~ \text{sgn}(s) \tag{14} u=−ce˙−f^​(x)+x¨d​−η sgn(s)(14)

则  
V ˙ = s ( f (x) − f ^ ( x ) − η  sgn ( s ) ) + 1 γ W ~ T W ^ ˙ = s ( − W ~ T h ( x ) + ε − η  sgn ( s ) ) + 1 γ W ~ T W ^ ˙ = s ε − s η  sgn ( s ) − s W ~ T h ( x ) + 1 γ W ~ T W ^ ˙ = s ε − s η  sgn ( s ) + W ~ T ( 1 γ W ^ ˙ − s h ( x ) ) (15)

$$\begin{aligned} \dot{V} &= s (f(x) - \hat{f}(x) - \eta~ \text{sgn}(s)) + \frac{1}{\gamma}\tilde{W}^\text{T} \dot{\hat{W}} \\ &= s(-\tilde{W}^\text{T} h(x) + \varepsilon - \eta~\text{sgn}(s)) + \frac{1}{\gamma}\tilde{W}^\text{T} \dot{\hat{W}} \\ &= s \varepsilon - s \eta ~\text{sgn}(s) - s\tilde{W}^\text{T} h(x) + \frac{1}{\gamma}\tilde{W}^\text{T} \dot{\hat{W}} \\ &= s \varepsilon - s \eta ~\text{sgn}(s) + \tilde{W}^\text{T} (\frac{1}{\gamma}\dot{\hat{W}}-s h(x)) \\ \end{aligned}$$

\tag{15}

V˙​=s(f(x)−f^​(x)−η sgn(s))+γ1​W~TW^˙=s(−W~Th(x)+ε−η sgn(s))+γ1​W~TW^˙=sε−sη sgn(s)−sW~Th(x)+γ1​W~TW^˙=sε−sη sgn(s)+W~T(γ1​W^˙−sh(x))​(15)

取自适应律为  
W ^ ˙ = γ s h (x) (16) \dot{\hat{W}} = \gamma s h(x) \tag{16} W^˙=γsh(x)(16)

则  
V ˙ = s ε − s η  sgn (s) + W ~ T ( 1 γ W ^ ˙ − s h ( x ) ) = s ε − η ∣ s ∣ (17)

$$\begin{aligned} \dot{V} &= s \varepsilon - s \eta ~\text{sgn}(s) + \tilde{W}^\text{T} (\frac{1}{\gamma}\dot{\hat{W}}-s h(x)) \\ &= s \varepsilon - \eta |s| \end{aligned}$$

\tag{17}

V˙​=sε−sη sgn(s)+W~T(γ1​W^˙−sh(x))=sε−η∣s∣​(17)

再令 η > ∣ ε ∣ max ⁡ \eta > |\varepsilon|_{\max} η>∣ε∣max​，则有 V ˙ = ε s − η ∣ s ∣ < 0 \dot{V}=\varepsilon s - \eta |s| < 0 V˙=εs−η∣s∣<0。

### 1.4 验证有未知项和 RBF 的控制器

在 3.2 的基础上，我们考虑如下被控对象

x ˙ 1 = x 2 x ˙ 2 = f (x) + u (18)

$$\begin{aligned} &\dot{x}_1 = x_2 \\ &\dot{x}_2 = f(x) + u \end{aligned}$$

\tag{18}

​x˙1​=x2​x˙2​=f(x)+u​(18)

其中 f (x) = 10 x 1 x 2 f(x)=10 x_1 x_2 f(x)=10x1​x2​。

控制律采用式（14），自适应律采用式（16）。RBF 网络采用 2-5-1 的神经网络结构。参数取 γ = 500 , η = 0.50 \gamma=500, \eta=0.50 γ=500,η=0.50。根据网络的输入 x 1 x_1 x1​ 和 x 2 x_2 x2​ 的实际范围，高斯基函数的参数 c i c_i ci​ 和 b i b_i bi​ 的取值分别为 [ − 2 − 1 0 1 2 − 2 − 1 0 1 2 ] \left[

$$\begin{matrix} -2 & -1 & 0 & 1 & 2 \\ -2 & -1 & 0 & 1 & 2 \end{matrix}$$

\right]

[−2−2​−1−1​00​11​22​]

和

3.0 3.0 3.0

。网络权值矩阵中各个元素的初始值取

0.10 0.10 0.10

。

仿真结果如下图所示。

![](https://img-blog.csdnimg.cn/4036214cbaf544e79fa59d9ec33aecd4.png#pic_center)

![](https://img-blog.csdnimg.cn/761027f1f4da4a189357090334adedea.png#pic_center)

谈一点自己再调试程序时的感受。  
一开始各种振荡，而且效果特别不好。然后就是不停的调参数，但是一直不理想。  
![](https://img-blog.csdnimg.cn/5c36ff16eb7346d9ace62b2c96ba5808.png#pic_center)

后来转变了一下思路，因为控制器振荡这个是滑模控制中趋近律的问题，这一点可以通过 3.2 控制器效果看出来，很振荡。  
所以，我紧接着就回退了一步，想着要不先把控制器中的振荡问题解决掉吧，采用的解决方案就是修改符号函数 sgn (s) \text{sgn}(s) sgn(s) 改成饱和函数 sat (s) \text{sat}(s) sat(s)，然后程序就好了结果很完美。  
于是，我就又想着再把饱和函数改回到符号函数，结果效果还是很好。现在想回去之前的振荡效果都回不去了，就很迷茫😂

这个疑惑🤔先留待着吧，把程序放出来

```
% 
% Author: Z-JC
% Data: 2022-10-16
clear
clc

%%
% states
x_1(:,1) = 2;
x_2(:,1) = 0.2;

fx(:,1) = 10 * x_1(:,1) * x_2(:,1);

% Control inputs
u(:,1) = 0;

% Desired 
x_d(:,1) = sin(0);
ddot_x_d(:,1) = cos(0);
dot_x_d(:,1) = -sin(0);

% Parameters
c = 1;
eta = 0.5;

% RBF
InpLayNum = 2;
HidLayNum = 5;
OutLayNum = 1;
gamma = 500;
eta = 0.5;
c_i = [-2 -1  0  1  1
       -2 -1  0  1  1];
b_i = 3;
hat_W = 0.1*ones(HidLayNum, OutLayNum);

%% Time state
t(1,1) = 0;
tBegin = 0;
tFinal = 20;
dT = 0.01;
times = (tFinal-tBegin)/dT;

% Iterations
for i=1:times
    % Record time
    t(:,i+1) = t(:,i) + dT;
    
    % error
    x_d(:,i+1) = sin(t(:,i+1));
    e = x_1(:,i) - x_d(:,i+1);
    
    dot_x_d(:,i+1) = cos(t(:,i+1));
    dot_e = x_2(:,i) - dot_x_d(:,i+1);
    
    s = c*e + dot_e;
    
    % RBF
    % calculate Gaussian basis function
    for j = 1:HidLayNum
        h(j,:) = exp( -norm([x_1(:,i); x_2(:,i)]-c_i(:,j))^2/(2*b_i^2) );
    end
    
    hat_fx(:,i+1) = hat_W' * h;
    
    % adaptive law
    hat_W = hat_W + dT * (gamma * s * h);
    
    % control input
    ddot_x_d(:,i+1) = -sin(t(:,i+1));
    u(:,i+1) = -c*dot_e + ddot_x_d(:,i+1) - eta * sign(s) - hat_fx(:,i+1);
    
    % update states
    x_2(:,i+1) = x_2(:,i) + dT * ( fx(:,i)+u(:,i+1) );
    x_1(:,i+1) = x_1(:,i) + dT * x_2(:,i+1);
    
    % update unknown
    fx(:,i+1) = 10 * x_1(:,i+1) * x_2(:,i+1);
end

%% Plot results
figure(1)
subplot(2,1,1)
plot(t,x_1, t,x_d, 'linewidth',1.5);
legend('$x_{1}$', '$x_{d}$', 'interpreter','latex');
grid on

subplot(2,1,2)
plot(t,x_2, t,dot_x_d, 'linewidth',1.5);
legend('$x_{2}$', '$\dot{x}_{d}$', 'interpreter', 'latex');
grid on

figure(2)
subplot(2,1,1)
plot(t,u, 'linewidth',1.5);
legend('$u$', 'interpreter','latex');
grid on

subplot(2,1,2)
plot(t,fx, t,hat_fx,'k:', 'linewidth',1.5);
legend('$f(x)$', '$\hat{f}(x)$', 'interpreter','latex');
grid on


function out = sat(s)
    out = sign(s) * min(10,abs(s));
end

```

最后，给出来主要参考的文章[一种简单的基于 RBF 网络逼近的自适应控制](https://wenku.baidu.com/view/26eed46ef342336c1eb91a37f111f18583d00cb2.html)。

2. RBF 神经网络[自适应控制](https://so.csdn.net/so/search?q=%E8%87%AA%E9%80%82%E5%BA%94%E6%8E%A7%E5%88%B6&spm=1001.2101.3001.7020) matlab 仿真_RBF 神经网络及其在控制中的应用简介
-------------------------------------------------------------------------------------------------------------------------------------------------------

RBF 神经网络在控制中的应用，可以按其隐含层与输出层连接权值的计算方式分为以下两类：

### 2.1 采用梯度下降法计算权值

### 2.2 依据稳定性理论设计权值

依据稳定性理论设计权值，即通过分析系统的 Lyapunov 稳定性，设计权值，从而保证系统的稳定性和收敛性。

考虑如下二阶非线性系统，以自适应 RBF 控制器的设计为例，对该权值设计方式进行简要介绍。

x ¨ = f ( x , x ˙ ) + g ( x , x ˙ ) u (3) \ddot{x} = f(x, \dot{x}) + g(x, \dot{x}) u \tag{3} x¨=f(x,x˙)+g(x,x˙)u(3)

其中， f f f 为未知非线性函数， g g g 为已知非线性函数； u ∈ R n u\in\mathbb{R}^n u∈Rn 和 y ∈ R n y\in\mathbb{R}^n y∈Rn 分别为系统的控制输入和输出。

令 x 1 = x , x 2 = x ˙ x_1 = x, x_2 = \dot{x} x1​=x,x2​=x˙ 和 y = x 1 y=x_1 y=x1​，（3）式可改写为

x ˙ 1 = x 2 x ˙ 2 = f ( x 1 , x 2 ) + g ( x 1 , x 2 ) u y = x 1

$$\begin{aligned} &\dot{x}_1 = x_2 \\ &\dot{x}_2 = f(x_1,x_2) + g(x_1,x_2) u \\ & y = x_1 \end{aligned}$$

​x˙1​=x2​x˙2​=f(x1​,x2​)+g(x1​,x2​)uy=x1​​

设理想跟踪指令为 y d y_d yd​，则误差为  
e = y d − y = y d − x 1 E = [ e , e ˙ ] T

$$\begin{aligned} e &= y_d - y \\ &= y_d - x_1 \\ E &= [e, \dot{e}]^\text{T} \end{aligned}$$

eE​=yd​−y=yd​−x1​=[e,e˙]T​

设计 K = [ k p , k d ] T K=[k_p, k_d]^\text{T} K=[kp​,kd​]T 使多项式 s 2 + k d s + k p = 0 s^2 + k_d s + k_p = 0 s2+kd​s+kp​=0 的根都在左半复平面。

将 RBF 神经网络的输出代替式（3）中未知函数，可设计控制律为  
u = 1 g (x) [ ] u = \frac{1}{g(x)} [] u=g(x)1​[]

Ref: [rbf 神经网络自适应控制 matlab 仿真_RBF 神经网络及其在控制中的应用简介](https://blog.csdn.net/weixin_31652625/article/details/113315483)