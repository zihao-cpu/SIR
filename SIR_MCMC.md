# SIR_MCMC

## Metropolis-Hastings算法

```matlab
for iter = 1:nIter
    
    %Blocco X
    [tSim, XSim] = ode45('model',tspan,X0,'',X); %simulo il modello con i parametri X
    
    %Calcolo di pi(X) (nel logaritmo per efficienza numerica)
    lX = -(N/2)*log(2*pi)-(N/2)*log((SDw^2))-0.5*sum(((XSim(:,2)-y)/SDw).^2); %calcolo log-likelihood
    piX = lX + log(priorB(X(1))) + log(priorK(X(2))); %calcolo pi(X) 
    
    %Blocco Y
    Y = X + randn(length(X),1).*stdProp;
    [tSim, XSim] = ode45('model',tspan,X0,'',Y); %simulo il modello con i parametri Y
    
    %Calcolo di pi(Y) (nel logaritmo per efficienza numerica)
    lY = -(N/2)*log(2*pi)-(N/2)*log((SDw^2))-0.5*sum(((XSim(:,2)-y)/SDw).^2); %calcolo log-likelihood
    piY = lY + log(priorB(Y(1))) + log(priorK(Y(2))); %calcolo pi(Y) 
    
    % Decido se accetto o rifiuto il nuovo candidato
    U = rand(1);
    alfa = min(1,exp(piY-piX));
    if(U<=alfa && ~isnan(exp(piY-piX)))
        X = Y;
        accept = accept + 1;
    end %if
    
    %Salvo la catena
    bHat(iter) = X(1);
    kHat(iter) = X(2);
    
    %Visualizzo come procede (SOLO DEBUG)
    if(debug)
        subplot(211)
        title(['Iterazione ' num2str(iter) ' di ' num2str(nIter)]);
        hold on
        scatter(iter,bHat(iter),'MarkerEdgeColor',[0 0 0]);
        subplot(212)
        hold on
        scatter(iter,kHat(iter),'MarkerEdgeColor',[0 0 0]);
        pause(0.001)
    end
    % ---------------------------------------------------------------------
    
end
```

1.初始化：

```matlab
X = theta0; % setto X alla proposta iniziale
accept = 0; %conto il numero di volte che accetto
```

首先，初始化了参数 $$X$$，即 $$b$$ 和 $$k$$ 的初始值。`accept` 变量用于记录接受新样本的次数。

2.模拟当前参属下的系统行为：

```matlab
[tSim, XSim] = ode45('model', tspan, X0, '', X); 
```

对于每次迭代，使用当前的参数 $$X$$（即当前的感染率 $$b$$ 和康复率 $$k$$）来模拟系统的行为。`ode45` 是 MATLAB 的数值解算器，它使用 **Runge-Kutta 方法** 求解微分方程，并生成时间步长 $$tSim$$ 和系统状态 $$XSim$$，其中 `XSim(:,2)` 是表示感染状态的列向量。

3.计算当前参数下的似然函数和后验分布：

假设观测值 $$y$$ 和预测值 $$\hat{y}$$之间的差异遵循正态分布。对于每个时间点 $$t$$，我们可以用一个正态分布来表示感染者的数量。正态分布的概率密度函数形式为$$f(y|\mu,\sigma^2)=\frac{1}{\sqrt {2\pi\sigma^2}}exp(-\frac{(y-\mu)^2}{2\sigma^2})$$

3.1对数似然函数$$log(f(y_i|\mu,\sigma^2))=-\frac{1}{2}log(2\pi)-\frac{1}{2}log(\sigma^2)-\frac{(y_i-\mu)^2}{2\sigma^2}$$

3.2所有观测值的对数似然函数：

假设我们有 $$N$$ 个观测值 $$y1,y2,…,yN$$，我们可以通过将每个观测值的对数似然函数相加，得到整体的对数似然函数：$$\log L = \sum_{i=1}^{N} \log(f(y_i \mid \mu, \sigma^2)) = -\frac{N}{2} \log(2\pi) - \frac{N}{2} \log(\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{N} (y_i - \mu)^2$$

$$p(\theta|x)\propto p(x|\theta)p(\theta)$$

```matlab
lX = -(N/2)*log(2*pi)-(N/2)*log((SDw^2))-0.5*sum(((XSim(:,2)-y)/SDw).^2);%似然
piX = lX + log(priorB(X(1))) + log(priorK(X(2)));后验正比于似然*先验

```

4.提出候选参数：

```matlab
Y = X + randn(length(X),1).*stdProp;
```

接下来，提出一个新的候选参数 $$Y$$，这个候选参数是通过在当前参数 $$X$$ 上加入高斯噪声（服从均值为零、方差为 `stdProp` 的正态分布）生成的。这样，我们就生成了一个新的参数集合 $$Y$$，用来作为下一步的候选值。

5.模拟新候选参数下的系统行为：

```matlab
[tSim, XSim] = ode45('model', tspan, X0, '', Y);
```

使用新候选参数 $$Y$$ 来模拟系统行为，得到新的模拟结果 `XSim`。

6.计算新参数的似然函数和后验分布：

```matlab
lY = -(N/2)*log(2*pi)-(N/2)*log((SDw^2))-0.5*sum(((XSim(:,2)-y)/SDw).^2);
piY = lY + log(priorB(Y(1))) + log(priorK(Y(2)));
```

7.拒绝或接受新候选的参数：

```matlab
U = rand(1);
alfa = min(1,exp(piY - piX));
if(U <= alfa && ~isnan(exp(piY - piX)))
    X = Y;
    accept = accept + 1;
end
```

在这里，我们应用 **Metropolis-Hastings 接受准则** 来决定是否接受新候选参数 $$Y$$。

$$\alpha = \min(1, \exp(\pi_Y - \pi_X))$$ 是接受概率，如果 $\pi_Y$$ 大于 $$\pi_X$，则接受概率为 1；否则，它是一个小于 1 的值，表示有可能拒绝候选参数 $$Y$$。

如果生成的随机数 $$U$$ 小于等于接受概率 $$\alpha$$，则接受新的候选参数 $$Y$$，并将 $$X = Y$$。否则，保持当前参数不变。