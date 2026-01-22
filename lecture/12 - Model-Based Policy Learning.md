# 1. Open-loop Case

<p align="center">
  <img src="asset/12/planning_openloop.jpg" alt="Model-based RL Version 1.5" width="800" style="vertical-align:middle;"/>
</p>

이전 강의에서 planning 후 하나의 transition만 관찰한 후 다시 planning을 진행(MPC)하는 model-based reinforcement learning version 1.5를 살펴보았다.
이 방법은 optimal control planning 또는 direct optimization 등 다양한 planning 방법을 사용할 수 있지만, 대부분 open loop plan이라는 단점이 있다.

Open loop plan은 관측되지 않은 state를 미리 예상하여 모든 action을 계획하기 때문에 suboptimal이라는 문제가 있다.
* 예를 들어, 2개의 숫자를 더하는 수학 문제가 있고, RL은 이를 2단계로 나눠서 planning한다고 가정하자.
  1. 문제를 풀지 안 풀지 결정한다.
    * 문제 풀 때 맞추면 1000 달러를 얻고, 못 맞추면 2000 달러를 잃는다.
    * 문제를 안 풀면 아무 것도 없다.
  2. 2개의 숫자 더한다.
* Open loop plan은 문제를 푼다고 가정했을 때, 문제를 보지도 않고 답을 제공해야 한다.
* 문제 관찰 없이 random으로 답을 제공해 답을 맞출 확률은 매우 낮기 때문에 optimal planner는 문제를 풀지 않는 결정을 내릴 것이다.

MPC 방법도 이를 해결하지 못한다.
MPC는 매 time step마다 replan을 하지만, 각 planning 시점에서는 open-loop이기 때문에 "나중에 다시 plan할 수 있다"는 사실을 고려하지 못한다.
* 수학 문제 푸는 예시로 확장하면, MPC는 문제를 푼다는 state로 갔을 때 답을 맞출 것이다.
하지만, 첫 step에서 문제를 보지 못한 open-loop 상황이기 때문에 문제를 푼다는 state로 나아가지 않는다.

이는 control 알고리즘 보다 간단하게 만들 수 있는 요인이지만, 가장 큰 단점 중 하나이다.
해결을 위해선 action의 sequence $a_1, \cdots, a_T$ 대신 policy $\pi(a_t|s_t)$를 출력하는 closed-loop case 방법을 활용해야 한다.

# 2. Closed-loop Case

<p align="center">
  <img src="asset/12/planning_closedloop.jpg" alt="The stochastic closed-loop case" width="800" style="vertical-align:middle;"/>
</p>

Closed-loop case에서는 state를 관찰하면, 그에 적절한 policy를 전달한다.
Closed-loop의 objective는 model-free RL problem과 동일하고, 차이점은 $p(s_{t+1}|s_t, a_t)$를 명시적으로 모델링한다는 것이다.
* Model-free RL은 경험을 통해 수집한 data $s_t, a_t, r_t, s_{t+1}$을 policy 학습에 사용하는 거였지만, model-based RL은 data를 수집하지 않아도 이미 알고 있는 model을 활용해 policy 학습을 진행한다.

Closed-loop contorl 알고리즘은 관찰 가능한 모든 state에 대해 올바들 답을 제공하는 policy를 학습할 수 있기 때문에 open-loop의 단점을 해결할 수 있다.
* 쉬운 수학 시험을 안전하게 치를 수 있다는 것을 인식하고 올바른 action을 취할 것이다.

Model-free RL에서는 neural network같은 high capacity로 policy를 표현해 모든 state에서 괜찮은 action을 생성하는 global policy를 제공하였다.
Model-based RL에서는 lecture 10에서 살펴본 것처럼 보다 간단한 time-varying linear policy인 iLQR로 local policy(linear feedback controller)를 제공할 수 있다.

이번 강의에서는 학습된 dynamics model을 활용해 neural network 같은 global policy를 학습하는 데 집중할 것이다.

# 3. Backpropagation Into The Policy

<p align="center">
  <img src="asset/12/backpropagation1.jpg" alt="Backpropagate directlyy into the policy" width="800" style="vertical-align:middle;"/>
</p>

Objective는 reward의 총 합을 늘리는 것이다.
이를 computaiton graph로 나타내면 위와 같고 3개의 function(policy, dynamics model, reward)이 존재한다.
* Reward function $r$은 알고있고 미분 가능하고, dynamics model $f$와 policy $\pi$는 학습 가능하다고 가정한다.
* Stochastic의 경우 re-parameterization trick을 활용해 학습 가능하지만, 지금은 deterministic한 경우만 고려한다.

각 reward node에서 backpropagation을 진행해 $\pi_\theta$를 최적화할 것이다.
이를 model-based reinforcement learning version 2.0이라고 부르자.

<p align="center">
  <img src="asset/12/backpropagation2.jpg" alt="Backpropagate directlyy into the policy" width="800" style="vertical-align:middle;"/>
</p>

안타깝게도 version 2.0은 일반적으로 잘 작동하지 않는다.
* Lecture 10에서 살펴본 shooting 방법을 살펴봤을 때와 마찬가지로, trajectory 초반의 actions은 미래에 훨씬 더 큰 영향을 끼친다. 
  * 이런 관점으로 초반 actions는 큰 gradient를 가지고 마지막 actions는 작은 gradient를 가지게 된다.
  * 다른 분야에서 DNN model이 겪는 gradient exploding/vanishing 문제이다.
  * 일부 parameter는 큰 graidnet를 받고 일부 parameter는 작은 gradient를 받는, 수치 최적화 관점에서 ill-conditioned 상황이다.
* 초반 action에 대한 작은 변화로 나머지 trajectories가 많이 바뀌게 되는데, 이는 lecture 10 shooting 방법과 trajectory optimization에서 겪는 민감도와 유사하다.
다만, LQR와 같이 2-order Taylor expansion을 활용할 수 없다.
  * LQR은 cost fucntion에서 2-order Taylor expansion으로 Hessian의 근사치를 구하고 최적화를 진행한다.
  * Hessian은 curvature(곡률)까지 고려해서 최적화를 진행한다. 
  * 즉, reward 총 합계에 민감하게 영향을 주는 초반 actions는 curvature가 크고, 이를 반영해 최적화를 진행하기 때문에 gradient exploding이 완화된다.
  * Gradient vanishing도 마찬가지이다.
* Policy 최적화는 모든 time step을 함께 결합시키기 때문에 끝에서 시작해서 거꾸로 작업할 수 있는 편리한 optimization을 진행할 수 없다.
  * LQR은 현재 time step $t$ 해당하는 $a_t$의 Hessian 값만 계산한다.
    * $\partial J / \partial a_t$
    * 모든 time step 관점에서 Hessian의 대각 element $H_{ii}$만 계산한다.
  * Policy 최적화는 모든 time step $t$에 대한 Hessian 값을 계산하기 때문에 시간이 오래 걸린다.
    * $\partial J / \partial \{a_1, \cdots, a_T\}$
    * 모든 time step 관점에서 Hessian의 모든 element $H_{ij}$를 계산한다.
* 또한 DNN model에서 2-order optimization은 잘 동작하지 않는다.
  * Non-convexity 특성(saddle point, 여러 개의 local minima) 때문에 suboptimal이 될 수 있다.
  * Lecture 10강 마지막에서 살펴봤다시피 curvature의 급격한 변화가 있으면 수렴이 쉽지 않다.
  * ... 등등

이러한 현상은 기본적인 RNN의 BPTT(Back Propagation Through Time)에서 발생하는 gradient vanishing/exploding 문제와 유사하다.
* Pure RNN에서는 jacobian의 값이 계속 곱해지는 구조이기 때문에 jacobian의 값이 1보다 크면 gradient exploding이 발생하고 1보다 작으면 gradient vanishing이 발생한다.

RNN에서의 해결책을 model-based에 적용할 수 있지 않을까라는 생각을 가질 수 있는데 결론적으로 불가능하다.
* RNN에서는 dynamics model 구조(GRU, LSTM, transformer, ...)를 선택하여 문제를 해결한다.
  * LSTM에서는 역전파를 통한 시간에 참여하는 함수들의 형태를 선택할 수 있어 jacobian이 거의 1에 가깝도록 강제한다.
* 하지만, model-based에서 dynamics model($f(s_t, a_t)$)는 실제 environment에 근사하도록 학습한 것으로 직접 선택할 수 없다.
  * 로봇을 학습시킨다고 했을 때 로봇에 적용되는 물리법칙은 바꿀 수 없다.
  * 실제 물리 법칙이 높은 curvature를 가질 수 있고 jacobian이 1에서 매우 먼 값을 가질 수 있다.

이러한 이유로 model-based RL에서 gradient 기반 policy 최적화는 매우 까다로운 주제이다.

<p align="center">
  <img src="asset/12/backpropagation3.jpg" alt="Backpropagate directlyy into the policy" width="800" style="vertical-align:middle;"/>
</p>

이번 강의에서 논의할 해결책은 학습한 dynamics model을 synthetic sample을 생성하는 데만 활용하고, model-free RL 알고리즘으로 policy를 학습하는 것이다.
* Policy 학습 시 dynamics model을 simulator로만 활용하며, dynamics model을 통한 backpropagation은 수행하지 않는다.
* Simulation으로 생성된 많은 data를 model-free RL을 가속화하기 위해 사용한다.

기본 model-free RL 방법은 이전에 살펴 본 것과 거의 동일할 것(policy gradient, actor-critic, Q-learning)이다.
하지만 학습된 dynamics model에 의해 생성된 추가 데이터를 활용한다.
이는 역설적으로 보일 수 있지만, 매우 잘 작동한다.
* 실제로는 model-free RL과 planning 중간인 어느 정도 hybrid 방법론으로 볼 수 있다.
