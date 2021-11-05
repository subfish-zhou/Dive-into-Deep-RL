REINFORCE算法
==============

REINFORCE算法是由Ronald J. Williams在1992年的论文《联结主义强化学习的简单统计梯度跟踪算法》（Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning）中提出的基于策略的算法。

REINFORCE算法的想法很自然：在学习过程中，产生高收益的行动应该多做些，导致坏结果的行动应该少做点。我们希望成功的学习结果会让策略函数收敛到能在环境中表现最好的情况。策略函数中动作概率的更新是由策略梯度实现的，因此REINFORCE被称为策略梯度算法。

通过这个想法，我们即将设计的算法会有三个主要组件：
1. 一个策略函数，里面有一些参数，这些参数在学习过程中将要被更新；
2. 一个要最大化的目标，能给策略提供即时评价；
3. 一个参数更新方法。

```Python
from torch.distributions import Categorical
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
gamma = 0.99

class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train() # set training mode

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        pdparam = self.model(x)
        return pdparam

    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32)) # to tensor
        pdparam = self.forward(x) # forward pass
        pd = Categorical(logits=pdparam) # probability distribution
        action = pd.sample() # pi(a|s) in action via pd
        log_prob = pd.log_prob(action) # log_prob of pi(a|s)
        self.log_probs.append(log_prob) # store for training
        return action.item()

    def train(pi, optimizer):
        # Inner gradient-ascent loop of REINFORCE algorithm
        T = len(pi.rewards)
        rets = np.empty(T, dtype=np.float32) # the returns
        future_ret = 0.0
        # compute the returns efficiently
        for t in reversed(range(T)):
            future_ret = pi.rewards[t] + gamma * future_ret
            rets[t] = future_ret
        rets = torch.tensor(rets)
        log_probs = torch.stack(pi.log_probs)
        loss = - log_probs * rets # gradient term; Negative for maximizing
        loss = torch.sum(loss)
        optimizer.zero_grad()
        loss.backward() # backpropagate, compute gradients
        optimizer.step() # gradient-ascent, update the weights
        return loss

    def main():
        env = gym.make('CartPole-v0')
        in_dim = env.observation_space.shape[0] # 4
        out_dim = env.action_space.n # 2
        pi = Pi(in_dim, out_dim) # policy pi_theta for REINFORCE
        optimizer = optim.Adam(pi.parameters(), lr=0.01)
        for epi in range(300):
            state = env.reset()
            for t in range(200): # cartpole max timestep is 200
                action = pi.act(state)
                state, reward, done, _ = env.step(action)
                pi.rewards.append(reward)
                env.render()
                if done:
                    break
            loss = train(pi, optimizer) # train per episode
            total_reward = sum(pi.rewards)
            solved = total_reward > 195.0
            pi.onpolicy_reset() # onpolicy: clear memory after training
            print(f'Episode {epi}, loss: {loss}, \
            total_reward: {total_reward}, solved: {solved}')

if __name__ == '__main__':
    main()
```