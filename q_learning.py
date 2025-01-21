import numpy as np
import gymnasium as gym
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

env = gym.make("CliffWalking-v0")
transitions=env.unwrapped.P
observation,info=env.reset()


V=np.zeros(env.observation_space.n)
pi=np.zeros(env.observation_space.n)


def epsilon_greedy(epsilon,Q,observation):
    if np.random.random()<epsilon:
        action=np.random.randint(env.action_space.n)
    else:
        action=np.argmax(Q[observation])
    return action

discounting_factor=0.9

def q_learning(step_size,Q,epsilon):
    observation,_=env.reset()
    cur_sum=0
    while True:
        action=epsilon_greedy(epsilon,Q,observation)
        new_state,rew,terminated,truncated,_=env.step(action)
        Q[observation,action]=Q[observation,action]+step_size*(rew+discounting_factor*np.max(Q[new_state])-Q[observation,action])
        observation=new_state
        cur_sum+=rew
        if terminated or truncated:
            break
    return cur_sum
        
        
def display(v):
    grid=np.zeros((4,12))
    for i in range(len(v)):
        row=i//12
        column=i%12
        grid[row,column]=v[i]
    return grid

def display_policy(Q):
    grid=np.zeros((4,12))
    for i in range(len(Q)):
        row=i//12
        column=i%12
        grid[row,column]=np.argmax(Q[i])
    return grid



step_size=0.1
num_episodes=1000
num_runs=50
sums=np.zeros(num_episodes)
for j in tqdm(range(num_runs),total=num_runs):
    Q=np.zeros((env.observation_space.n,env.action_space.n))
    for i in range(num_episodes):
        cur_sum=q_learning(step_size,Q,0.1)
        sums[i]+=cur_sum
sums=sums/num_runs
grid=display_policy(Q)
print(grid)
plt.plot(np.arange(0,num_episodes),sums)
plt.ylim(-100,0)
plt.show()
