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
        # indices=np.where(Q[observation]==np.max(Q[observation]))[0]
        action=np.argmax(Q[observation])
        # action=np.random.choice(indices)
    return action

discounting_factor=0.99

def expectation(Q,observation,epsilon):
    sum=0
    # indices=np.argwhere(Q[observation]==np.max(Q[observation]))
    for i in range(env.action_space.n):
        if i==np.argmax(Q[observation]):
            sum+=(1-epsilon+(epsilon/4))*Q[observation,i]
        else:
            sum+=(epsilon/4)*Q[observation,i]
    return sum

def nsarsa(step_size,Q,epsilon,n):
    observation,_=env.reset()
    action=epsilon_greedy(epsilon,Q,observation)
    states=[observation]
    actions=[action]
    rewards=[0]
    time=0
    T=float('inf')
    cur_sum=0
    while True:
        # print(time)
        if time<T:
            new_state,rew,terminated,truncated,_=env.step(action)
            states.append(new_state)
            rewards.append(rew)
            cur_sum+=rew
            if terminated or truncated:
                T=time+1
            else:
                action=epsilon_greedy(epsilon,Q,new_state)
                actions.append(action)

        update_time=time-n+1
        if update_time>=0:
            gt=0
            for i in range(update_time+1,min(update_time+n,T)+1):
                gt+=np.power(discounting_factor,i-update_time-1)*rewards[i]
            # gt=np.sum(np.power(discounting_factor,np.arange(update_time+1,min(update_time+n,T)+1))*np.array(rewards[update_time+1:min(update_time+n,T)+1]))
            if update_time+n<T:
                gt=gt+np.power(discounting_factor,n)*expectation(Q,states[update_time+n],actions[update_time+n])
                # gt=gt+np.power(discounting_factor,n)*Q[states[update_time+n],actions[update_time+n]]
            # print(gt)
            Q[states[update_time],actions[update_time]]=Q[states[update_time],actions[update_time]]+step_size*(gt-Q[states[update_time],actions[update_time]])
            
        if update_time==T-1:
            break       
        time+=1
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
num_episodes=2000
num_runs=10
sums=np.zeros(num_episodes)
for j in tqdm(range(num_runs),total=num_runs):
    Q=np.zeros((env.observation_space.n,env.action_space.n))
    for i in range(num_episodes):
        cur_sum=nsarsa(step_size,Q,0.2,2)
        sums[i]+=cur_sum
sums=sums/num_runs
grid=display_policy(Q)
print(grid)
plt.plot(np.arange(0,num_episodes),sums)
plt.ylim(-100,0)
plt.show()
