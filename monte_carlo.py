import numpy as np
import gymnasium as gym
import sys

env = gym.make("CliffWalking-v0",render_mode="human")
transitions=env.unwrapped.P
observation,info=env.reset()
x=list(transitions[35][2][0])
x[2]=10
transitions[35][2]=[tuple(x)]


def generate_episode(pi):
    observations=[]
    actions=[]
    rews=[0]
    observation,_=env.reset()
    observations.append(observation)
    while True:
        action=np.random.choice(np.arange(env.action_space.n),p=pi[observation])
        actions.append(action)
        observation,reward,terminated,truncated,info=env.step(action)
        observations.append(observation)
        rews.append(reward)
        if terminated or truncated:
            print("terminated")
            break
    return observations,actions,rews

V=np.zeros(env.observation_space.n)
counts_V=np.zeros(env.observation_space.n)
pi=np.zeros((env.observation_space.n,env.action_space.n))+1
pi[:,3:4]=0
pi=pi/3
pi[25:35,:]=1
pi[25:35,2:4]=0
pi[25:35]=pi[25:35]/2

print(pi)
discounting_factor=0.9

def monte_carlo_evaluation_fs(V):
    observations,_,rews=generate_episode(pi)
    gt=0
    for i in range(len(observations)-2,-1,-1):
        observation=observations[i]
        gt=rews[i+1]+discounting_factor*gt
        if observation not in observations[0:i]:
            V[observation]=(V[observation]*counts_V[observation]+gt)/(counts_V[observation]+1)
            counts_V[observation]+=1

Q=np.zeros((env.observation_space.n,env.action_space.n))-1e9
counts_q=np.zeros((env.observation_space.n,env.action_space.n))

def monte_carlo_es(pi,Q,num_episodes):
    for j in range(num_episodes):
        print("episode: ",j)
        observations,actions,rews=generate_episode(pi)
        trajectory=list(zip(observations[:-1],actions))
        gt=0
        for i in range(len(observations)-2,-1,-1):
            observation=observations[i]
            action=actions[i]
            gt=rews[i+1]+discounting_factor*gt
            if (observation,action) not in trajectory[0:i]:
                Q[observation,action]=(Q[observation,action]*counts_q[observation,action]+gt)/(counts_q[observation,action]+1)
                counts_q[observation,action]+=1
                print(observation,action,Q[observation])
                pi[observation,:]=0
                pi[observation,np.argmax(Q[observation])]=1


def display(v):
    grid=np.zeros((4,12))
    for i in range(len(v)):
        row=i//12
        column=i%12
        grid[row,column]=v[i]
    return grid

def display_policy(pi):
    grid=np.zeros((4,12))
    for i in range(len(pi)):
        row=i//12
        column=i%12
        grid[row,column]=pi[i][0]
    return grid

num_episodes=5
# for i in range(num_episodes):
#     monte_carlo_evaluation_fs(V)

monte_carlo_es(pi,Q,num_episodes)
grid=display_policy(pi)
print(grid)
