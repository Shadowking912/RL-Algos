import numpy as np
import gymnasium as gym
import sys

env = gym.make("CliffWalking-v0",render_mode="human")
transitions=env.unwrapped.P
observation,info=env.reset()


V=np.zeros(env.observation_space.n)
pi=np.zeros((env.observation_space.n,env.action_space.n))+1
pi[:,3:4]=0
pi=pi/3
pi[25:35,:]=1
pi[25:35,2:4]=0
pi[25:35]=pi[25:35]/2
discounting_factor=0.9

def td(step_size,V,pi):
    observation,_=env.reset()
    while True:
        action=np.random.choice(np.arange(env.action_space.n),p=pi[observation])
        new_state,rew,terminated,truncated,_=env.step(action)
        V[observation]=V[observation]+step_size*(rew+discounting_factor*V[new_state]-V[observation])
        if terminated or truncated:
            print("terminated")
            break
        observation=new_state
        
def display(v):
    grid=np.zeros((4,12))
    for i in range(len(v)):
        row=i//12
        column=i%12
        grid[row,column]=v[i]
    return grid

step_size=0.1
num_episodes=5
for i in range(num_episodes):
    td(step_size,V,pi)
grid=display(V)
print(grid)
