import sys

import numpy as np
import time
import gym
import mujoco_maze

from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from icecream import ic
from tqdm import trange


def learnModel():
    max_qpos = np.asarray([-float('inf')] * 15)
    min_qpos = np.asarray([float('inf')] * 15)
    max_qvel = np.asarray([-float('inf')] * 14)
    min_qvel = np.asarray([float('inf')] * 14)
    obs = env.reset()
    qpos = obs[:15]
    qvel = obs[15:-1]
    for i in range(15):
            if qpos[i] > max_qpos[i]:
                max_qpos[i] = qpos[i] 
            
            if qpos[i]  < min_qpos[i]:
                min_qpos[i] = qpos[i]
        
    for i in range(14):
        if qvel[i] > max_qvel[i]:
            max_qvel[i] = qvel[i] 
        
        if qvel[i]  < min_qvel[i]:
            min_qvel[i] = qvel[i]
    for _ in trange(int(5e5)):
        act = env.action_space.sample()
        obs, rew, done, _ = env.step(act)
        qpos = obs[:15]
        qvel = obs[15:-1]
        
        for i in range(15):
            if qpos[i] > max_qpos[i]:
                max_qpos[i] = qpos[i] 
            
            if qpos[i]  < min_qpos[i]:
                min_qpos[i] = qpos[i]
        
        for i in range(14):
            if qvel[i] > max_qvel[i]:
                max_qvel[i] = qvel[i] 
            
            if qvel[i]  < min_qvel[i]:
                min_qvel[i] = qvel[i]
        if done:
            obs = env.reset()
            qpos = obs[:15]
            qvel = obs[15:-1]
            for i in range(15):
                    if qpos[i] > max_qpos[i]:
                        max_qpos[i] = qpos[i] 
                    
                    if qpos[i]  < min_qpos[i]:
                        min_qpos[i] = qpos[i]
                
            for i in range(14):
                if qvel[i] > max_qvel[i]:
                    max_qvel[i] = qvel[i] 
                
                if qvel[i]  < min_qvel[i]:
                    min_qvel[i] = qvel[i]
        
        
    ic(min_qpos, max_qpos)
    ic(min_qvel, max_qvel)
    min_qpos_deg = np.rad2deg(min_qpos[7:])
    max_qpos_deg = np.rad2deg(max_qpos[7:])
    ic(min_qpos_deg, max_qpos_deg)
    ic(np.around(min_qpos_deg), np.around(max_qpos_deg))

def minMax(obs, min_qpos, max_qpos, min_qvel, max_qvel):
    qpos = obs[:15]
    qvel = obs[15:-1]
    
    for i in range(15):
            if qpos[i] > max_qpos[i]:
                max_qpos[i] = qpos[i] 
            
            if qpos[i]  < min_qpos[i]:
                min_qpos[i] = qpos[i]
        
    for j in range(14):
        if qvel[j] > max_qvel[j]:
            max_qvel[j] = qvel[j] 
        
        if qvel[j]  < min_qvel[j]:
            min_qvel[j] = qvel[j]
    

if __name__ == '__main__':
    env = TimeFeatureWrapper(gym.make('AntUMaze-v0'))
    test = False
    
    model = TQC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_starts=10_000,
        buffer_size=int(3e6),
        learning_rate=3e-4,
        device='auto', 
    )
    model.learn(total_timesteps=5e5)
    
    model.save("ant_expert")
    
    
    if test: 
        model = SAC.load("ant_expert")
    
    
        inf = float("inf")
        qpos_min = np.full(15, inf)
        qpos_max = np.full(15, -inf)
        qvel_min = np.full(14, inf)
        qvel_max = np.full(14, -inf)

        for i in trange(100):
            done = False
            j = 0
            obs = env.reset()
            minMax(obs, qpos_min, qpos_max, qvel_min, qvel_max)
            while not done:
                # act,_ = model.predict(obs)
                act =env.action_space.sample()
                obs, _, done, _ = env.step(act)
                minMax(obs, qpos_min, qpos_max, qvel_min, qvel_max)
                
                # env.render()
                if done:
                    obs = env.reset()
                    minMax(obs, qpos_min, qpos_max, qvel_min, qvel_max)
                    
        ic(qpos_min.reshape(-1,1), qpos_max.reshape(-1,1), qvel_min.reshape(-1,1), qvel_max.reshape(-1,1))
    
    
