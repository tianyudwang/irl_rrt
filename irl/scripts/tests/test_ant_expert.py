import sys
import os
import pickle

import numpy as np
import gym
import mujoco_maze
import pandas as pd

from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
from icecream import ic
from tqdm import trange


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
    
    test = False
    env_name = 'AntUMaze-v1'
    algo = "TQC"
    
    env = gym.make(env_name)
    
    ant_expert = os.path.abspath(f"./logs/{env_name}/{algo}/best_model/best_model")
    
    
    if test:
        model = SAC.load(ant_expert)
        
        
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
                act,_ = model.predict(obs)
                obs, _, done, _ = env.step(act)
                minMax(obs, qpos_min, qpos_max, qvel_min, qvel_max)
                
                # env.render()
                if done:
                    obs = env.reset()
                    minMax(obs, qpos_min, qpos_max, qvel_min, qvel_max)
                    
        # ic(qpos_min.reshape(-1,1), qpos_max.reshape(-1,1), qvel_min.reshape(-1,1), qvel_max.reshape(-1,1))
        
        learned_model = {
            "qpos_min": qpos_min,
            "qpos_max": qpos_max,
            "qvel_min": qvel_min,
            "qvel_max": qvel_max
        }
        
        ic(learned_model)

        with open(f"./learned_model.pkl", "wb") as f:
            pickle.dump(learned_model, f)
        
    
    with open(f"./learned_model.pkl", "rb") as f:
        learned_model = pickle.load(f)
    
    
    qpos_result = np.hstack([learned_model["qpos_min"].reshape(-1,1), learned_model["qpos_max"].reshape(-1,1)])
    qvel_result = np.hstack([learned_model["qvel_min"].reshape(-1,1), learned_model["qvel_max"].reshape(-1,1)])
    ic(qpos_result)
    ic(qvel_result)        
    
    
    df_qpos = pd.DataFrame(
        qpos_result,
        index = ["x", "y", "z", "qw", "qx", "qy", "qz", "hip1", "ankle1", "hip2", "ankle2", "hip3", "ankle3", "hip4", "ankle4"],
        columns=["qpos_min", "qpos_max"])
    print(df_qpos.head(15))
    
    df_qvel = pd.DataFrame(
        qvel_result,
        index=["x_dot", "y_dot", "z_dot", "qx_dot", "qy_dot", "qz_dot", "hip1_dot", "ankle1_dot", "hip2_dot", "ankle2_dot", "hip3_dot", "ankle3_dot", "hip4_dot", "ankle4_dot"],
        columns=["qvel_min", "qvel_max"])
    print(df_qvel.head(15))
    
    

    
    
