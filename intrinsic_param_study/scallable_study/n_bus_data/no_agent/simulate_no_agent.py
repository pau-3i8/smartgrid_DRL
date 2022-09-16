from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO, DDPG, TD3
from sb3_contrib import TRPO

from optuna.visualization._pareto_front import plot_pareto_front
from optuna.visualization import plot_param_importances

import os, gym, optuna, numpy as np, pandas as pd, shutil, psutil
from powergrids import two_bus_network, n_bus_network
from custom_env import GridEnv

from typing import Any, Dict
from tqdm import tqdm

GRID = n_bus_network()

N_EVAL_EPISODES = 24*365

env_params = {
    'T': 16,
    'tau': 6,
    'beta_1': 100,
    'beta_2': 1,
    'beta_3': 1,
    'beta_4': 1,
    'beta_5': 1000,
    'beta_6': 1000
}

import matplotlib.pyplot as plt

def eval_model(model, env, episodes): #custom evaluate_policy
    obs = env.reset()
    done = False
    
    for i in tqdm(range(episodes)):
        # the custom env generates the files needed for the no_agent analysis
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
                
        if done: obs = env.reset()        
        
if __name__ == "__main__":

    env = GridEnv(powergrid=GRID, total_timesteps=N_EVAL_EPISODES, evaluation=True,
                  seed=0, id_=0, **env_params)
    env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", env, seed=set_random_seed(0))
    eval_model(model, env, N_EVAL_EPISODES)
    
