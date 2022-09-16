from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO, DDPG, TD3
from sb3_contrib import TRPO

from optuna.visualization._pareto_front import plot_pareto_front
from optuna.visualization import plot_param_importances

import os, gym, optuna, numpy as np, pandas as pd, shutil, psutil
from powergrids import two_bus_network, n_bus_network
from custom_env import GridEnv
from tqdm import tqdm

from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
from typing import Any, Dict

GRID = two_bus_network()
SAVE_PATH = 'training/opt'
MODEL = '216_0'

N_EVAL_EPISODES = 24*365

HYPARAMS_FILE = 'suggested_params.csv'
pd.DataFrame({
    'T': [],
    'tau': [],
    'beta_1': [],
    'beta_2': [],
    'beta_3': [],
    'beta_4': [],
    'beta_5': [],
    'beta_6': []
}).to_csv(HYPARAMS_FILE, mode='a', header = not os.path.exists(HYPARAMS_FILE), index=False)

STUDY_NAME = 'distributed-env-study'
STORAGE_NAME = 'sqlite:///env_study_hyperparams.db'
study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_NAME)
env_params = study.trials[int(f'{MODEL}'.split('_')[0])].params

def eval_model(model, env, episodes): #custom evaluate_policy
    obs = env.reset()
    done = False
    
    rewards = np.zeros(episodes)
    v_violations = np.zeros(episodes)
    i_violations = np.zeros(episodes)
    losses = np.zeros(episodes)
    
    demand = np.zeros(episodes)
    agent_demand = np.zeros(episodes)
    #delta_p = np.zeros(episodes)
    
    n_actions = len(model.predict(obs)[0][0])
    actions_df = pd.DataFrame(np.zeros((episodes, n_actions)))
    actions_df.columns = ['flex_load_'+str(i) for i in range(n_actions)]
    
    for i in tqdm(range(episodes)):
        action, _ = model.predict(obs) #el segon argument és el següent estat
        obs, reward, done, info = env.step(action)
        info = info[0]
        rewards[i] = reward
        v_violations[i] = info['v_violations']
        i_violations[i] = info['i_violations']
        losses[i] = info['losses']
        
        demand[i] = info['real_demand']
        agent_demand[i] = info['result']
        #delta_p[i] = info['load_modulation']
        
        actions_df.iloc[i, :] = action
        
        if done: obs = env.reset()
    
    actions_df.to_csv('actions.csv') #only 1 run
    
    #plt.step(np.arange(1, episodes+1, 1), demand, label='demand')
    #plt.step(np.arange(1, episodes+1, 1), agent_demand, label='agent_demand')
    #plt.step(np.arange(1, episodes+1, 1), delta_p, label='modulation_curve')
    #plt.hlines(0, 1, episodes+1, lw=0.5, alpha=0.7, zorder=2)
    #plt.legend()
    #plt.show()
    
    error = abs(demand.sum()-agent_demand.sum())/demand.sum()*100 #%
    
    print(np.mean(rewards), np.mean(v_violations), np.mean(i_violations), np.mean(losses), error)

        
env = GridEnv(powergrid=GRID, total_timesteps=N_EVAL_EPISODES, evaluation=False, seed=0, **env_params)
env = DummyVecEnv([lambda: env])
model = PPO.load(f"{SAVE_PATH}/{MODEL}", env)
eval_model(model, env, N_EVAL_EPISODES)

import action_analisys
