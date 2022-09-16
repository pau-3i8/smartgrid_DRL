from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO, DDPG, TD3
from sb3_contrib import TRPO

from models_hyperparams import a2c_params, ppo_params, trpo_params, ddpg_params, td3_params
import os, gym, optuna, numpy as np, pandas as pd, psutil, sys
from powergrids import two_bus_network, n_bus_network
from custom_env import GridEnv

from typing import Any, Dict

from optuna.visualization import plot_param_importances, plot_optimization_history

GRID = n_bus_network()
SAVE_PATH = f'training_algos/opt/{sys.argv[1]}'
LOG_DIR = f'./training_algos/logs/{sys.argv[1]}' 
# tensorboard_log = LOG_DIR, tb_log_name = LOG_DIR/log_name
VERBOSE = 0

TIMESTEPS = 100000
N_EVAL_EPISODES = 365*24 #eval
N_SEEDS = 5
N_TRIALS = 100

# Params from env_tuning study
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

# sys.argv[1] defines the model from models_dict that will be optimized
models_dict = {
    'A2C': [A2C, a2c_params],
    'PPO': [PPO, ppo_params],
    'TRPO': [TRPO, trpo_params],
    'DDPG': [DDPG, ddpg_params],
    'TD3': [TD3, td3_params]
}

def eval_model(model, env, episodes): #custom evaluate_policy
    obs = env.reset()
    done = False    
    rewards = np.zeros(episodes)
    
    for i in range(episodes):    
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        rewards[i] = reward
        if done: obs = env.reset()
            
    return np.mean(rewards)
      
# CUSTOM MODEL HYPERPARAMS
def optimize_agent(trial, seed):
    
    nan_encountered = False
    
    try:
        env = GridEnv(powergrid=GRID, total_timesteps=TIMESTEPS, seed=seed, 
                      id_=trial.number, **env_params)
        env = DummyVecEnv([lambda: env])            
        model_alg, model_params = models_dict[sys.argv[1]]
        model = model_alg("MlpPolicy", env, verbose=VERBOSE, seed=set_random_seed(seed),
                          tensorboard_log = LOG_DIR, **model_params(trial))
        model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False,
                    tb_log_name = f'{trial.number}_{seed}')
        env.close()
                
        # Starts evaluation
        env = GridEnv(powergrid=GRID, total_timesteps=N_EVAL_EPISODES, model=f'{sys.argv[1]}',
                      evaluation=True, seed=seed, id_=trial.number, **env_params)
        env = DummyVecEnv([lambda: env])
        
        #metrics are saved in a file
        mean_reward = eval_model(model, env, N_EVAL_EPISODES)
            
    except AssertionError as e:
        # Sometimes, random hyperparams can generate nan
        print(e)
        nan_encountered = True
    finally:
        # Free memory in case render() is called
        model.env.close()
        env.close()
    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")

    save_path = f"{SAVE_PATH}/{trial.number}_{seed}"
    model.save(save_path)
    return mean_reward

def objective_wrapper(trial):
    results = {}
    for seed_value in range(N_SEEDS):
      result = optimize_agent(trial, seed=seed_value)
      results[seed_value] = result
    # Aggregate results and determine the score.
    return np.mean(list(results.values()))

STUDY_NAME = f'distributed-study-{sys.argv[1]}'
STORAGE_NAME = f'sqlite:///models_study_hyperparams.db' #single DB for all models' studies

if __name__ == "__main__":

    for dir_ in [SAVE_PATH, LOG_DIR]:
        path = os.path.join(os.getcwd(), dir_)
        os.makedirs(path, exist_ok=True)
    
    seed = psutil.Process().cpu_affinity()[0] #seed is the processor number.
    
    # Create a storage
    storage = optuna.storages.RDBStorage(url=STORAGE_NAME)
    
    # Single-objective optimization
    study = optuna.create_study(
        directions=["maximize"],
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=seed),
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=True
    )
    try:
        study.optimize(
            objective_wrapper,
            n_trials=N_TRIALS,
            show_progress_bar=True,
            n_jobs=1
        )
    except KeyboardInterrupt:
        pass
