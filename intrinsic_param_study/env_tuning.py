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

GRID = n_bus_network()
SAVE_PATH = 'training/opt'
LOG_DIR = 'training/logs'
VERBOSE = 1

STUDY_NAME = 'distributed-env-study'
STORAGE_NAME = 'sqlite:///env_study_hyperparams.db'

# n_steps > timesteps --> timestemps <= n_steps

TIMESTEPS = 100000
N_EVAL_EPISODES = 24*365
N_SEEDS = 5
N_TRIALS = 1

#if it is not used a distributed load, the following file can be a global dictionary.
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
    
def env_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for env hyperparams.
    :param trial:
    :return:
    """
    beta1 = trial.suggest_categorical("beta_1", [1, 10, 100, 1000])
    beta2 = trial.suggest_categorical("beta_2", [1, 10, 100, 1000])
    beta3 = trial.suggest_categorical("beta_3", [1, 10, 100, 1000])
    beta4 = trial.suggest_categorical("beta_4", [1, 10, 100, 1000])
    beta5 = trial.suggest_categorical("beta_5", [1, 10, 100, 1000])
    beta6 = trial.suggest_categorical("beta_6", [1, 10, 100, 1000])
    tau = trial.suggest_categorical("tau", [6, 8, 12, 24])
    T = trial.suggest_categorical("T", [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
    
    return {
        'T': T,
        'tau': tau,
        'beta_1': beta1,
        'beta_2': beta2,
        'beta_3': beta3,
        'beta_4': beta4,
        'beta_5': beta5,
        'beta_6': beta6
    }

import matplotlib.pyplot as plt

def eval_model(model, env, episodes): #custom evaluate_policy
    obs = env.reset()
    done = False
    
    rewards = np.zeros(episodes)
    v_violations = np.zeros(episodes)
    i_violations = np.zeros(episodes)
    losses = np.zeros(episodes)
    
    demand = np.zeros(episodes)
    agent_demand = np.zeros(episodes)
    
    for i in range(episodes):    
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        info = info[0]
        rewards[i] = reward
        v_violations[i] = info['v_violations']
        i_violations[i] = info['i_violations']
        losses[i] = info['losses']
        
        demand[i] = info['real_demand']
        agent_demand[i] = info['result']
        
        if done: obs = env.reset()
        
    error = abs(demand.sum()-agent_demand.sum())/demand.sum()*100 #%
    
    return (
            np.mean(rewards),
            np.mean(v_violations),
            np.mean(i_violations),
            np.mean(losses),
            error
           )

# CUSTOM ENV HYPERPARAMS with PPO
def optimize_agent(trial, seed):
    
    nan_encountered = False
    
    suggested_hyparams = pd.read_csv(HYPARAMS_FILE)
    for hyparams_set in range(len(suggested_hyparams)):
        if env_params(trial) == suggested_hyparams.iloc[hyparams_set, :].to_dict() and seed==0:
            raise optuna.exceptions.TrialPruned()
    
    df = pd.DataFrame.from_dict([env_params(trial)])
    df.to_csv(HYPARAMS_FILE, mode='a', header= not os.path.exists(HYPARAMS_FILE), index=False)
    
    try:
        
        env = GridEnv(powergrid=GRID, total_timesteps=TIMESTEPS, seed=seed, 
                      id_=trial.number, **env_params(trial))
        env = DummyVecEnv([lambda: env])
        model = PPO("MlpPolicy", env, verbose=VERBOSE, seed=set_random_seed(seed),
                    tensorboard_log=LOG_DIR)
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,
                    tb_log_name=f"{trial.number}_{seed}")
        env.close()
        
        # Starts evaluation
        env = GridEnv(powergrid=GRID, total_timesteps=N_EVAL_EPISODES, evaluation=True,
                      seed=seed, id_=trial.number, **env_params(trial))
        env = DummyVecEnv([lambda: env])
        
        #metrics are saved in a file
        reward, v_viol, i_viol, losses, error = eval_model(model, env, N_EVAL_EPISODES)
        # if error > 0.5% prune the trial and dont explore the seeds
        if error > 0.5:
            #remove the evaluation file for consistency.
            for s in range(N_SEEDS): #si peta en un seed posterior al primer
                path = os.path.join('evaluation', f"{trial.number}_{s}.csv")
                if os.path.exists(path): os.remove(path)
            
            raise optuna.exceptions.TrialPruned()
        else:
            save_path = f"{SAVE_PATH}/{trial.number}_{seed}"
            model.save(save_path)
    
    except AssertionError as error:
        # Sometimes, random hyperparams can generate nan
        print(error)
        nan_encountered = True
    finally:
        # Free memory in case render() is called
        model.env.close()
        env.close()
    # Tell the optimizer that the trial failed
    if nan_encountered:
        return [float("nan")]*5

    return reward, v_viol, i_viol, losses, error
    
def objective_wrapper(trial):
    results = {}
    for seed_value in range(N_SEEDS):
      result = optimize_agent(trial, seed=seed_value)
      results[seed_value] = result    
    # Aggregate results and determine the score.
    return [np.mean([values for values in results.values()][i]) for i in range(len(results))]
    
if __name__ == "__main__":

    for dir_ in [SAVE_PATH, LOG_DIR]:
        path = os.path.join(os.getcwd(), dir_)
        os.makedirs(path, exist_ok=True)
    
    seed = psutil.Process().cpu_affinity()[0] #seed is the processor number.
    
    # Create a storage
    storage = optuna.storages.RDBStorage(url=STORAGE_NAME)
    
    # Multi-objective optimization
    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize", "minimize", "minimize"],
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
