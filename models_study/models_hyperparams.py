from typing import Any, Dict
import numpy as np, optuna
from torch import nn as nn

def common_params(trial: optuna.Trial) -> Dict[str, Any]:    
    """
    Sampler for common hyperparams.
    :param trial:
    :return:
    """
    
    activation_fn = trial.suggest_categorical('activation_fn',['tanh','relu','elu','leaky_relu'])
    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU
    }[activation_fn]
    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)    
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    
    return {
        "learning_rate": learning_rate,
        "gamma": gamma
    }, activation_fn
    
def a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for A2C hyperparams.
    :param trial:
    :return:
    """
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3,0.5,0.6,0.7,0.8,0.9,1,2,5])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    
    # Adding User Attributes to Trials. To identify the corresponding model of each trial
    trial.set_user_attr('Model', 'A2C')
    
    # Independent networks usually work best when not working with images
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
        "big": [dict(pi=[400, 300], vf=[400, 300])]
    }[net_arch]
    
    com_params, activation_fn = common_params(trial)
           
    return {
        "n_steps": n_steps,
        "max_grad_norm": max_grad_norm,
        "gae_lambda": gae_lambda,
        "policy_kwargs": dict(net_arch=net_arch, activation_fn=activation_fn),
        **com_params
    }

def ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """
    # warning if not: n_steps * n_envs = k * batch_size -> batch_size < n_steps if n_envs = 1
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3,0.5,0.6,0.7,0.8,0.9,1,2,5])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    
    trial.set_user_attr('Model', 'PPO')
    
    if batch_size > n_steps:
        """
        the trial is correctly modified but the trial.params info is not, it is a report made
        when calling trail.suggest_ . When representing the results, the actual value of the
        batch has to be n_steps if batch > n_steps.
        """
        batch_size = n_steps
        
    # Independent networks usually work best when not working with images
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
        "big": [dict(pi=[400, 300], vf=[400, 300])]
    }[net_arch]
    
    com_params, activation_fn = common_params(trial)
    
    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "max_grad_norm": max_grad_norm,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "policy_kwargs": dict(net_arch=net_arch, activation_fn=activation_fn),
        **com_params
    }

def trpo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for TRPO hyperparams.
    :param trial:
    :return:
    """
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    cg_max_steps = trial.suggest_categorical("cg_max_steps", [5, 10, 20, 25, 30])
    n_critic_updates = trial.suggest_categorical("n_critic_updates", [5, 10, 20, 25, 30])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3,0.5,0.6,0.7,0.8,0.9,1,2,5])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    
    trial.set_user_attr('Model', 'TRPO')
    
    if batch_size > n_steps:
        batch_size = n_steps
        
    # Independent networks usually work best when not working with images
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
        "big": [dict(pi=[400, 300], vf=[400, 300])]
    }[net_arch]
    
    com_params, activation_fn = common_params(trial)
    
    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "cg_max_steps": cg_max_steps,
        "n_critic_updates": n_critic_updates,
        "gae_lambda": gae_lambda,
        "policy_kwargs": dict(net_arch=net_arch, activation_fn=activation_fn),
        **com_params
    }

def ddpg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for DDPG hyperparams.
    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [16,32,64,100,128,256,512,1024,2048])
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512])
    gradient_steps = train_freq
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    # tau = Polyak coeff
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])
    
    trial.set_user_attr('Model', 'DDPG')
    
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300]
    }[net_arch]
    
    com_params, activation_fn = common_params(trial)
    
    return {
        "batch_size": batch_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "buffer_size": buffer_size,
        "tau": tau,
        "policy_kwargs": dict(net_arch=net_arch, activation_fn=activation_fn),
        **com_params
    }

def td3_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for TD3 hyperparams.
    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [16,32,64,100,128,256,512,1024,2048])
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512])
    gradient_steps = train_freq
    policy_delay = trial.suggest_categorical("policy_delay", [2, 3])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    # tau = Polyak coeff
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])
    
    trial.set_user_attr('Model', 'TD3')
    
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300]
    }[net_arch]
    
    com_params, activation_fn = common_params(trial)
    
    return {
        "batch_size": batch_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "policy_delay": policy_delay,
        "buffer_size": buffer_size,
        "tau": tau,
        "policy_kwargs": dict(net_arch=net_arch, activation_fn=activation_fn),
        **com_params
    }
