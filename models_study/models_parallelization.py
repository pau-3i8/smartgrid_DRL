import os, psutil, subprocess, time

MODELS = ['A2C', 'PPO', 'TRPO', 'DDPG', 'TD3']
workers_per_model = 8 #user needs to know how many CPUs has the system. os.cpu_count()
    
if __name__ == "__main__":
    
    if (workers_per_model)*len(MODELS) > os.cpu_count():
        print('-- ERROR --: Too many simulataneous models.')
        print('Suggestion: Segment the models or Reduce the workers_per_model')
        exit()
        
    # optuna handles the parallelization through their distributed implementation.
    worker = 0
    for model in MODELS:
    
        for _ in range(workers_per_model): # each worker has N_TRIALS
        
            psutil.Process().cpu_affinity({worker})
            subprocess.Popen(f"python3 models_tuning.py {model}", shell=True)
            
            worker += 1
            time.sleep(3)
