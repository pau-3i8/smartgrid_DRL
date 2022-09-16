import os, psutil, subprocess, time

N_WORKERS = 1 # each worker has N_TRIALS

if __name__ == "__main__":
    # optuna handles the parallelization through their distributed implementation.
    for worker in range(N_WORKERS):
        psutil.Process().cpu_affinity({worker}) #allows fixing a processor number and use it as a seed. All distributed trials will have different seeds and allow reproducibility
        subprocess.Popen("python3 env_tuning.py", shell=True)
        # add a wait time to avoid race conditions
        time.sleep(2)
