from stable_baselines3.common.env_checker import check_env
from custom_env import GridEnv
from powergrids import two_bus_network, n_bus_network

# It will check your custom environment and output additional warnings if needed
env = GridEnv(two_bus_network(), 100, seed=0)
check_env(env)
