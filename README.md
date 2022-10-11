# smartgrid_DRL

This repository is a deep reinforcement learning tool for demand response in smart grids, with high penetration of renewable energy sources.
___

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Documentation](#documentation)
<!---
- [Author Info](#author-info)
-->
___

## Description

The respository has been developed along side [this master's thesis](https://upcommons.upc.edu/handle/2117/374273), which provides the mathematical calculations, data used and analysis results present in this repository. The original work remains practically intact, despite some barely perceptible imprecisions, which has been [updated.](https://github.com/pau-3i8/smartgrid_DRL/tree/main/master_thesis). It is reproducibe, even in a distributed environment, being able to obtain the same result present in the thesis, as long as the simulations run in a 40 CPU machine. The reproducibility depends on the number of cores used for the distributed simulation, hence the CPU number is used as a seed.

It is used the [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) agents for the explored DRL algorithms, and the [Optuna](https://github.com/optuna/optuna) hyperparameter optimization framework.

The repository provides scripts for training and evaluating the agents in a custom environment for the smart grid simulation, distributed hyperparameters tuning, writing report tables and plotting results. The environment is also integrated with the [Pandapower](https://github.com/e2nIEE/pandapower) power flow calculator.
___

## Installation

#### Prerequisites

The Python packages used are:
- kaleido >= 0.2.1
- natsort >= 8.1.0
- numba >= 0.56.0
- optuna >= 3.0.2
- pandapower >= 2.10.1
- plotly >= 5.10.0
- psutil >= 5.9.1
- sb3_contrib >= 1.6.0
- sklearn >= 0.0
- stable-baselines3 >= 1.6.0
- tensorboard >= 2.10.0

#### Plotting set-up

Installing latex for plotting
```
sudo apt-get install python3-graphviz python3-tk texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvipng cm-super
```
___
## Documentation

The code runs as is. A single-model multi-objective optimization performed in [env_tuning](https://github.com/pau-3i8/smartgrid_DRL/tree/main/intrinsic_param_study/env_tuning.py) is distributed with the [env_parallelization](https://github.com/pau-3i8/smartgrid_DRL/tree/main/intrinsic_param_study/env_parallelization.py) program. The same idea is reproduced for a multi-model single-objective optimization performed with the analogous [models_tuning](https://github.com/pau-3i8/smartgrid_DRL/tree/main/models_study/models_tuning.py) and [models_parallelization](https://github.com/pau-3i8/smartgrid_DRL/tree/main/models_study/models_parallelization.py) programs.

Once the training and evaluation are done, the files with "visualization" in the name will plot analysis acording to the data from the created "evaluation/" folder. The files with "report" in the name will look at the .db file created by optuna once the study is over, to generate .tex and .csv table reports and plot information from the hyperparameters in the optuna.create_study().

[Back to the top](#smartgrid_DRL)

<!---
___
## Authos Infor

- LinkedIn - [Pau Fisco](https://www.linkedin.com/in/pau-fisco-compte/?locale=en_US)

[Back To The Top](#smartgrid_DRL)
-->
