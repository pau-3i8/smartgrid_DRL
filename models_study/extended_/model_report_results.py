from optuna.visualization._pareto_front import plot_pareto_front
from optuna.visualization import plot_param_importances, plot_optimization_history
import os, gym, optuna, numpy as np, pandas as pd, shutil, datetime

PLOTS_PATH = 'model_plot_results'
RESULTS_PATH = 'model_report_results'
N_TRIALS = None #if None reports from all trials

def doc_results(file_name, model):
    df = pd.read_csv(file_name+'.csv')    
    if N_TRIALS != None: df = df[:N_TRIALS]
    df.rename(columns = {'value':'reward',
                         'number':'trial'}, inplace=True)
                         
    forbidden = list(df[df.loc[:, 'state'] == 'RUNNING'].index.values)
    df.drop(forbidden, inplace=True, axis=0)
    
    for i in range(len(df)):
      if i not in forbidden:
        start = datetime.datetime.strptime(df.loc[i,'datetime_start'], "%Y-%m-%d %H:%M:%S.%f")
        end = datetime.datetime.strptime(df.loc[i,'datetime_complete'], "%Y-%m-%d %H:%M:%S.%f")
        df.loc[i, 'duration'] = float((end-start).total_seconds())
        
        if df.loc[i, 'params_activation_fn'] == 'leaky_relu':
            df.loc[i, 'params_activation_fn'] = 'leaky\_relu'
        if 'params_batch_size' in df.columns and 'params_n_steps' in df.columns:
            if df.loc[i, 'params_batch_size'] > df.loc[i, 'params_n_steps']:
                df.loc[i, 'params_batch_size'] = df.loc[i, 'params_n_steps']
                
    df.drop(['datetime_start', 'datetime_complete', 'user_attrs_Model'], inplace=True, axis=1)
    new = df.sort_values(by=['reward'], ascending=False)
    new.to_csv(f"{RESULTS_PATH}/sorted_reward.csv", mode='w', index=False)
    convert_to_latex(df, file_name, model)
    
def convert_to_latex(df, file_name, model):
    if model == 'A2C':
     header = [
         [
         '\multicolumn{1}{c}{Trial}',
         '\multicolumn{1}{c}{Reward}',
         '\multicolumn{1}{c}{Duration [sec]}',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'State',
         ],
         [
         '{}',
         '\multicolumn{1}{c}{}',
         '\multicolumn{1}{c}{}',
         'activation\_fn',
         'gae\_lambda',
         'gamma',
         'learning\_rate',
         'max\_grad\_norm',
         'n\_steps',
         'net\_arch',
         '{}',
         ]
     ]
    if model == 'PPO':
     header = [
         [
         '\multicolumn{1}{c}{Trial}',
         '\multicolumn{1}{c}{Reward}',
         '\multicolumn{1}{c}{Duration [sec]}',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'State',
         ],
         [
         '{}',
         '\multicolumn{1}{c}{}',
         '\multicolumn{1}{c}{}',
         'activation\_fn',
         'batch\_size',
         'clip\_range',
         'gae\_lambda',
         'gamma',
         'learning\_rate',
         'max\_grad\_norm',
         'n\_epochs',
         'n\_steps',
         'net\_arch',
         '{}',
         ]
     ]
    if model == 'TRPO':
     header = [
         [
         '\multicolumn{1}{c}{Trial}',
         '\multicolumn{1}{c}{Reward}',
         '\multicolumn{1}{c}{Duration [sec]}',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'State',
         ],
         [
         '{}',
         '\multicolumn{1}{c}{}',
         '\multicolumn{1}{c}{}',
         'activation\_fn',
         'batch\_size',
         'cg\_max\_steps',
         'gae\_lambda',
         'gamma',
         'learning\_rate',
         'max\_grad\_norm',
         'n\_critic\_updates',
         'n\_steps',
         'net\_arch',
         '{}',
         ]
     ]
    if model == 'DDPG':
     header = [
         [
         '\multicolumn{1}{c}{Trial}',
         '\multicolumn{1}{c}{Reward}',
         '\multicolumn{1}{c}{Duration [sec]}',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'State',
         ],
         [
         '{}',
         '\multicolumn{1}{c}{}',
         '\multicolumn{1}{c}{}',
         'activation\_fn',
         'batch\_size',
         'buffer\_size',
         'gamma',
         'learning\_rate',
         'net\_arch',
         'tau',
         'train\_freq',
         '{}',
         ]
     ]
    if model == 'TD3':
     header = [
         [
         '\multicolumn{1}{c}{Trial}',
         '\multicolumn{1}{c}{Reward}',
         '\multicolumn{1}{c}{Duration [sec]}',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'Hyperparameters',
         'State',
         ],
         [
         '{}',
         '\multicolumn{1}{c}{}',
         '\multicolumn{1}{c}{}',
         'activation\_fn',
         'batch\_size',
         'buffer\_size',
         'gamma',
         'learning\_rate',
         'net\_arch',
         'policy\_delay',
         'tau',
         'train\_freq',
         '{}',
         ]
     ] 
    df.columns = header
    with open(os.path.join(os.getcwd(), file_name+'.tex'), 'w') as file:
        file.write(
            df
            .round(6)
            .to_latex(
                index = False,
                caption = f'{model} hyperparameters optimization results.',
                label = f'tab:{model}_hyperparam',
                escape = False,
                column_format = 'c'*len(header[0]),
                multicolumn_format = 'c',
                longtable=True
            )
        )
        
def adjust_fig(fig, file_title, fig_title, y_title, x_title):
    fig.update_layout(
        title=fig_title,
        yaxis_title=y_title,
        xaxis_title=x_title,
        xaxis=dict(
            titlefont_size=14,
            tickfont_size=14
        ),
        yaxis=dict(
            titlefont_size=14,
            tickfont_size=14
        ),
        legend_font_size=14,
        font_family="Serif",
        font_size=12,
    )    
    fig.write_image(file_title, width=7.3, height=4.2, scale=1)

def best_model_results(study):

    # trials selecte from DB
    print("Number of finished trials: ", len(study.trials))
    trial = study.best_trial
    print("Best trial:", trial.number)
    print("  Value:", trial.value)
    print("  Params:")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    #print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
    
STORAGE_NAME = f'sqlite:///models_study_hyperparams.db'
STUDY_NAME = 'distributed-study-'
MODELS = ['A2C', 'PPO', 'TRPO', 'TD3', 'DDPG']

if __name__ == "__main__":

    for dir_ in [PLOTS_PATH, RESULTS_PATH]:
        path = os.path.join(os.getcwd(), dir_)
        os.makedirs(path, exist_ok=True)

    for model in MODELS:
    
        study = optuna.load_study(
            study_name=f'{STUDY_NAME}{model}',
            storage=STORAGE_NAME
        )
        
        best_model_results(study)
        
        # Plots
        fig = plot_optimization_history(study)
        adjust_fig(fig, f"{PLOTS_PATH}/{model}_optimization_history.png",
                        'Optimization history plot', 'Reward', '#Trials')
        fig = plot_param_importances(study)
        adjust_fig(fig, f"{PLOTS_PATH}/{model}_hyperparams_importance.png", 
                        "Hyperparameter importances", "Hyperparameters",
                        "Importance for reward")
        # Write report
        df_name = f"{RESULTS_PATH}/study_results_{model}_hyperparams"
        study.trials_dataframe().to_csv(df_name+'.csv', index=False)
    
        # Write latex
        doc_results(df_name, model)
