from optuna.visualization._pareto_front import plot_pareto_front
from optuna.visualization import plot_param_importances
import os, gym, optuna, numpy as np, pandas as pd, shutil, copy

PLOTS_PATH = 'plot_results'
RESULTS_PATH = 'report_results'
pd.options.display.float_format = '{:.6f}'.format

def doc_results(file_name):
    df = pd.read_csv(file_name+'.csv')
    df.rename(columns = {'values_0':'reward',
                         'values_1':'v_viol',
                         'values_2':'i_viol',
                         'values_3':'losses',
                         'values_4':'error',
                         'number':'trial'}, inplace=True)
    df.drop(['datetime_start','datetime_complete','duration'], inplace=True, axis=1)
    #inplace = False to not overwrite the original df
    for col in ['v_viol', 'i_viol', 'losses']:
        new = df.sort_values(by=[col], ascending=True) #ascending=True 1st values is the lowest
        new.to_csv(f"{RESULTS_PATH}/sorted_{col}.csv", mode='w', index=False)
    new = df.sort_values(by=['reward'], ascending=False)
    new.to_csv(f"{RESULTS_PATH}/sorted_reward.csv", mode='w', index=False)
    df = df[df['state'] == 'COMPLETE']
    convert_to_latex(df, file_name)
    
def convert_to_latex(df, file_name):
    header = [
        [
        '\multicolumn{1}{c}{Trial}',
        '\multicolumn{1}{c}{Reward}',
        '\multicolumn{1}{c}{V\_viol}',
        '\multicolumn{1}{c}{I\_viol}',
        '\multicolumn{1}{c}{Losses}',
        '\multicolumn{1}{c}{Error}',
        'Hyperparameters',
        'Hyperparameters',
        'Hyperparameters',
        'Hyperparameters',
        'Hyperparameters',
        'Hyperparameters',
        'Hyperparameters',
        'Hyperparameters',
        '\multicolumn{1}{c}{State}'
        ],
        [
        '\multicolumn{1}{c}{}',
        '\multicolumn{1}{c}{}',
        '\multicolumn{1}{c}{}',
        '\multicolumn{1}{c}{}',
        '\multicolumn{1}{c}{}',
        '\multicolumn{1}{c}{}',
        '\multicolumn{1}{c}{T}',
        '\multicolumn{1}{c}{beta\_1}',
        '\multicolumn{1}{c}{beta\_2}',
        '\multicolumn{1}{c}{beta\_3}',
        '\multicolumn{1}{c}{beta\_4}',
        '\multicolumn{1}{c}{beta\_5}',
        '\multicolumn{1}{c}{beta\_6}',
        '\multicolumn{1}{c}{tau}',
        '\multicolumn{1}{c}{}',
        ]
    ]
    df.columns = header
    with open(os.path.join(os.getcwd(), file_name+'.tex'), 'w') as file:
        file.write(
            df
            .round(6)
            .to_latex(
                index = False,
                caption='Custom environment hyperparameters optimization results.',
                label='tab:env_hyperparam',
                escape=False,
                column_format='ccH{}{'+f"{df.iloc[:,2].max()}"+'}{'+\
                                       f"{df.iloc[:,2].min()}"+'}H{}{'+\
                                       f"{df.iloc[:,3].max()}"+'}{'+\
                                       f"{df.iloc[:,3].min()}"+'}H{}{'+\
                                       f"{df.iloc[:,4].max()}"+'}{'+\
                                       f"{df.iloc[:,4].min()}"+'}H{}{'+\
                                       f"{df.iloc[:,5].max()}"+'}{'+\
                                       f"{df.iloc[:,5].min()}"+\
                              '}ccccccccc',
                multicolumn_format='c',
                longtable=True
            )
        )
        
def adjust_fig(fig, file_title, fig_title, y_title, x_title='Reward'):
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
    fig.show()
    
def best_trial_results(metric, trial):
    print(f"Trial with lowest {metric}:")
    print("  Trial number:", trial.number)    
    print("  Value:")
    for obj, value in zip(['Reward', 'v_viol', 'i_viol', 'losses', 'error'], trial.values):
        print("    {}: {}".format(obj, value))
    print("  Params:")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

STUDY_NAME = 'distributed-env-study'
STORAGE_NAME = 'sqlite:///env_study_hyperparams.db'

if __name__ == "__main__":


    for dir_ in [PLOTS_PATH, RESULTS_PATH]:
        path = os.path.join(os.getcwd(), dir_)
        os.makedirs(path, exist_ok=True)
    
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=STORAGE_NAME
    )
    
    # Write csv table reports
    df_name = f"{RESULTS_PATH}/study_results_env_hyperparams"
    study.trials_dataframe().to_csv(df_name+'.csv', index=False)
    # Write latex table reports
    doc_results(df_name)
    
    print("Number of finished trials: ", len(study.trials))
    for i, metric in zip(range(1,4), ['Voltage violations', 'Current violations', 'Losses']):
        fig = plot_pareto_front(study, target_names=['Reward', metric],
                                targets=lambda t: (t.values[0], t.values[i]))
        adjust_fig(fig, f"{PLOTS_PATH}/reward_vs_metric{i}.png", 
                        f"Pareto Reward vs. {metric}", metric)

    for i, metric in zip(range(1,4), ['v_viol', 'i_viol', 'losses']):
        trial_with_best_result = min(study.best_trials, key=lambda t: t.values[i])
        best_trial_results(metric, trial_with_best_result)

    #Learn which hyperparameters are affecting the reward most with hyperparameter importance
    fig = plot_param_importances(study, target=lambda t: t.values[0], target_name="reward")
    adjust_fig(fig, f"{PLOTS_PATH}/hyperparams_importance.png", "Hyperparameter importances", 
                    "Hyperparameters", "Importance for reward")
