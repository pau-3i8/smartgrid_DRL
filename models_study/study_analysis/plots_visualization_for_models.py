import numpy as np, pandas as pd, copy, os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from datetime import datetime, timedelta

from numpy.lib.stride_tricks import as_strided
from natsort import natsorted
from tqdm import tqdm

#graphic's format
pylab.rcParams['axes.labelsize'] = 9
pylab.rcParams['xtick.labelsize'] = 9
pylab.rcParams['ytick.labelsize'] = 9
pylab.rcParams['legend.fontsize'] = 9
pylab.rcParams['font.family'] = 'serif'
pylab.rcParams['font.sans-serif'] = ['Computer Modern Roman']
pylab.rcParams['text.usetex'] = True
pylab.rcParams['figure.figsize'] = 7.3, 4.2

PATH = 'evaluation/'
PLOTS_PATH = 'plot_results'

START = '2015-06-01'
END = '2015-06-07'

DAYS = 59 # days power range
N_SEEDS = 5
N_TRIALS = 50


def compute(x):
    mean = np.mean(x)
    sup_bound = np.mean(x) + np.std(x)/np.sqrt(len(x))*1.96 #conf. int 95% normal distrib.
    inf_bound = np.mean(x) - np.std(x)/np.sqrt(len(x))*1.96 #conf. int 95% normal distrib.
    return (inf_bound, mean, sup_bound)

def metric_vectors(files, metric, total_periods, period):
    # Mean and variance throughout a 'total' period, times 'total_period'.
    
    temp = [[] for _ in range(N_SEEDS)]
    #np.zeros((N_SEEDS, total_periods)) with tuples as elements
    array = [[] for _ in range(len(files)//N_SEEDS)]
    #np.zeros((len(files)//N_SEEDS, total_periods + 1)) with tuples as elements
    k = 0
    for s in range(len(files)//N_SEEDS): # n_sets
        for j in range(N_SEEDS):            
            file_path = os.path.join(PATH, files[k+j])
            name = files[k+j].split('_')
            df = pd.read_csv(file_path)
            for i in range(total_periods):
                temp[j].append(compute(df.loc[period*i:period*(i+1), metric].values.ravel()))
        for i in range(total_periods):
            array[s].append((np.mean([metric[0] for metric in [seed[i] for seed in temp]]), 
                             np.mean([metric[1] for metric in [seed[i] for seed in temp]]),
                             np.mean([metric[2] for metric in [seed[i] for seed in temp]])))
        
        temp = [[] for _ in range(N_SEEDS)] 
        array[s].append([files[k+j].split('_')[1]])
        # trial = set
        k += j+1
        
        model = ''.join([m if m in ['A2C','PPO','TRPO','DDPG','TD3'] else '' for m in name])
        array[s].append(model)
    # array = [set1 = [(week1_), (week_2), ... [trial], model]
    #          set2 = [(week1_), (week_2), ... [trial], model]]
    return array

def accumulated_results():

    #files = next(os.walk(PATH), (None, None, []))[2]
    
    files = []
    for model in tqdm(['A2C', 'PPO', 'TRPO', 'DDPG', 'TD3'], leave = True):
        for trial in tqdm(range(N_TRIALS), leave = False):
            try:
                # PPO had fails for hyperparams incompatibility (?)
                for j in tqdm(range(N_SEEDS), leave =False):
                    f = os.path.join(PATH, f'{model}_{trial}_{j}.csv')
                    if os.path.exists(f): files.append(f'{model}_{trial}_{j}.csv')
            except FileNotFoundError:
                print(f'File {model}_{trial}_{j}.csv does not exist')
    
    files = natsorted(files)
    
    ## DATA FROM LEFT SUBPLOTS
    period = 7*24 # 1 week
    total_periods = 52 # 52 weeks per year (left plot dots per model)
    weekly_rewards = metric_vectors(files, 'reward', total_periods, period)
    weekly_v_violations = metric_vectors(files, 'v_violations', total_periods, period)
    weekly_i_violations = metric_vectors(files, 'i_violations', total_periods, period)
    weekly_losses = metric_vectors(files, 'losses', total_periods, period)
    
    ## DATA FROM RIGHT SUBPLOTS
    period = 24*365 # 1 year
    total_periods = 1 # 1 single dot per model (right plot)
    anual_rewards = metric_vectors(files, 'reward', total_periods, period)
    anual_v_violations = metric_vectors(files, 'v_violations', total_periods, period)
    anual_i_violations = metric_vectors(files, 'i_violations', total_periods, period)
    anual_losses = metric_vectors(files, 'losses', total_periods, period)
    
    return ([weekly_rewards, weekly_v_violations, weekly_i_violations, weekly_losses],
            [anual_rewards, anual_v_violations, anual_i_violations, anual_losses])

def select_color(model):
    if model == 'A2C': return 'tomato'
    if model == 'PPO': return 'green'
    if model == 'TRPO': return 'red'
    if model == 'DDPG': return 'blue'
    if model == 'TD3': return 'purple'
    
def left_plot(subplot, data, model):

    subplot.plot(np.arange(1, len(data)+1, 1), np.array([tup[1] for tup in data]),
                 color = select_color(model), lw = 0.5)
    subplot.fill_between(np.arange(1, len(data)+1, 1),
                         np.array([tup[0] for tup in data]),
                         np.array([tup[2] for tup in data]),
                         color = select_color(model), alpha=0.2)

def right_plot(subplot, data):

    l = np.array([tup[0][1] for tup in data])
    l_lower = np.array([tup[0][0] for tup in data])    
    l_upper = np.array([tup[0][2] for tup in data])
    
    for model in ['A2C', 'PPO', 'TRPO', 'DDPG', 'TD3']:
        lidx = [idx for idx, elem in enumerate(data) if elem[-1] == model]
        subplot.plot(np.arange(min(lidx)+1, max(lidx)+2, 1),
                     l[min(lidx):max(lidx)+1], color = select_color(model), lw = 0.5)
        subplot.fill_between(np.arange(min(lidx)+1, max(lidx)+2, 1),
                             l_lower[min(lidx):max(lidx)+1],
                             l_upper[min(lidx):max(lidx)+1],
                             color = select_color(model), alpha=0.2)
    
def results(fig_name):
    # Weekly subplots on the left, yeaarly sobplots on the right.
    
    # Data
    weekly, anual = accumulated_results()
    
    fig = plt.figure(figsize=(8.3, 10.2))
    gs = gridspec.GridSpec(4,2)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    
    ax5 = fig.add_subplot(gs[4])
    ax6 = fig.add_subplot(gs[5])
    
    ax7 = fig.add_subplot(gs[6])
    ax8 = fig.add_subplot(gs[7])
  
    plot_titles = [
        ('Weekly reward', 'Average annual reward'),
        ('Weekly voltage violations', 'Average annual voltage violations'),
        ('Weekly current violations', 'Average annual current violations'),
        ('Weekly active losses', 'Average annual active losses')
    ]
    ylabels_list = [
        'Reward',
        'Volatge violation [pu]',
        'Current violation ['+r'$\frac{1}{100}\%$]',
        'Active losses [MW]'
    ]
    
    for row, ax in enumerate(zip([ax1, ax3, ax5, ax7], [ax2, ax4, ax6, ax8])):
        
        sub1, sub2 = ax
        
        gl = 0.5 #gruix_linia
        
        sub1.set_title(plot_titles[row][0], fontsize=10) #per un any
        for set_ in weekly[row]:
                left_plot(sub1, set_[:-2], set_[-1])
        sub1.set_ylabel(ylabels_list[row])
        
        sub2.set_title(plot_titles[row][1], fontsize=10)
        right_plot(sub2, anual[row])
        
        if row == 3:
            sub1.set_xlabel('Time [weeks]')
        else:
            plt.setp(sub1.get_xticklabels(), visible=False)
        
    
        if row == 3:
            sub2.set_xlabel("Parameter's sets")
        else:
            plt.setp(sub2.get_xticklabels(), visible=False)
        
    legend_elements = [Line2D([0], [0], color='tomato', label='A2C', lw=1.5),
                       Line2D([0], [0], color='green', label='PPO', lw=1.5),
                       Line2D([0], [0], color='red', label='TRPO', lw=1.5),
                       Line2D([0], [0], color='blue', label='DDPG', lw=1.5),
                       Line2D([0], [0], color='purple', label='TD3', lw=1.5)]
        
    # Put a legend below current axis
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(-0.1, -0.25),
               fancybox=True, shadow=False, ncol=5)
       
    plt.subplots_adjust(top=0.9,bottom=0.1,left=0.10,right=0.95, hspace=0.25, wspace=0.17)
    plt.savefig(fig_name, dpi = 300, bbox_inches = 'tight')
    plt.show()


def results_study(fig_name, no_agent):
    
    fig = plt.figure(figsize=(8.3, 10.2))
    
    dataset = import_data(path_file='dataset.csv')
    demand = dataset.loc[:, ['demand']]
    total_demand = demand.iloc[:365*24, 0].sum()
    
    error = []
    files = []
    for model in tqdm(['A2C', 'PPO', 'TRPO', 'DDPG', 'TD3'], leave = True):
        for trial in tqdm(range(N_TRIALS), leave = False):
            try:
                # PPO had fails for hyperparams incompatibility (?)
                l = np.zeros(N_SEEDS)
                for j in tqdm(range(N_SEEDS), leave =False):
                    f = os.path.join(PATH, f'{model}_{trial}_{j}.csv')
                    ad = pd.read_csv(f).loc[:,['demand']].sum()
                    files.append(f'{model}_{trial}_{j}.csv')
                    l[j] = abs(total_demand-ad)/total_demand*100
                error.append([np.mean(l), model, trial])
            except FileNotFoundError:
                print(f'File {model}_{trial}_{j}.csv does not exist')
    
    period = 24*365 # 1 year
    total_periods = 1
    #anual_rewards = metric_vectors(files, 'reward', total_periods, period)
    i_viol = metric_vectors(files, 'i_violations', total_periods, period)
    v_viol = metric_vectors(files, 'v_violations', total_periods, period)
    losses = metric_vectors(files, 'losses', total_periods, period)    
    
    
    real = no_agent.loc[:, 'i_violations'].values.ravel().mean()
    var_i = np.array([real - tup[0][1] for tup in i_viol])/real*100 # means
    real = no_agent.loc[:, 'v_violations'].values.ravel().mean()
    var_v = np.array([real - tup[0][1] for tup in v_viol])/real*100
    real = no_agent.loc[:, 'losses'].values.ravel().mean()
    var_losses = np.array([real - tup[0][1] for tup in losses])/real*100
    
    
    # zoom range
    #ini_reward, interval_reward = 100, 500
    ini_i, interval_i = -10, 11
    ini_v, interval_v = -0.5, 5
    ini_losses, interval_losses = -1, 2
    ini_error, interval_error = 0., 0.1
    
    ax1 = plt.subplot(321)
    ax2 = plt.subplot(322)
    
    for i in tqdm(range(len(error))):
        ax1.scatter(var_losses[i], error[i][0],
                    color=select_color(error[i][1]), lw=1, marker='.')
        ax2.scatter(var_losses[i], error[i][0],
                    color=select_color(error[i][1]), lw=1, marker='.')
        ax2.annotate(error[i][2], (var_losses[i], error[i][0]),
                     textcoords='offset points', xytext=(2,6))        
        
    x1, x2, y1, y2 = ini_losses, (ini_losses+interval_losses), ini_error, interval_error
    ax2.set_xlim(x1, x2)
    ax2.set_ylim(y1, y2)
    ax1.indicate_inset_zoom(ax2, edgecolor="black", lw = 0.6, alpha = 0.6)
    
    ax1.set_xlabel('Relative active losses [\%]')
    ax1.set_ylabel('Modified demand error [\%]')
    
    
    ax3 = plt.subplot(323)
    ax4 = plt.subplot(324)
    
    for i in tqdm(range(len(i_viol))):
        ax3.scatter(var_v[i], error[i][0], color=select_color(error[i][1]), lw=1, marker='.')
        ax4.scatter(var_v[i], error[i][0], color=select_color(error[i][1]), lw=1, marker='.')
        ax4.annotate(error[i][2], (var_v[i], error[i][0]),
                     textcoords='offset points', xytext=(2,6))
        
    x1, x2, y1, y2 = ini_v, (ini_v+interval_v), ini_error, interval_error
    ax4.set_xlim(x1, x2)
    ax4.set_ylim(y1, y2)
    ax3.indicate_inset_zoom(ax4, edgecolor="black", lw = 0.6, alpha = 0.6)
    
    ax3.set_xlabel('Relative voltage violations [\%]')
    ax3.set_ylabel('Relative error in demand [\%]')
    
    
    ax5 = plt.subplot(325)
    ax6 = plt.subplot(326)
    
    for i in tqdm(range(len(error))):
        ax5.scatter(var_i[i], error[i][0], color=select_color(error[i][1]), lw=1, marker='.')
        ax6.scatter(var_i[i], error[i][0], color=select_color(error[i][1]), lw=1, marker='.')
        ax6.annotate(error[i][2], (var_i[i], error[i][0]),
                     textcoords='offset points', xytext=(2,6))
        
    x1, x2, y1, y2 = ini_i, (ini_i+interval_i), ini_error, interval_error
    ax6.set_xlim(x1, x2)
    ax6.set_ylim(y1, y2)
    ax5.indicate_inset_zoom(ax6, edgecolor="black", lw = 0.6, alpha = 0.6)
    
    ax5.set_xlabel('Relative current violations [\%]')
    ax5.set_ylabel('Relative error in demand [\%]')
    
    
    legend_elements = [
        Line2D([0], [0], color='tomato', label='A2C', lw = 0, marker='o', markersize=5),
        Line2D([0], [0], color='green', label='PPO', lw = 0, marker='o', markersize=5),
        Line2D([0], [0], color='red', label='TRPO', lw = 0, marker='o', markersize=5),
        Line2D([0], [0], color='blue', label='DDPG', lw = 0, marker='o', markersize=5),
        Line2D([0], [0], color='purple', label='TD3', lw = 0, marker='o', markersize=5)
    ]
    # Legend
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(-0.14, -0.2),
               fancybox=True, shadow=False, ncol=5)
            
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.95, hspace=0.2, wspace=0.15)
    plt.savefig(fig_name, dpi = 300, bbox_inches = 'tight')
    
    plt.show()
    


#################################################################################################

def agent_vs_noagent(fig_name, demand):
  
    fig = plt.figure()
    subfigs = fig.subfigures(nrows=1, ncols=1)
    
    axs = subfigs.subplots(nrows=1, ncols=2, sharey=True)    
    files = natsorted([f'{BEST_MODEL}_{i}.csv' for i in range(N_SEEDS)])
    adjusted_demand = metric_vectors(files, 'demand', 365*24, 1)
    
    start = datetime.strptime(START, "%Y-%m-%d")
    end = (start + timedelta(days=DEMAND_PERIOD)).strftime("%Y-%m-%d")
    
    start_index = demand.index.get_loc(' '.join([START,'00:00:00+00:00']))
    end_index = demand.index.get_loc(' '.join([end,'00:00:00+00:00']))
    
    x = demand[start_index:end_index].index[:, None]
    x = x.reshape(1,len(x))[0]
    gl = 1 #lw

    sub1 = axs.flatten()[0]
    sub1.step(x, demand.iloc[start_index:end_index, 0].values.ravel(),
              label='No agent demand',
              color = 'tomato', lw = gl, alpha=1, zorder=1)
              
    set_ = adjusted_demand[0][:-2][start_index:end_index]
    sub1.step(x, np.array([tup[1] for tup in set_]),
              label='Agent demand',
              color='blue', lw = gl, alpha=1, zorder=3)
              
    sub1.fill_between(x, np.array([tup[0] for tup in set_]),
                         np.array([tup[2] for tup in set_]),
                      step="pre",  color='blue', alpha=0.2, zorder=2)
                      
    sub1.set_ylabel('Power [MW]')
    sub1.set_xlabel('Time [days]')
    for tick in sub1.get_xticklabels():
       tick.set_rotation(45)
    
    sub1.legend(loc='best', prop={'size': 6.5})
    
    sub2 = axs.flatten()[1]
    
    set_ = adjusted_demand[0][:-2]
    
    sub2.set_xlabel('Time [days]')
    
    l = np.zeros(N_SEEDS)
    total_demand = demand.iloc[:365*24, 0].sum()
    for j in range(N_SEEDS):
        ad = pd.read_csv(os.path.join(PATH, f'{BEST_MODEL}_{j}.csv')).loc[:,['demand']].sum()
        l[j] = abs(total_demand-ad)/total_demand*100
    print(f"Total error {BEST_MODEL}:", np.mean(l), '%')
    
    corr = crosscorrelation(
        demand.iloc[:365*24, 0].values.ravel(), np.array([tup[1] for tup in set_]), (365*24)//2
    )
    lag = np.where(corr == corr.max())[0][0]-(365*24)//2
    #print(list(corr).index(max(corr)))
    
    print(f"Lag {BEST_MODEL}:", lag)
    # if lag<0, adjusted demand is moved to the right, and viceversa.
    
    demand_lag = demand.iloc[:365*24, 0].values.ravel()
    if lag < 0:
       demand_lag = demand_lag[:-abs(lag)]
       ad_lag = set_[abs(lag):] # adjusted demand with lag
    elif lag > 0:
       demand_lag = demand_lag[lag:]
       ad_lag = set_[:-lag]
    else:
       demand_lag = demand_lag
       ad_lag = set_
    
    days = int(365-np.ceil(abs(lag)/24))
    min_array_d = np.zeros(days)
    max_array_d = np.zeros(days)
    min_array_ad = np.zeros(days)
    max_array_ad = np.zeros(days)    
    min_array_ad_lower = np.zeros(days)
    max_array_ad_upper = np.zeros(days)
    
    for day in range(days):
        min_array_d[day] = demand_lag[24*day:24*(day+1)].min()
        max_array_d[day] = demand_lag[24*day:24*(day+1)].max()
        min_array_ad[day] = np.array([tup[1] for tup in ad_lag])[24*day:24*(day+1)].min()
        max_array_ad[day] = np.array([tup[1] for tup in ad_lag])[24*day:24*(day+1)].max()
        min_array_ad_lower[day] = np.array([tup[0] for tup in ad_lag])[24*day:24*(day+1)].min()
        max_array_ad_upper[day] = np.array([tup[2] for tup in ad_lag])[24*day:24*(day+1)].max()
        
    sub2.fill_between(np.arange(1, DAYS+1, 1),
                      min_array_d[:DAYS],
                      max_array_d[:DAYS],
                      color = 'orange',
                      label='No agent demand',
                      step="mid", alpha=1, zorder=1)
                      
    sub2.fill_between(np.arange(1, DAYS+1, 1),
                      min_array_ad_lower[:DAYS],
                      max_array_ad_upper[:DAYS],
                      color = 'blue',
                      label='95\% confidence interval',
                      step="mid", alpha=0.5, zorder=2)
                      
    sub2.fill_between(np.arange(1, DAYS+1, 1),
                      min_array_ad[:DAYS],
                      max_array_ad[:DAYS],
                      color = 'tomato',
                      label='Agent demand',    
                      step="mid", alpha=0.7, zorder=3)
    
    sub2.legend(loc='upper right', prop={'size': 6.5})
    # daily
    valey = (min_array_ad - min_array_d)/min_array_d*100    
    peak = (max_array_d - max_array_ad)/max_array_d*100
    print(f"Valley filling {BEST_MODEL}:", np.mean(valey), '%')
    print(f"Peak reduction {BEST_MODEL}:", np.mean(peak), '%')
    
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.08, right=0.955, hspace=0.1, wspace=0.1)
    plt.savefig(fig_name, dpi = 300, bbox_inches = 'tight')
    plt.show()

#################################################################################################
 
def performance(fig_name, no_agent, demand):

    fig = plt.figure(figsize=(8.3, 7.2))
    gs = gridspec.GridSpec(3,2)
    
    ax1 = fig.add_subplot(gs[0:2])  
    ax3 = fig.add_subplot(gs[2:4])
    ax5 = fig.add_subplot(gs[4:6])
    
    plots_list = ['v_violations', 'i_violations', 'losses']
    ylabels_list = ['Volatge violation [pu]',
                    'Current violation ['+r'$\frac{1}{100}\%$]',
                    'Active losses [MW]']
    gl = 1 #lw
    
    for row, ax in enumerate([ax1, ax3, ax5]):
        
        sub1 = ax
        
        start = datetime.strptime(START, "%Y-%m-%d")
        end = (start + timedelta(days=DEMAND_PERIOD)).strftime("%Y-%m-%d")
        
        start_index = demand.index.get_loc(' '.join([START,'00:00:00+00:00']))
        end_index = demand.index.get_loc(' '.join([end,'00:00:00+00:00']))
        x = demand[start_index:end_index].index[:, None]
        x = x.reshape(1,len(x))[0]
        
        no_agent_v_viol = no_agent.loc[start_index:end_index-1, plots_list[row]].values.ravel()
        sub1.step(x, no_agent_v_viol, color = 'tomato', lw = gl, alpha=1, zorder=1)
                    
        files = natsorted([f'{BEST_MODEL}_{i}.csv' for i in range(N_SEEDS)])
        adjusted_demand = metric_vectors(files, plots_list[row], 365*24, 1)
        set_ = adjusted_demand[0][:-2][start_index:end_index]
        sub1.step(x, np.array([tup[1] for tup in set_]), color='blue', lw=gl, alpha=1,zorder=3)
        sub1.fill_between(x, np.array([tup[0] for tup in set_]),
                          np.array([tup[2] for tup in set_]),
                          step="pre",  color='blue', alpha=0.2, zorder=2)
                      
        sub1.set_ylabel(ylabels_list[row])
        if row == 0: sub1.set_title('Hourly data CIGRE study case')
        if row == 2:
            sub1.set_xlabel('Time [days]')
            for tick in sub1.get_xticklabels():
                tick.set_rotation(45)
        else:
            plt.setp(sub1.get_xticklabels(), visible=False)
        
        real = no_agent.loc[:, plots_list[row]].values.ravel()
        adjusted = np.array([tup[1] for tup in adjusted_demand[0][:-2]]).sum()
        variation = (real.sum() - adjusted)/real.sum()*100    
        print(f"Relative variation {plots_list[row]}", variation, '%')
        
    legend_elements = [Line2D([0], [0], color='tomato', label='Signal w/o agent', lw=1.5),
                       Line2D([0], [0], color='blue', label='Signal w/ agent', lw=1.5)]
        
    # Put a legend below current axis
    ax1.legend(handles=legend_elements, loc='upper right')
             
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.08, right=0.955, hspace=0.1, wspace=0.1)
    plt.savefig(fig_name, dpi = 300, bbox_inches = 'tight')
    plt.show()
  
#################################################################################################
 
# https://stackoverflow.com/questions/30677241/how-to-limit-cross-correlation-window-width-in-numpy

def _check_arg(x, xname):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('%s must be one-dimensional.' % xname)
    return x

def crosscorrelation(x, y, maxlag):
    """
    Cross correlation with a maximum number of lags.
    `x` and `y` must be one-dimensional numpy arrays with the same length.
    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(x)-maxlag-1:len(x)+maxlag]
    The return vaue has length 2*maxlag + 1.
    """
    x = _check_arg(x, 'x')
    y = _check_arg(y, 'y')
    py = np.pad(y.conj(), 2*maxlag, mode='constant')
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')
    return T.dot(px)/(np.linalg.norm(x)*np.linalg.norm(y))
    
#################################################################################################

def import_data(path_file):    
    df = pd.read_csv(path_file)
    df.index = pd.to_datetime(df.iloc[:, 0], utc=True) + pd.DateOffset(hours=1)
    df.index.name = 'Timestamp'
    df = df.iloc[:, 1:]
    return df

if __name__ == "__main__":

    path = os.path.join(os.getcwd(), PLOTS_PATH)
    os.makedirs(path, exist_ok=True)
    
    dataset = import_data(path_file='dataset.csv')
    demand = dataset.loc[:, ['demand']]
    
    results(f'{PLOTS_PATH}/models_results.png')
    
    no_agent = pd.read_csv('no_agent.csv')    
    results_study(f'{PLOTS_PATH}/comparativa.png', no_agent)
    
    BEST_MODEL = 'TRPO_37' #'A2C_24'
    DEMAND_PERIOD = 7 # days
    agent_vs_noagent(f'{PLOTS_PATH}/{BEST_MODEL}.png', demand)
    
    DEMAND_PERIOD = 14 # days 
    performance(f'{PLOTS_PATH}/performance_best_model.png', no_agent, demand)
