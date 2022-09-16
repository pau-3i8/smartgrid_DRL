import numpy as np, pandas as pd, copy, os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from datetime import datetime, timedelta

from natsort import natsorted

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

DEMAND_PERIOD = 7 # days
BEST_MODEL = '506'

N_SEEDS = 5
DAYS = 90

def agent_vs_noagent(fig_name, demand):

    files=[]
    for i in range(N_SEEDS):
        files.append(f'{BEST_MODEL}_{i}.csv')
    
    files = natsorted(files)
    adjusted_demand = metric_vectors(files, 'demand', 365*24, 1)
    
    start = datetime.strptime(START, "%Y-%m-%d")
    end = (start + timedelta(days=DEMAND_PERIOD)).strftime("%Y-%m-%d")
    
    start_index = demand.index.get_loc(' '.join([START,'00:00:00+00:00']))
    end_index = demand.index.get_loc(' '.join([end,'00:00:00+00:00']))
    
    x = demand[start_index:end_index].index[:, None]
    x = x.reshape(1,len(x))[0]
    gl = 1 #lw

    fig = plt.figure(1)
    plt.step(x, demand.iloc[start_index:end_index, 0].values.ravel(),
             label='No agent demand', color = 'tomato', lw = gl, alpha=1, zorder=1)
    
    for i, set_ in enumerate(adjusted_demand):
        set_ = set_[:-1][start_index:end_index]
        plt.step(x, np.array([tup[1] for tup in set_]),
                 label='Agent demand', color='blue', lw = gl, alpha=1, zorder=3)
        plt.fill_between(x,
                         np.array([tup[0] for tup in set_]),
                         np.array([tup[2] for tup in set_]),
                         step="pre",  color='blue', alpha=0.2, zorder=2)
        
    l = np.zeros(N_SEEDS)
    total_demand = demand.iloc[:365*24, 0].sum()
    for i in range(N_SEEDS):
        ad = pd.read_csv(os.path.join(PATH, f'{BEST_MODEL}_{i}.csv')).loc[:,['demand']].sum()
        l[i] = abs(total_demand-ad)/total_demand*100
    print(f'Total error {BEST_MODEL}:', np.mean(l))
    
    plt.xlabel('Time [days]')
    plt.xticks(rotation=45)
    plt.ylabel('Power [MW]')
    #plt.title('Representation of the agent in demand curves')
    plt.legend(loc='best')
    
    plt.savefig(fig_name, dpi = 300, bbox_inches = 'tight')
    
    plt.figure(2)
    
    min_array_d = np.zeros(365)
    max_array_d = np.zeros(365)
    min_array_ad = np.zeros(365)
    max_array_ad = np.zeros(365)    
    min_array_ad_lower = np.zeros(365)
    max_array_ad_upper = np.zeros(365)
    set_ = adjusted_demand[0][:-1]
    
    for day in range(365):
        min_array_d[day] = demand.iloc[:, 0].values.ravel()[24*day:24*(day+1)].min()
        max_array_d[day] = demand.iloc[:, 0].values.ravel()[24*day:24*(day+1)].max()
        min_array_ad[day] = np.array([tup[1] for tup in set_])[24*day:24*(day+1)].min()
        max_array_ad[day] = np.array([tup[1] for tup in set_])[24*day:24*(day+1)].max()
        min_array_ad_lower[day] = np.array([tup[0] for tup in set_])[24*day:24*(day+1)].min()
        max_array_ad_upper[day] = np.array([tup[2] for tup in set_])[24*day:24*(day+1)].max()
            
    plt.fill_between(np.arange(1, DAYS+1, 1),
                      min_array_d[:DAYS],
                      max_array_d[:DAYS],
                      color = 'orange',
                      label='No agent demand',
                      step="mid", alpha=1, zorder=1)
                      
    plt.fill_between(np.arange(1, DAYS+1, 1),
                      min_array_ad_lower[:DAYS],
                      max_array_ad_upper[:DAYS],
                      color = 'blue',
                      label='95\% confidence interval',
                      step="mid", alpha=0.5, zorder=2)
                      
    plt.fill_between(np.arange(1, DAYS+1, 1),
                      min_array_ad[:DAYS],
                      max_array_ad[:DAYS],
                      color = 'tomato',
                      label='Agent demand',    
                      step="mid", alpha=0.7, zorder=3)
    
    # els màxims i minims són diaris.
    valey = (min_array_ad - min_array_d)/min_array_d*100    
    peak = (max_array_d - max_array_ad)/max_array_d*100
    print(f"CORREGIT Valley filling {BEST_MODEL}:", np.mean(valey), '%')
    print(f"CORREGIT Peak reduction {BEST_MODEL}:", np.mean(peak), '%')
    
    plt.xlabel('Time [days]')
    plt.ylabel('Power [MW]')
    #plt.title('Power range')
    plt.legend(loc='best')
    
    plt.savefig(f"{PLOTS_PATH}/power_range.png", dpi = 300, bbox_inches = 'tight')
    plt.show()

#################################################################################################

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
            df = pd.read_csv(file_path)
            for i in range(total_periods):
                temp[j].append(compute(df.loc[period*i:period*(i+1), metric].values.ravel()))
        for i in range(total_periods):
            array[s].append((np.mean([metric[0] for metric in [seed[i] for seed in temp]]), 
                             np.mean([metric[1] for metric in [seed[i] for seed in temp]]),
                             np.mean([metric[2] for metric in [seed[i] for seed in temp]])))
        
        temp = [[] for _ in range(N_SEEDS)]
        array[s].append([files[k+j].split('_')[0]])
        # trial = set
        k += j+1
    # array = [set1 = [(week1_), (week_2), ... [set1_dict]]
    #          set2 = [(week1_), (week_2), ... [set2_dict]]]
    return array

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
    agent_vs_noagent(f"{PLOTS_PATH}/agent_vs_noagent.png", demand)
