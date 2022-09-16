import numpy as np, pandas as pd, copy, os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from datetime import datetime, timedelta

from natsort import natsorted
from numpy.lib.stride_tricks import as_strided

from matplotlib import gridspec

#graphic's format
pylab.rcParams['axes.labelsize'] = 9
pylab.rcParams['xtick.labelsize'] = 9
pylab.rcParams['ytick.labelsize'] = 9
pylab.rcParams['legend.fontsize'] = 9
pylab.rcParams['font.family'] = 'serif'
pylab.rcParams['font.sans-serif'] = ['Computer Modern Roman']
pylab.rcParams['text.usetex'] = True
pylab.rcParams['figure.figsize'] = 8.3, 10.2

PATH_1_LOAD = '1_load/evaluation/'
PATH_6_LOAD = '6_loads/evaluation/'
PATH_9_LOAD = '9_loads/evaluation/'
PATH_13_LOAD = '13_loads/evaluation/'
PATH_18_LOAD = '18_loads/evaluation/'

PLOTS_PATH = 'plot_results'

START = '2015-06-01'
END = '2015-06-07'

DAYS = 59 # days power range

N_SEEDS = 5

def agent_vs_noagent(fig_name, demand):
  
    fig = plt.figure()
    
    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=5, ncols=1)
    
    path_list = [PATH_1_LOAD, PATH_6_LOAD, PATH_9_LOAD, PATH_13_LOAD, PATH_18_LOAD]
    subtitles = [
        '1 flexible load CIGRE study case demand response',
        '6 flexible loads CIGRE study case demand response',
        '9 flexible loads CIGRE study case demand response',
        '13 flexible loads CIGRE study case demand response',
        '18 flexible loads CIGRE study case demand response',
    ]
    
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(subtitles[row])
        
        axs = subfig.subplots(nrows=1, ncols=2, sharey=True)    
        files = natsorted(os.listdir(path_list[row]))
        adjusted_demand = metric_vectors(path_list[row], files, 'demand', 365*24, 1)
        
        start = datetime.strptime(START, "%Y-%m-%d")
        end = (start + timedelta(days=DEMAND_PERIOD)).strftime("%Y-%m-%d")
        
        start_index = demand.index.get_loc(' '.join([START,'00:00:00+00:00']))
        end_index = demand.index.get_loc(' '.join([end,'00:00:00+00:00']))
	    
        x = demand[start_index:end_index].index[:, None]
        x = x.reshape(1,len(x))[0]
        gl = 1 #lw

        sub1 = axs.flatten()[0]
        sub1.step(x, demand.iloc[start_index:end_index, 0].values.ravel(),
                  #label='No agent demand',
                  color = 'tomato', lw = gl, alpha=1, zorder=1)
		      
        set_ = adjusted_demand[0][:-1][start_index:end_index]
        sub1.step(x, np.array([tup[1] for tup in set_]),
                  #label='Agent demand',
                  color='blue', lw = gl, alpha=1, zorder=3)
		      
        sub1.fill_between(x, np.array([tup[0] for tup in set_]),
                          np.array([tup[2] for tup in set_]),
                          step="pre",  color='blue', alpha=0.2, zorder=2)
		              
        sub1.set_ylabel('Power [MW]')
	    
        if row == 4:
            sub1.set_xlabel('Time [days]')
            for tick in sub1.get_xticklabels():
                tick.set_rotation(45)
        else:
            plt.setp(sub1.get_xticklabels(), visible=False)
        
        sub2 = axs.flatten()[1]
        set_ = adjusted_demand[0][:-1]
        
        if row == 4:
            sub2.set_xlabel('Time [days]')
        else:
            plt.setp(sub2.get_xticklabels(), visible=False)
        
        l = np.zeros(N_SEEDS)
        total_demand = demand.iloc[:365*24, 0].sum()
        for j in range(N_SEEDS):
            ad = pd.read_csv(os.path.join(path_list[row], f'0_{j}.csv')).loc[:,['demand']].sum()
            l[j] = abs(total_demand-ad)/total_demand*100
        print(f"Total error {path_list[row].split('_')[0]}:", np.mean(l), '%')
        
        corr = crosscorrelation(
            demand.iloc[:365*24, 0].values.ravel(), np.array([tup[1] for tup in set_]),
            (365*24)//2
        )           
        lag = np.where(corr == corr.max())[0][0]-(365*24)//2
        #print(list(corr).index(max(corr)))
        
        print(f"Lag {path_list[row].split('_')[0]}:", lag)
	    
        if lag < 0:
            demand_lag = demand.iloc[:365*24, 0].values.ravel()[:-abs(lag)]
            ad_lag = set_[abs(lag):] # adjusted demand with lag
        elif lag > 0:
            demand_lag = demand.iloc[:365*24, 0].values.ravel()[lag:]
            ad_lag = set_[:-lag]
        else:
            demand_lag = demand.iloc[:365*24, 0].values.ravel()
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
                          #label='No agent demand',
                          step="mid", alpha=1, zorder=1)
        
        sub2.fill_between(np.arange(1, DAYS+1, 1),
                          min_array_ad_lower[:DAYS],
                          max_array_ad_upper[:DAYS],
                          color = 'blue',
                          #label='95\% confidence interval',
                          step="mid", alpha=0.5, zorder=2)
        
        sub2.fill_between(np.arange(1, DAYS+1, 1),
                          min_array_ad[:DAYS],
                          max_array_ad[:DAYS],
                          color = 'tomato',
                          #label='Agent demand',    
                          step="mid", alpha=0.7, zorder=3)
        
        # daily
        valey = (min_array_ad - min_array_d)/min_array_d*100    
        peak = (max_array_d - max_array_ad)/max_array_d*100
        print(f"Valley filling {path_list[row].split('_')[0]}:", np.mean(valey), '%')
        print(f"Peak reduction {path_list[row].split('_')[0]}:", np.mean(peak), '%')
        
    legend_elements = [Line2D([0],[0], color='tomato', label='Signal w/o agent', lw=1.5),
           Line2D([0],[0], color='blue', label='Signal w/ agent', lw=1.5),
           Line2D([0],[0], color='orange', label='Range w/o agent', lw=5),
           Line2D([0],[0], color='blue', label='95\% confidence interval', alpha=0.5, lw=5),
           Line2D([0],[0], color='tomato', label='Range w/ agent', alpha=0.7, lw=5)]
    
    # Put a legend below current axis
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(-0.05, -0.5),
               fancybox=True, shadow=False, ncol=5)
             
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.08, right=0.955, hspace=0.1, wspace=0.05)
    plt.savefig(fig_name, dpi = 300, bbox_inches = 'tight')
    plt.show()
    
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
                    
        files = natsorted(os.listdir(PATH_9_LOAD))
        adjusted_demand = metric_vectors(PATH_9_LOAD, files, plots_list[row], 365*24, 1)
        set_ = adjusted_demand[0][:-1][start_index:end_index]
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
        adjusted = np.array([tup[1] for tup in adjusted_demand[0][:-1]]).sum()
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

def compute(x):
    mean = np.mean(x)
    sup_bound = np.mean(x) + np.std(x)/np.sqrt(len(x))*1.96 #conf. int 95% normal distrib.
    inf_bound = np.mean(x) - np.std(x)/np.sqrt(len(x))*1.96 #conf. int 95% normal distrib.
    return (inf_bound, mean, sup_bound)

def metric_vectors(path, files, metric, total_periods, period):
    # Mean and variance throughout a 'total' period, times 'total_period'.
    
    temp = [[] for _ in range(N_SEEDS)]
    #np.zeros((N_SEEDS, total_periods)) with tuples as elements
    array = [[] for _ in range(len(files)//N_SEEDS)]
    #np.zeros((len(files)//N_SEEDS, total_periods + 1)) with tuples as elements
    k = 0
    for s in range(len(files)//N_SEEDS): # n_sets
        for j in range(N_SEEDS):            
            file_path = os.path.join(path, files[k+j])
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
    
    DEMAND_PERIOD = 7 # days
    agent_vs_noagent(f"{PLOTS_PATH}/agent_vs_noagent_cigre.png", demand)
    
    DEMAND_PERIOD = 14 # days
    no_agent = pd.read_csv('no_agent.csv')
    performance(f"{PLOTS_PATH}/performance_cigre.png", no_agent, demand)
