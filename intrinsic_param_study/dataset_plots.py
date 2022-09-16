import numpy as np, pandas as pd, copy, os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from datetime import datetime, timedelta

from custom_env import data_augmentation

#graphic's format
pylab.rcParams['axes.labelsize'] = 9
pylab.rcParams['xtick.labelsize'] = 9
pylab.rcParams['ytick.labelsize'] = 9
pylab.rcParams['legend.fontsize'] = 9
pylab.rcParams['font.family'] = 'serif'
pylab.rcParams['font.sans-serif'] = ['Computer Modern Roman']
pylab.rcParams['text.usetex'] = True
pylab.rcParams['figure.figsize'] = 7.3, 4.2

PLOTS_PATH = 'plot_results'

START = '2015-01-01'
END = '2015-01-07'

DEMAND_PERIOD = 7 # days

def demand_and_production(fig_name, demand, demand_noise, solar, solar_noise):

    x = demand.loc[START:END].index[:,None]
    gl = 1 # line width

    fig, _ = plt.subplots(figsize=(9.3, 4.2))
    #fig.suptitle('Representation of data augmentation in demand and solar generation curves')
    ax1 = plt.subplot(121)
    ax1.step(x, demand.loc[START:END, 'demand'].values.ravel(),
             color = 'tomato', lw = gl, alpha=1, zorder=1)
    ax1.step(x, demand_noise.loc[START:END, 'demand'].values.ravel(), 
             color = 'green', lw = gl, alpha=0.7, zorder=2)
    ax1.set_xlabel('Time [days]')
    ax1.set_ylabel('Power [MW]')
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)
    
    ax2 = plt.subplot(122)
    ax2.step(x, solar.loc[START:END, 'gen_solar'].values.ravel(),
             color = 'red', lw = gl, alpha=1, zorder=1)
    ax2.step(x, solar_noise.loc[START:END, 'gen_solar'].values.ravel(),
             color = 'blue', lw = gl, alpha=0.6, zorder=2)
    ax2.set_xlabel('Time [days]')
    ax2.set_ylabel('Power [MW]')
    for tick in ax2.get_xticklabels():
        tick.set_rotation(45)
    
    legend_elements = [Line2D([0], [0], color='tomato', label='Real demand', lw=1.5),
                       Line2D([0], [0], color='green', label='Noisy demand', lw=1.5),
                       Line2D([0], [0], color='red', label='Real solar generation', lw=1.5),
                       Line2D([0], [0], color='blue', label='Noisy solar generation', lw=1.5)]
    
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(-0.13, -0.3),
               fancybox=True, shadow=False, ncol=4)
    
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.95, hspace=0.5, wspace=0.25)
    plt.savefig(fig_name, dpi = 300, bbox_inches = 'tight')
    plt.show()

def irradiation_and_solar(fig_name, ir, ir_noise, solar, solar_noise):

    x = demand.loc[START:END].index[:,None]
    gl = 1 # lw

    fig, ax1 = plt.subplots()
    ax1.step(x, ir.loc[START:END, 'ir'].values.ravel(), color = 'tomato', lw = gl, alpha=1, zorder=1)
    ax1.step(x, ir_noise.loc[START:END, 'ir'].values.ravel(), color = 'green', lw = gl, alpha=0.6, zorder=2)
    
    ax2 = ax1.twinx()
    ax2.step(x, solar.loc[START:END, 'gen_solar'].values.ravel(), color = 'red', lw = gl, alpha=1, zorder=1)
    ax2.step(x, solar_noise.loc[START:END, 'gen_solar'].values.ravel(), color = 'blue', lw = gl, alpha=0.6, zorder=2)
    
    legend_elements = [Line2D([0], [0], color='tomato', label='Real irradiation', lw=1.5),
                       Line2D([0], [0], color='green', label='Noisy irradiation', lw=1.5),
                       Line2D([0], [0], color='red', label='Real solar generation', lw=1.5),
                       Line2D([0], [0], color='blue', label='Noisy solar generation', lw=1.5)]
    ax1.set_xlabel('Time [days]')
    ax1.set_ylabel('Power [MWh]')    
    ax2.set_ylabel('Irradiation [w/$m^2$]')
    #plt.title('Irradiation and solar generation curves')
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)
    
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.3),
               fancybox=True, shadow=False, ncol=4)
    
    plt.savefig(fig_name, dpi = 300, bbox_inches = 'tight')
    plt.show()

if __name__ == "__main__":

    path = os.path.join(os.getcwd(), PLOTS_PATH)
    os.makedirs(path, exist_ok=True)
    
    dataset = data_augmentation(path_file='dataset.csv', eval_=False)
    demand_noise = dataset.loc[:, ['demand']]
    solar_noise = dataset.loc[:, ['gen_solar']]
    ir_noise = dataset.loc[:, ['ir']]
    
    dataset = data_augmentation(path_file='dataset.csv', eval_=True)
    demand = dataset.loc[:, ['demand']]
    solar = dataset.loc[:, ['gen_solar']]
    ir = dataset.loc[:, ['ir']]
    
    demand_and_production(f"{PLOTS_PATH}/dem_&_prod.png", demand,demand_noise, solar,solar_noise)
    irradiation_and_solar(f"{PLOTS_PATH}/ir_&_solar.png", ir, ir_noise, solar, solar_noise)
