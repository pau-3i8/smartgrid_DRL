import numpy as np, pandas as pd, copy, os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime, timedelta

#graphic's format
pylab.rcParams['axes.labelsize'] = 9
pylab.rcParams['xtick.labelsize'] = 9
pylab.rcParams['ytick.labelsize'] = 9
pylab.rcParams['legend.fontsize'] = 9
pylab.rcParams['font.family'] = 'serif'
pylab.rcParams['font.sans-serif'] = ['Computer Modern Roman']
pylab.rcParams['text.usetex'] = True
pylab.rcParams['figure.figsize'] = 7.3, 4.2

def avg_day(x):
    total_hours = len(x)
    total_days = total_hours//24
    mean_day = np.zeros(24)
    for hour in range(24):
        for i in range(total_days):
            mean_day[hour] += x[24*i+hour]
    return mean_day/total_days

actions = pd.read_csv('actions.csv')
actions.drop(['Unnamed: 0'], inplace=True, axis=1)
n_plots = int(len(actions.columns))

# compute the number of rows and columns
n_cols = int(np.sqrt(n_plots))
n_rows = int(np.ceil(n_plots / n_cols))

# setup the plot
gs = gridspec.GridSpec(n_rows, n_cols)
scale = max(n_cols, n_rows)
fig = plt.figure(figsize=(5 * scale, 5 * scale))

# loop through each subplot and plot values there
for i in range(n_plots):
  ax = fig.add_subplot(gs[i])
  y = avg_day(actions.iloc[:, i])
  ax.plot(np.arange(1, 24+1, 1), y)

plt.savefig('actions.png', dpi = 300, bbox_inches = 'tight')
plt.show()
