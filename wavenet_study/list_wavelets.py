from activation_functions import psi, phi
import numpy as np

from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

#graphic's format
pylab.rcParams['axes.labelsize'] = 11
pylab.rcParams['xtick.labelsize'] = 9
pylab.rcParams['ytick.labelsize'] = 9
pylab.rcParams['legend.fontsize'] = 9
pylab.rcParams['font.family'] = 'serif'
pylab.rcParams['font.sans-serif'] = ['Computer Modern Roman']
pylab.rcParams['text.usetex'] = True
pylab.rcParams['figure.figsize'] = 8.3, 10.2

if __name__ == "__main__":

    x = np.linspace(0,1,1000)
    phi_haar = np.zeros(1000)
    phi_hat = np.zeros(1000)
    phi_quadratic = np.zeros(1000)
    phi_cubic = np.zeros(1000)
    psi_haar = np.zeros(1000)
    psi_hat = np.zeros(1000)
    psi_quadratic = np.zeros(1000)
    psi_cubic = np.zeros(1000)
    
    for i in range(len(x)):
        phi_haar[i] = phi('haar', x[i], 0, 1)
        psi_haar[i] = psi('haar', x[i], 0, 1)
        phi_hat[i] = phi('hat', x[i], 0, 1)
        psi_hat[i] = psi('hat', x[i], 0, 1)
        phi_quadratic[i] = phi('quadratic', x[i], 0, 1)
        psi_quadratic[i] = psi('quadratic', x[i], 0, 1)
        phi_cubic[i] = phi('cubic', x[i], 0, 1)
        psi_cubic[i] = psi('cubic', x[i], 0, 1)

    plt.subplot(421)
    plt.title('Haar scale function')
    plt.xlabel('x')
    plt.ylabel('$\phi(x)$')
    plt.plot(x, phi_haar, color='blue', marker=',', linestyle='-')
    
    plt.yticks([1])
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.xticks(np.linspace(0, 1, 5))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

    plt.subplot(422)
    plt.title('Haar wavelet')
    plt.xlabel('x')
    plt.ylabel('$\psi(x)$')
    plt.plot(x, psi_haar, color='blue', marker=',', linestyle='-')
    
    plt.xticks(np.linspace(0, 1, 5))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    
    plt.subplot(423)
    plt.title('Hat scale function')
    plt.xlabel('x')
    plt.ylabel('$\phi(x)$')
    plt.plot(x*2, phi_hat, color='blue', marker=',', linestyle='-')
    
    plt.xticks(np.linspace(0, 2, 5))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

    plt.subplot(424)
    plt.title('Hat wavelet')
    plt.xlabel('x')
    plt.ylabel('$\psi(x)$')
    plt.plot(x*2, psi_hat, color='blue', marker=',', linestyle='-')
    
    plt.xticks(np.linspace(0, 2, 5))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    
    plt.subplot(425)
    plt.title('Quadratic spline scale function')
    plt.xlabel('x')
    plt.ylabel('$\phi(x)$')
    plt.plot(x*3, phi_quadratic, color='blue', marker=',', linestyle='-')
    
    plt.xticks(np.linspace(0, 3, 5))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

    plt.subplot(426)
    plt.title('Quadratic spline wavelet')
    plt.xlabel('x')
    plt.ylabel('$\psi(x)$')
    plt.plot(x*3, psi_quadratic, color='blue', marker=',', linestyle='-')
    
    plt.xticks(np.linspace(0, 3, 5))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    
    plt.subplot(427)
    plt.title('Cubic spline scale function')
    plt.xlabel('x')
    plt.ylabel('$\phi(x)$')
    plt.plot(x*4, phi_cubic, color='blue', marker=',', linestyle='-')
    
    plt.xticks(np.linspace(0, 4, 5))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

    plt.subplot(428)
    plt.title('Cubic spline wavelet')
    plt.xlabel('x')
    plt.ylabel('$\psi(x)$')
    plt.plot(x*4, psi_cubic, color='blue', marker=',', linestyle='-')
    
    plt.xticks(np.linspace(0, 4, 5))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.95, hspace=0.55, wspace=0.35)
    plt.savefig('wavelets.png', bbox_inches='tight', dpi=300)
    plt.show()
