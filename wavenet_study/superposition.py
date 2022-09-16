from activation_functions import psi, phi
import numpy as np
### GRAPHICS
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
pylab.rcParams['figure.figsize'] = 7.3, 4.2

    
def graphic(l, x):
            
    y1, y2 = l[0], l[1]
    y3 = l[2], l[4], l[6]
    y4 = l[3], l[5], l[7]
    
    plt.figure(1)
    
    plt.subplot(221)
    plt.title('One scale function')
    plt.xlabel('x')
    plt.ylabel('$\phi(x)$')
    plt.plot(x*4, y1, color='blue', marker=',', linestyle='-')

    plt.subplot(222)
    plt.title('One wavelet')
    plt.xlabel('x')
    plt.ylabel('$\psi(x)$')
    plt.plot(x*4, y2, color='blue', marker=',', linestyle='-')
    
    plt.subplot(223)
    plt.title('Three superposed scale functions')
    plt.xlabel('x')
    plt.ylabel('$\phi(x)$')
    for elem in y3:
        for i,e in enumerate(elem):
            if e==0:
                elem[i]=None
    """
        plt.plot(x, elem, color='blue', marker=',', linestyle='-')
    """
    plt.plot(x*4, y3[0], color='b', marker=',', linestyle='-')
    plt.plot(x*4, y3[1], color='g', marker=',', linestyle='-')
    plt.plot(x*4, y3[2], color='r', marker=',', linestyle='-')

    plt.subplot(224)
    plt.title('Three superposed wavelets')
    plt.xlabel('x')
    plt.ylabel('$\psi(x)$')
    for elem in y4:
        for i,e in enumerate(elem):
            if e==0:
                elem[i]=None
    """
        plt.plot(x, elem, color='blue', marker=',', linestyle='-')
    """
    plt.plot(x*4, y4[0], color='b', marker=',', linestyle='-')
    plt.plot(x*4, y4[1], color='g', marker=',', linestyle='-')
    plt.plot(x*4, y4[2], color='r', marker=',', linestyle='-')
    
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.95, hspace=0.55, wspace=0.35)
    plt.savefig('superposition.png', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":

    xi = np.linspace(0,1,1000)
    y1 = np.zeros(1000)
    y2 = np.zeros(1000)
    l = []
    for i in range(len(y1)):
        y1[i] = phi('cubic', xi[i], 0, 1)
        y2[i] = psi('cubic', xi[i], 0, 1)
    l.append(y1)
    l.append(y2)
    
    for j in range(3):
        y1 = np.zeros(1000)
        y2 = np.zeros(1000)
        for i in range(len(y1)):
            y1[i] = phi('cubic', xi[i], j, 3)
            y2[i] = psi('cubic', xi[i], j, 3)
        l.append(y1)
        l.append(y2)
    graphic(l, xi)
