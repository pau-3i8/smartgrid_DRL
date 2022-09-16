#cython: language_level = 3
#cython: boundscheck = False
#cython: wraparound = False
#cython: nonecheck = False
#cython: cdivision = False
#cython: profile = True
#cython: initializedcheck = True

"""
to compile: $ python3 setup.py build_ext --inplace
for optimization checking: $ cython -a file.pyx
"""

from cython.parallel import parallel, prange
cimport numpy as np
import numpy as np
cimport cython

##################################### ACTIVATION FUNCTIONS ######################################

@cython.profile(False)
cdef inline double haar(double x) nogil:
    if 0<= x <= 1: return 1
    return 0

@cython.profile(False)
cdef inline double hat(double x) nogil: 
    if 0 <= x <= 2:
        if 0 <= x <= 1: return x
        elif 1 < x <= 2: return -x+2
    return 0
    
@cython.profile(False)
cdef inline double quadratic(double x) nogil:
    if 0 <= x <= 3:
        # x*x is faster than x**2
        if 0<= x <= 1: return 1/2*x*x 
        elif 1 < x <= 2: return -x*x+3*x-3/2
        elif 2 < x <= 3: return 1/2*x*x-3*x+9/2
    return 0

@cython.profile(False)
cdef inline double bicubic(double x) nogil:
    if 0 <= x <= 4:
        if 0 <= x <= 1: return 1/6*x*x*x
        elif 1 < x <= 2: return -1/2*x*x*x+2*x*x-2*x+2/3
        elif 2 < x <= 3: return 1/2*x*x*x-4*x*x+10*x-22/3
        elif 3 < x <= 4: return -1/6*x*x*x+2*x*x-8*x+32/3
    return 0

# select_phi_scaled
@cython.profile(False)
def phi(unicode name, double x, size_t n, size_t n_sf):
    if name == 'quadratic':
        return quadratic(sup(x, n, 3, n_sf))
    elif name == 'bicubic':
        return bicubic(sup(x, n, 4, n_sf))
    elif name == 'hat':
        return hat(sup(x, n, 2, n_sf))
    elif name == 'haar':
        return haar(sup(x, n, 1, n_sf))
     
# select_psi
@cython.profile(False)
def psi(unicode name, double x, size_t n, size_t n_sf):
    if name == 'quadratic':
        return 1/4*quadratic(2*sup(x, n, 3, n_sf)) - 3/4*quadratic(2*sup(x, n, 3, n_sf)-1) + 3/4*quadratic(2*sup(x, n, 3, n_sf)-2) - 1/4*quadratic(2*sup(x, n, 3, n_sf)-3)
    elif name == 'bicubic':
        return 1/8*bicubic(2*sup(x, n, 4, n_sf))-1/2*bicubic(2*sup(x, n, 4, n_sf)-1)+3/4*bicubic(2*sup(x, n, 4, n_sf)-2)-1/2*bicubic(2*sup(x, n, 4, n_sf)-3)+1/8*bicubic(2*sup(x, n, 4, n_sf)-4)
    elif name == 'hat':
        return 1/2*hat(2*sup(x, n, 2, n_sf))-hat(2*sup(x, n, 2, n_sf)-1)+1/2*hat(2*sup(x, n, 2, n_sf)-2)
    elif name == 'haar':
        return haar(2*sup(x, n, 1, n_sf))-haar(2*sup(x, n, 1, n_sf)-1)
    
@cython.profile(False)
cdef inline double sup(double x, size_t n, size_t d, size_t n_sf) nogil:
    if n_sf <= 1: return (x-1/2 + (n+1)/(n_sf+1))*d
    if n_sf > 1: return (x-1/2 + n/(n_sf-1))*d
    
#################################### 3 INPUTS WAVENET BASIS #####################################

# For arrays np.ndarray[dtype_t, ndim=1, mode = 'c'] --> dtype_t[::1].

def matriu_2D(object param, double[::1] input_1 not None, double[::1] input_2 not None,
              double[::1] Iapp not None, size_t wavelons):
              
    cdef str sf_name = param['fscale']
    cdef size_t n_sf = param['n_sf']
    cdef long int res = param['resolution']
    cdef size_t N = input_1.shape[0], i = 0, j, n1, n2, n3
    cdef long int c1, c2, c3, m
    
    cdef double[:,::1] matriu = np.zeros((wavelons, N), dtype = np.double)
    
    ## Linear component
    if param['bool_lineal']:
        i = 2
        matriu[0,:] = input_1
        matriu[1,:] = input_2
    
    ## V0
    for n1 in range(n_sf):
        for n2 in range(n_sf):
            for n3 in range(n_sf):
                for j in range(N):
                    matriu[i, j] = phi(sf_name, input_1[j], n1, n_sf)* phi(sf_name, input_2[j], n2, n_sf)*phi(sf_name, Iapp[j], n3, n_sf)
                i+=1

    ## W0 and beyond
    for m in range(res+1):
        # K's are eq. 6.11 TFG
        for n1 in range(n_sf):
            for n2 in range(n_sf):
                for n3 in range(n_sf):
                    for c1 in range(2**m): #K3
                        for j in range(N):
                            matriu[i, j] = phi(sf_name, input_1[j], n1, n_sf)* phi(sf_name, input_2[j], n2, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c1, n3, n_sf)
                        i+=1
                    for c1 in range(2**m):
                        for j in range(N):
                            matriu[i, j] = phi(sf_name, Iapp[j], n1, n_sf)* phi(sf_name, input_1[j], n2, n_sf)* psi(sf_name, (2**m)* input_2[j] - c1, n3, n_sf)
                        i+=1
                    for c1 in range(2**m):
                        for j in range(N):
                            matriu[i, j] = phi(sf_name, input_2[j], n1, n_sf)* phi(sf_name, Iapp[j], n2, n_sf)* psi(sf_name, (2**m)* input_1[j] - c1, n3, n_sf)
                        i+=1
                    for c1 in range(2**m): #K2
                        for c2 in range(2**m):
                            for j in range(N):
                                matriu[i, j] = phi(sf_name, input_1[j], n1, n_sf)* psi(sf_name, (2**m)* input_2[j] - c1, n2, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c2, n3, n_sf)
                            i+=1
                    for c1 in range(2**m):
                        for c2 in range(2**m):
                            for j in range(N):
                                matriu[i, j] = phi(sf_name, Iapp[j], n1, n_sf)* psi(sf_name, (2**m)* input_1[j] - c1, n2, n_sf)* psi(sf_name, (2**m)* input_2[j] - c2, n3, n_sf)
                            i+=1
                    for c1 in range(2**m):
                        for c2 in range(2**m):
                            for j in range(N):
                                matriu[i, j] = phi(sf_name, input_2[j], n1, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c1, n2, n_sf)* psi(sf_name, (2**m)* input_1[j] - c2, n3, n_sf)
                            i+=1
                    for c1 in range(2**m): #K1
                        for c2 in range(2**m):
                            for c3 in range(2**m):
                                for j in range(N):
                                    matriu[i, j] = psi(sf_name, (2**m)* input_1[j] - c1, n1, n_sf)* psi(sf_name, (2**m)* input_2[j] - c2, n2, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c3, n3, n_sf)
                                i+=1
    return np.asarray(matriu).T

#################################### 4 INPUTS WAVENET BASIS #####################################
    
def matriu_3D(object param, double[::1] input_1 not None, double[::1] input_2 not None,
              double[::1] input_3 not None, double[::1] Iapp not None, size_t wavelons):
              
    cdef str sf_name = param['fscale']
    cdef size_t n_sf = param['n_sf']
    cdef long int res = param['resolution']
    cdef size_t N = input_1.shape[0], i = 0, j, n1, n2, n3, n4
    cdef long int c1, c2, c3, c4, m
    
    cdef double[:,::1] matriu = np.zeros((wavelons, N), dtype = np.double)
    
    if param['bool_lineal']:
        i = 3
        matriu[0,:] = input_1
        matriu[1,:] = input_2
        matriu[2,:] = input_3
    
    ## V0
    for n1 in range(n_sf):
        for n2 in range(n_sf):
            for n3 in range(n_sf):
                for n4 in range(n_sf):
                    for j in range(N):
                        matriu[i, j] = phi(sf_name, input_1[j], n1, n_sf)* phi(sf_name, input_2[j], n2, n_sf)* phi(sf_name, input_3[j], n3, n_sf)* phi(sf_name, Iapp[j], n4, n_sf)
                    i+=1

    ## W0 and beyond
    for m in range(res+1):
        # K's are eq. 6.11 TFG
        for n1 in range(n_sf):
            for n2 in range(n_sf):
                for n3 in range(n_sf):
                    for n4 in range(n_sf):
                        for c1 in range(2**m): #K4
                            for j in range(N):
                                matriu[i, j] = phi(sf_name, input_2[j], n1, n_sf)* phi(sf_name, input_1[j], n2, n_sf)* phi(sf_name, input_3[j], n3, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c1, n4, n_sf)
                            i+=1
                        for c1 in range(2**m):
                            for j in range(N):
                                matriu[i, j] = phi(sf_name, Iapp[j], n1, n_sf)* phi(sf_name, input_2[j], n2, n_sf)* phi(sf_name, input_1[j], n3, n_sf)* psi(sf_name, (2**m)* input_3[j] - c1, n4, n_sf)
                            i+=1
                        for c1 in range(2**m):
                            for j in range(N):
                                matriu[i, j] = phi(sf_name, input_3[j], n1, n_sf)* phi(sf_name, Iapp[j], n2, n_sf)* phi(sf_name, input_2[j], n3, n_sf)* psi(sf_name, (2**m)* input_1[j] - c1, n4, n_sf)
                            i+=1
                        for c1 in range(2**m):
                            for j in range(N):
                                matriu[i, j] = phi(sf_name, input_1[j], n1, n_sf)* phi(sf_name, input_3[j], n2, n_sf)* phi(sf_name, Iapp[j], n3, n_sf)* psi(sf_name, (2**m)* input_2[j] - c1, n4, n_sf)
                            i+=1
                        for c1 in range(2**m): #K3
                            for c2 in range(2**m):
                                for j in range(N):
                                    matriu[i, j] = phi(sf_name, input_2[j], n1, n_sf)* phi(sf_name, input_1[j], n2, n_sf)* psi(sf_name, (2**m)* input_3[j] - c1, n3, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c2, n4, n_sf)
                                i+=1
                        for c1 in range(2**m):
                            for c2 in range(2**m):
                                for j in range(N):
                                    matriu[i, j] = phi(sf_name, Iapp[j], n1, n_sf)* phi(sf_name, input_2[j], n2, n_sf)* psi(sf_name, (2**m)* input_1[j] - c1, n3, n_sf)* psi(sf_name, (2**m)* input_3[j] - c2, n4, n_sf)
                                i+=1
                        for c1 in range(2**m):
                            for c2 in range(2**m):
                                for j in range(N):
                                    matriu[i, j] = phi(sf_name, input_3[j], n1, n_sf)* phi(sf_name, Iapp[j], n2, n_sf)* psi(sf_name, (2**m)* input_2[j] - c1, n3, n_sf)* psi(sf_name, (2**m)* input_1[j] - c2, n4, n_sf)
                                i+=1
                        for c1 in range(2**m):
                            for c2 in range(2**m):
                                for j in range(N):
                                    matriu[i, j] = phi(sf_name, input_1[j], n1, n_sf)* phi(sf_name, input_3[j], n2, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c1, n3, n_sf)* psi(sf_name, (2**m)* input_2[j] - c2, n4, n_sf)
                                i+=1
                        for c1 in range(2**m):
                            for c2 in range(2**m):
                                for j in range(N):
                                    matriu[i, j] = phi(sf_name, input_3[j], n1, n_sf)* phi(sf_name, input_2[j], n2, n_sf)* psi(sf_name, (2**m)* input_1[j] - c1, n3, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c2, n4, n_sf)
                                i+=1
                        for c1 in range(2**m):
                            for c2 in range(2**m):
                                for j in range(N):
                                    matriu[i, j] = phi(sf_name, input_1[j], n1, n_sf)* phi(sf_name, Iapp[j], n2, n_sf)* psi(sf_name, (2**m)* input_3[j] - c1, n3, n_sf)* psi(sf_name, (2**m)* input_2[j] - c2, n4, n_sf)
                                i+=1
                        for c1 in range(2**m): #K2
                            for c2 in range(2**m):
                                for c3 in range(2**m):
                                    for j in range(N):
                                        matriu[i, j] = psi(sf_name, (2**m)* input_2[j] - c1, n1, n_sf)* psi(sf_name, (2**m)* input_1[j] - c2, n2, n_sf)* psi(sf_name, (2**m)* input_3[j] - c3, n3, n_sf)* phi(sf_name, Iapp[j], n4, n_sf)
                                    i+=1
                        for c1 in range(2**m):
                            for c2 in range(2**m):
                                for c3 in range(2**m):
                                    for j in range(N):
                                        matriu[i, j] = psi(sf_name, (2**m)* Iapp[j] - c1, n1, n_sf)* psi(sf_name, (2**m)* input_2[j] - c2, n2, n_sf)* psi(sf_name, (2**m)* input_1[j] - c3, n3, n_sf)* phi(sf_name, input_3[j], n4, n_sf)
                                    i+=1
                        for c1 in range(2**m):
                            for c2 in range(2**m):
                                for c3 in range(2**m):
                                    for j in range(N):
                                        matriu[i, j] = psi(sf_name, (2**m)* input_3[j] - c1, n1, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c2, n2, n_sf)* psi(sf_name, (2**m)* input_2[j] - c3, n3, n_sf)* phi(sf_name, input_1[j], n4, n_sf)
                                    i+=1
                        for c1 in range(2**m):
                            for c2 in range(2**m):
                                for c3 in range(2**m):
                                    for j in range(N):
                                        matriu[i, j] = psi(sf_name, (2**m)* input_1[j] - c1, n1, n_sf)* psi(sf_name, (2**m)* input_3[j] - c2, n2, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c3, n3, n_sf)* phi(sf_name, input_2[j], n4, n_sf)
                                    i+=1
                        for c1 in range(2**m): #K1
                            for c2 in range(2**m):
                                for c3 in range(2**m):
                                    for c4 in range(2**m):
                                        for j in range(N):
                                            matriu[i, j] = psi(sf_name, (2**m)* input_2[j] - c1, n1, n_sf)* psi(sf_name, (2**m)* input_1[j] - c2, n2, n_sf)* psi(sf_name, (2**m)* input_3[j] - c3, n3, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c4, n4, n_sf)
                                        i+=1
    
    return np.asarray(matriu).T
