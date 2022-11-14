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
cdef inline double cubic(double x) nogil:
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
    elif name == 'cubic':
        return cubic(sup(x, n, 4, n_sf))
    elif name == 'hat':
        return hat(sup(x, n, 2, n_sf))
    elif name == 'haar':
        return haar(sup(x, n, 1, n_sf))
     
# select_psi
@cython.profile(False)
def psi(unicode name, double x, size_t n, size_t n_sf):
    if name == 'quadratic':
        return 1/4*quadratic(2*sup(x, n, 3, n_sf)) - 3/4*quadratic(2*sup(x, n, 3, n_sf)-1) + 3/4*quadratic(2*sup(x, n, 3, n_sf)-2) - 1/4*quadratic(2*sup(x, n, 3, n_sf)-3)
    elif name == 'cubic':
        return 1/8*cubic(2*sup(x, n, 4, n_sf))-1/2*cubic(2*sup(x, n, 4, n_sf)-1)+3/4*cubic(2*sup(x, n, 4, n_sf)-2)-1/2*cubic(2*sup(x, n, 4, n_sf)-3)+1/8*cubic(2*sup(x, n, 4, n_sf)-4)
    elif name == 'hat':
        return 1/2*hat(2*sup(x, n, 2, n_sf))-hat(2*sup(x, n, 2, n_sf)-1)+1/2*hat(2*sup(x, n, 2, n_sf)-2)
    elif name == 'haar':
        return haar(2*sup(x, n, 1, n_sf))-haar(2*sup(x, n, 1, n_sf)-1)
    
@cython.profile(False)
cdef inline double sup(double x, size_t n, size_t d, size_t n_sf) nogil:
    if n_sf <= 1: return (x-1/2 + (n+1)/(n_sf+1))*d
    if n_sf > 1: return (x-1/2 + n/(n_sf-1))*d
   
