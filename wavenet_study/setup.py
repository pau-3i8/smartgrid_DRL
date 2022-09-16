from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        'activation_functions',
        ['activation_functions.pyx'],
        extra_compile_args=['-fopenmp', '-ffast-math'],
        extra_link_args=['-fopenmp'],
    )]

setup(ext_modules = cythonize(ext_modules, compiler_directives={'language_level':'3'}), include_dirs=[numpy.get_include()])
