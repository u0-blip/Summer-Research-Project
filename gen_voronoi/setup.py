from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Process input file',
    ext_modules=cythonize("process_inp_file.pyx"),
    zip_safe=False,
)