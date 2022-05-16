#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import setuptools
from Cython.Build import cythonize ### Cython must be installed
import numpy
setuptools.setup(name="atac_module",
                 install_requires=[
                         "numpy",
                         "pandas",
                         "scipy",
                         "matplotlib",
                         "seaborn",
                         "anndata",
                         "python-igraph",
                         "leidenalg",
                         "adjustText",
                         "umap-learn",
                         "pyranges",
                         "tqdm",
                         "sklearn",
                         "infomap"
                 ],
                 packages=setuptools.find_packages(),
                 test_suite="test",
                 include_dirs=[numpy.get_include()],
                 ext_modules=cythonize("atac_module/utils.pyx")
)
