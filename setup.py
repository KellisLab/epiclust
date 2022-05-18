#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import setuptools
from distutils.extension import Extension
from Cython.Build import cythonize ### Cython must be installed
import numpy

extensions = [Extension("atac_module.utils", ["atac_module/utils.pyx"],
                        extra_compile_args=["-O3"])]
setuptools.setup(name="atac_module",
                 install_requires=[
                         "numpy",
                         "cython",
                         "dask",
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
                 packages=setuptools.find_packages("."),
                 test_suite="test",
                 scripts=["scripts/atac_module_compute_cor.py"],
                 include_dirs=[".", numpy.get_include()],
                 ext_modules=cythonize(extensions))
