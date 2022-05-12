#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import setuptools
from Cython.Build import cythonize ### Cython must be installed

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
                 ext_modules=cythonize("atac_module/utils.pyx")
)
