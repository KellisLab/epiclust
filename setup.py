#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from distutils.core import setup
from Cython.Build import cythonize ### Cython must be installed

setup(name="atac_module",
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
      ext_modules=cythonize("atac_module/utils.pyx")
)
