#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import setuptools
from glob import glob

setuptools.setup(name="epiclust",
                 version="0.3.2",
                 author="Benjamin James",
                 author_email="benjames@mit.edu",
                 url="https://github.com/KellisLab/epiclust",
                 license="GPL",
                 install_requires=[
                     "numpy",
                     "scipy",
                     "sklearn",
                     "numba",
                     "anndata>=0.8.0",
                     "pandas",
                     "umap-learn",
                     "pynndescent>=0.5.7",
                     "scanpy",
                     "matplotlib",
                     "seaborn",
                     "python-igraph",
                     "leidenalg",
                     "tqdm",
                     "infomap>=2.5.0",
                     "gtfparse",
                     "pytabix",
                     "pyranges",
                 ],
                 packages=setuptools.find_packages("."),
                 test_suite="test",
                 scripts=glob("scripts/*.py"),
                 )
