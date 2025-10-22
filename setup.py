# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 00:52:16 2019

@author: Jin Dou
"""

import setuptools


setuptools.setup(
  name="dynamically_warped_trf",
  version="2.0.0",
  author="Powerfulbean",
  author_email="powerfulbean@gmail.com",
  long_description_content_type="text/markdown",
  url="",
  packages=setuptools.find_packages(),
  python_requires=">=3.10",
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  install_requires=[
    "torch>=2.0.0",
    "numpy>=2.0.0",
    "matplotlib",
    "tqdm",
    "tour",
    "mtrf",
    "nntrf",
  ],
)