# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 14:54:59 2021

@author: vanhe
"""

import sys
# Root folder of main library
sys.path.insert(0, '/users/kkontras/Documents/Sleep_Project/Neureka/library')
# import loading
# import nedc

# Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import h5py

# loading.loadRecording('/esat/biomeddata/Neureka_challenge/IClabel/Neureka/edf/dev/01_tcp_ar/095/00009578/00009578_s002_t001_icalbl.edf', wiener=False)