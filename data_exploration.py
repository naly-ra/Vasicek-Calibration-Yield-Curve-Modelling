# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 01:22:54 2023

@author: nalya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from statsmodels.regression.linear_model import OLS
from scipy.optimize import minimize
import scipy.stats



swap_rates = pd.read_excel('US_swap_rates.xlsx').set_index('observation_date')
swap_rates = swap_rates.dropna()

swap_rates.DSWP1.plot(label='1Y',color='orange')
swap_rates.DSWP5.plot(label='5Y',color='blue')
swap_rates.DSWP10.plot(label='10Y',color='red')

plt.title('Swap rates')
plt.xlabel('Years')
plt.ylabel('Rates')
plt.legend()