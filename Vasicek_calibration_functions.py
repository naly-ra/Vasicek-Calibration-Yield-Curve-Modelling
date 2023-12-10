# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 01:24:19 2023

@author: nalya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from data_exploration import swap_rates

#Vasicek Simulator

def simulate_vasicek_euler(r0, kappa, theta, sigma, T, dt):
    n_steps = int(T / dt)
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    short_rate_path = np.zeros(n_steps + 1)
    short_rate_path[0] = r0

    for i in range(1, n_steps + 1):
        short_rate_path[i] = (short_rate_path[i - 1] + kappa * (theta - short_rate_path[i - 1]) * dt
                              + sigma * np.sqrt(dt) * dW[i - 1])

    return short_rate_path

def simulate_vasicek_euler_average(r0, kappa, theta, sigma, T, dt, n_paths):
    n_steps = int(T / dt)
    dW_matrix = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
    short_rate_matrix = np.zeros((n_paths, n_steps + 1))
    short_rate_matrix[:, 0] = r0

    for i in range(1, n_steps + 1):

        short_rate_matrix[:, i] = (short_rate_matrix[:, i - 1] + kappa * (theta - short_rate_matrix[:, i - 1]) * dt
                                                               + sigma * np.sqrt(dt) * dW_matrix[:, i - 1])

  
    return np.mean(short_rate_matrix, axis=0)

#MLE calibrator 

def mean(x,dt,k,theta):
    return x* np.exp(-k*dt) + theta*(1-np.exp(-k*dt))

def var(dt,k,sigma):
    return (sigma**2/2*k) * (1-np.exp(-2*k*dt))

def log_likelihood_vasicek(estimator,time_series):
    
    dt = 1/252 #trading days per year
    k = estimator[0]
    theta = estimator[1]
    sigma = estimator[2]
    
    x_t1 = time_series[1:]
    x_t0 = time_series[:-1]
    
    mean_t0 = mean(x_t0,dt,k,theta)
    var_t0 = var(dt,k,sigma)
    
    # sum of log-density of normal distributions with mean E[X_t] and Var[X_t]
    log_likelihood = np.sum(np.log(scipy.stats.norm.pdf(x_t1,loc=mean_t0,scale=var_t0)))
    
    return -log_likelihood

def optimizer(rates):
    opt = minimize(fun=log_likelihood_vasicek,x0=starting_point,args=(rates,),constraints=constraints)

        
    k, theta, sigma = opt.x

    
    return opt.x

rates = swap_rates['DSWP1'].values
constraints = [{'type':'ineq', 'fun': lambda k: k},
               {'type':'ineq', 'fun': lambda sigma: sigma}]

starting_point = [1,5,1]

#OLS Regressor

def OLS_regressor(time_series):
    
    #We will train our data on one year (252 points) and then forecast at year 2
    observed_data = time_series
    y = np.diff(observed_data)
    X = observed_data[:-1]
    X = sm.add_constant(X)
    model = OLS(y,X)
    results = model.fit()
    k =  - results.params[1] * 252
    theta = results.params[0] / k
    y_hat = model.predict(results.params,X)
    sigma = np.std(y - y_hat)* 252
    
    return [k,theta,sigma]

#SKlearn Regressor

def SK_regressor(time_series):
    X_t = time_series
    y = np.diff(X_t)
    X = X_t[:-1].reshape(-1, 1)
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)
    # regression coeficient and constant
    k = -reg.coef_[0]*252
    theta = reg.intercept_ / k
    y_hat = reg.predict(X)
    sigma = np.std(y - y_hat)*252
    return [k,theta,sigma]
