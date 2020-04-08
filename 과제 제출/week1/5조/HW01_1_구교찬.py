# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:42:33 2020

@author: KyoChan
"""


##Assignment 2##
import numpy as np
np.set_printoptions(precision=3)
import pandas as pd
pd.set_option('display.precision',3)
import matplotlib.pyplot as plt

#Distribution from scipy
#from scipy.stats import binom, beta
from scipy.stats import norm#since all distributions are norm

#Sampling Density
#population parameter
loc=5; scale=2
#generate toy sample
N = 30
np.random.seed(101)
data = norm.rvs(loc,scale, size= N)#notice scale is s.e not var
print(data)
print(data.size)
print(data.sum())

#Prior distribution of mu
#Our prior is norm
mu_pri=1 ; sigma_pri=2
prior = norm(mu_pri,sigma_pri)
theta = np.linspace(-1,3,100);theta
#Draw plot of prior
plt.plot(theta,prior.pdf(theta), color = 'r')#i added theta
plt.title('normal prior')
plt.xlabel('theta')
plt.ylabel('p(theta)')

#Likelihood
def likelihood(data,theta):
    mu = np.mean(data); sigma = np.std(data)
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(theta-mu)**2/(2*sigma**2))
theta = np.linspace(1,9,100)
plt.plot(theta, likelihood(data,theta), color = 'g')
plt.title("likelihood")
plt.xlabel("theta")
plt.ylabel("p(Data|theta)")
plt.ylim(0,1)

#Posterior distribution of mu
var_dat=np.std(data)**2
mu_pos_num= (mu_pri/sigma_pri**2)+(data.sum()/var_dat)
mu_pos_den= (1/sigma_pri**2)+(N/var_dat)
mu_pos= mu_pos_num/mu_pos_den
sigma_pos= 1/((1/sigma_pri**2)+N/var_dat)
posterior = norm(mu_pos,sigma_pos)
theta = np.linspace(4,6,100);theta
plt.plot(theta,posterior.pdf(theta), color='r')
plt.title('normal posterior')
plt.xlabel('mu')
plt.ylabel('p(mu)')