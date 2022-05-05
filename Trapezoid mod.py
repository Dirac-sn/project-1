# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 14:56:35 2022

@author: harsh
"""

import numpy as np 
from scipy.integrate import quad

def trapezoidal_rule(a,b,f,n=2**2,n_max=2**32,rtol=0.5e-3):
    nstart,nstop = np.log2(n),np.log2(n_max)
    n_array = np.logspace(nstart,nstop,base=2,num = int(nstop-nstart+1))
    I = np.zeros(n_array.shape)
    h = (b-a)/n_array
    x = np.linspace(a,b,int(n_array[0]+1))
    y = f(x)
    I[0] = (h[0]/2)*np.sum(y[:-1] + y[1:])
    
    for i in np.arange(1,n_array.shape[0]):
        x = np.linspace(a,b,int(n_array[i-1]+1))
        modedx = (x[:-1] +x[1:])/2
        I[i] = (1/2)*I[i-1] + h[i]*(np.sum(f(modedx)))
        
        if np.abs((I[i]-I[i-1])/I[i]) <= rtol:    
            n1 = n_array[i]
            return I[i],n1   
        
    return I[-1],-1


'''
def Simpson_rule(a,b,f,n=2**4,n_max=2**32,rtol=0.5e-6):
    nstart,nstop = np.log2(n),np.log2(n_max)
    n_array = np.logspace(nstart,nstop,base=2,num = int(nstop-nstart+1))
    I = np.zeros(n_array.shape)
    h = (b-a)/n_array
    x = np.linspace(a,b,int(n_array[0]+1))
    y = f(x)
    y_e1 = y[2::2]
    y_o = y[1:-1:2]
    y_e2 =  y[:-2:2]
    I[0] = (h[0]/3)*np.sum(y_e1 + 4*y_o + y_e2)
    
    for i in np.arange(1,n_array.shape[0]):
        x = np.linspace(a,b,int(n_array[i-1]+1))
        
        x_e1 = x[2::2] 
        x_e1 = x[1:-1:2] 
        x_e1 = x[:-2:2] 
        
        I[i] = (1/2)*I[i-1] + h[i]*(np.sum(f(modedx)))
        
        if np.abs((I[i]-I[i-1])/I[i]) <= rtol:    
            return I[i],0   
        
    return I[-1],-1
'''


def gauss2pt(a, b, f,n = 2**2, n_max =2**32, rtol = 0.5e-6):
    
    nstart,nstop = np.log2(n),np.log2(n_max)
    n_array = np.logspace(nstart,nstop,base=2,num = int(nstop-nstart+1))
    I = np.zeros(n_array.shape)
    h = (b-a)/n_array
        
    for j in range(0,n_array.shape[0]):
        y_e,y_o = [] , []
        x = np.linspace(a,b,int(n_array[j]+1))
        for i in range(len(x)):
            y_e.append(f(x[i] - (h[j]/2) - np.sqrt(1/3)*(h[j]/2)))
            y_o.append(f(x[i] - (h[j]/2) + np.sqrt(1/3)*(h[j]/2)))
    
        I[j] = (h[j]/2)*np.sum(y_e + y_o)

        if np.abs((I[j]-I[j-1])/I[j]) <= rtol:    
            n1 = n_array[j]
            return I[j],n1
 
    return I[-1],-1



def gauss_quad(f,a,b,n):
    
    h = (b-a)/n
    x = np.linspace(a,b,n+1)
    I_n = []
    for i in range(len(x)):
        y1 = f(x[i] - (h/2) - ((1/np.sqrt(3))*(h/2)))
        y2 = f(x[i] - (h/2) + ((1/np.sqrt(3))*(h/2)))
        I_n.append(y1 + y2)
    
    I = (h/2)*sum(I_n)
    return I

print(gauss2pt(0, 1, lambda x: 1/(1+x**2)))


