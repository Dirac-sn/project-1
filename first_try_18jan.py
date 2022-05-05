# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:52:25 2022

@author: harsh
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def trapezoidal_d(x,y):
    n = len(x)-1
    h = (x[-1]-x[0])/n #width of a subinterval
    k= y[0]+y[-1]
       
    app = (h/2)*( k + 2*sum(y[1:n]))
      
    return app

def trapezoid_int(a,b,f,n):
    h = (b-a)/n
    
    x_val = 
            
    y_val=[]
    for i in x_val:
        y_val.append(x(i))
  
    k = trapezoidal_d(x_val,y_val) 
      
    return k


def func(x):
    
    return np.exp(x)*np.sin(x)

a = 0 ; b = 2*np.pi


def Integration(x):
    return np.exp(x)*((np.sin(x)-np.cos(x))/2)


Int_f = Integration(b) - Integration(a)


def Int_list(f1,f_int,x0,xn,N):
    
    h = []
    fun_l = []
    for i in N:
        h.append((xn - x0)/i)
        z = f_int(x0,xn,i,f1)
        fun_l.append(z)
        
    
    return h,fun_l
    
    
def Error_list(Y_c,Y_a):
    err_l = []
    for i in range(len(Y_c)):
        m = abs(Y_a - Y_c[i])/abs(Y_a)
        err_l.append(m)
        
    return err_l    

def N_list(init,n):
    N = []
    i = 0
    while i < n:
         z = 2*init
         init = z
         N.append(z)
         i = i+1
    return N


N2 = np.arange(1,1002,1)
N1 = [2*i for i in N2]

print(N1)
h1,Y_trap = Int_list(func,trapezoid_int, a, b, N1)

h2,Y_simp = Int_list(func, simpson_int, a, b, N1)

Err_trap = Error_list(Y_trap,Int_f)

Err_simp = Error_list(Y_simp,Int_f)

fig,ax = plt.subplots()
ax.plot(N1,Err_simp,'ro',label = 'simpson')
ax.plot(N1,Err_trap,'go',label = 'trapezoid')
ax.set_title('N vs Error')
ax.legend()
ax.set_yscale('log')
ax.set_xscale('log')

print(Err_simp)




















'''
fig,ax = plt.subplots()
ax.plot(h1,Err_simp,'ro',label = 'simpson')
ax.plot(h1,Err_trap,'go',label = 'trapezoid')
ax.set_title('step size vs Error')
ax.legend()
ax.set_yscale('log')
ax.set_xscale('log')'''



