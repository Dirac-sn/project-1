# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 17:43:44 2022

@author: harsh
"""

from numpy import column_stack, dtype, loadtxt
from sqlalchemy import column
from Myintegration import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad
from scipy import stats
quadrature = np.vectorize(quad)
plt.style.use("ggplot")


#
# part(b)
#
'''
a = 0;b = 1
Func_l = [lambda x:x,lambda x: x**2,lambda x: x**3,lambda x: x**4,lambda x: x**5,lambda x: x**6,lambda x: x**7,lambda x: x**8]

trap_v =np.zeros(len(Func_l))
simp_v = np.zeros(len(Func_l))
quad_v2 = np.zeros(len(Func_l))
quad_v4 = np.zeros(len(Func_l))
Analytic = np.zeros(len(Func_l))
err = np.zeros(len(Func_l))
#trapezoidal
for i in range(len(Func_l)):
    trap_v[i] = MyTrap(Func_l[i],a,b,1)
    simp_v[i] = MySimp(Func_l[i],a,b,2)
    quad_v2[i] = MyLegQuadrature(Func_l[i],a,b,2,1)
    quad_v4[i] = MyLegQuadrature(Func_l[i],a,b,4,1)
    Analytic[i],err[i] = quad(Func_l[i],a,b)

df = pd.DataFrame({"Trapezoidal":trap_v,"Simpson":simp_v, "quad 2-point":quad_v2,"quad 4-point" : quad_v4,"Inbuilt":Analytic})
print(df)
df.to_csv('partb.csv')
'''
#
# Part(c)

n = 2*np.arange(1,17)
h = 1/n
#f_str = input("Enter the function : ")
#f = lambda x :eval(f_str)
f = lambda x : 1/(1+x**2)
my_pi_simp = 4*MySimp(f,0,1,n)
my_pi_trap = 4*MyTrap(f,0,1,n)

err_simp= np.abs(my_pi_simp-np.pi)
err_trap= np.abs(my_pi_trap-np.pi)
  
fig,ax = plt.subplots()
ax.plot(n,np.pi*np.ones(n.shape),'olive',label = "\u03C0/4")
ax.plot(n,my_pi_simp,'r*--',label= "my_pi_simp(n)")
ax.plot(n,my_pi_trap,'b*--',label= "my_pi_trap(n)")
ax.set_title('Convergence of integral to \u03C0/4')
ax.set_xlabel('no. of subintervals')
ax.set_ylabel('Integral')
ax.legend()

fig,ax = plt.subplots()
ax.plot(n,err_simp,'b1--',label= "e_simp(n)")
ax.plot(n,err_trap,'ro--',label= "e_trap(n)")
ax.set_title('Error in numerical method vs n')
ax.set_xlabel('no. of subintervals')
ax.set_ylabel('absolute error')
ax.legend()

fig,ax = plt.subplots()
ax.plot(np.log(h),np.log(err_trap),label= "e_trap(n)")
ax.plot(np.log(h),np.log(err_simp),label= "e_simp(n)")
ax.set_title('log(e) vs log(h)')
ax.set_xlabel('ln(h)')
ax.set_ylabel('ln(e)')
ax.legend()

slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(h),np.log(err_simp))
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(np.log(h),np.log(err_trap))
print(slope)
print(slope1)
#
# Part(d)
#
'''
signi_digits = 5
tab_dat = np.array([MyTrap(f,0,1,10000,signi_digits),MySimp(f,0,1,10000,signi_digits)])    
pi_arr =np.pi*np.ones((2,))
df = pd.DataFrame({"Method":["Trapezoidal","Simpson1/3"],"Pi_calc":4*tab_dat[:,0], "n":tab_dat[:,1],"E" : np.abs(4*tab_dat[:,0] - pi_arr)/pi_arr})
print(df)
df.to_csv('partd.csv')
'''
#
# Part (e)
#
'''
n_points,m_arr = 2**np.arange(1,7),2**np.arange(0,6)

nm_mat = np.ones((len(n_points),len(m_arr)))
for i,n_i in enumerate(n_points):
    nm_mat[i,:] = 4*MyLegQuadrature(f,0,1,n_i,m_arr)
np.savetxt("pi_quad-1114.dat",nm_mat,delimiter=",",fmt="%.16f")

nm_mat = np.loadtxt("pi_quad-1114.dat",delimiter= ",",dtype=float)
err_nm_mat = nm_mat - np.pi*np.ones((len(n_points),len(m_arr)))

print(nm_mat)

fig,axs1 = plt.subplots()
axs1.plot(n_points,nm_mat[:,np.where(m_arr==1)[0][0]],'o--',label="m=1",marker=".")
axs1.plot(n_points,nm_mat[:,np.where(m_arr==8)[0][0]],'*--',label="m=8",marker=".")
axs1.plot(n_points,np.arccos(-1)*np.ones(n_points.shape),label = "$\pi = \cos^{-1}(-1)}$")
axs1.set_title('\u03C0 calculated using n point quadrature vs n')
axs1.set_xlabel('n')
axs1.set_ylabel('pi_quad(n)')


axs1.legend()

fig1,ax1 = plt.subplots()
ax1.plot(m_arr,nm_mat[np.where(n_points==2)[0][0]],'2--',label="n=2",marker="1")
ax1.plot(m_arr,nm_mat[np.where(n_points==8)[0][0]],'1--',label="n=8",marker="1")
ax1.plot(m_arr,np.arccos(-1)*np.ones(m_arr.shape),label = "$\pi = \cos^{-1}(-1)}$")
ax1.set_title('\u03C0 calculated using n point quadrature vs no. of subintervals')
ax1.set_xlabel('m')
ax1.set_ylabel('pi_quad(m)')
ax1.legend()
   
#plt.legend()
plt.show()

print(err_nm_mat)

fig,axs1 = plt.subplots()
axs1.plot(n_points,err_nm_mat[:,np.where(m_arr==1)[0][0]],label="m=1",marker=".")
axs1.plot(n_points,err_nm_mat[:,np.where(m_arr==8)[0][0]],label="m=8",marker=".")
axs1.set_title('error in \u03C0 calculated using n point quadrature vs n')
axs1.set_xlabel('n')
axs1.set_ylabel('error(n)')
axs1.legend()

fig1,ax1 = plt.subplots()
ax1.plot(m_arr,err_nm_mat[np.where(n_points==2)[0][0]],label="n=2",marker="1")
ax1.plot(m_arr,err_nm_mat[np.where(n_points==8)[0][0]],label="n=8",marker="1")
ax1.set_title('error in \u03C0 calculated using n point quadrature vs no. of subintervals')
ax1.set_xlabel('m')
ax1.set_ylabel('error(m)')
ax1.legend()
'''

#
# Part (f)
#
# 


result = MyLegQuadrature(f,0,1,10000,2,8)

print(result)

'''
n_points = 2**np.arange(1,6)

tol_arr = np.arange(2,11)
#myttype =[('pi',float),('m',float)]
csv_dat = np.zeros((len(n_points),len(tol_arr)*2))
fixed_tol_mat = np.ndarray((len(n_points),len(tol_arr),2))
 
for i,n_i in enumerate(n_points):
    tmp = np.column_stack(MyLegQuadrature(f,0,1,n_i,m=1000,d=tol_arr))
    fixed_tol_mat[i,:]= tmp
    csv_dat[i,:] = tmp.flatten()
np.savetxt("f_data.csv",csv_dat,delimiter= ",",fmt="%g")

mat_i = np.loadtxt("f_data.csv",delimiter= ",",dtype=float)

print(mat_i)
'''