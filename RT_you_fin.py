import numpy as np
from scipy import *
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import *
from astropy.modeling import models
from scipy.optimize import fsolve
from astropy import units as u
from scipy.interpolate import interp1d
from cubature import cubature
from numba import njit
import copy

#units in sgs
tolerance=1e-12
N=100
epsilon=1e-4
T=10.0 #temperature K
nu=299792.458# 1 GHz
gamma=100#50#velocity gradient ,half of the photons escaped
ndim=1
fdim=1

def planck(lam, T):
    b = models.BlackBody(temperature=T*u.K)
    res = (b(lam * u.um)).to(u.Jy/u.sr) * np.pi
    return res.value
B=planck(nu,T)
print("planck",B)

def beta(mu):
    return gamma*(1. - np.exp(-1/(gamma*mu**2)))*mu**2

beta_esc, _ = cubature(beta,ndim,fdim, np.array([0]), np.array([1]))
print("beta",beta_esc)

one_min_e=(1-beta_esc)*(1-epsilon)

epsilon_star=1-(1-beta_esc)*(1-epsilon)
B_st=(epsilon*B)/(epsilon_star)
print("Bst",B_st,"Bst*eps_star=",B_st*epsilon_star)
print("B*eps/eps",B*epsilon/(epsilon_star**(1./2.)))
print("eps_st",epsilon_star)

@njit
def phi(x):
    return 1/(np.pi**(1./2))*np.exp(-x**2)
 

def core_function(s):
    def inner_inner_integral(t,x,mu):
        return phi(x + gamma * mu * t)
    def inner_integral(mu,x):
        if mu > 1.e-60:
            return phi(x) * 1 / mu * phi(x + gamma * mu * s) * np.exp(-1 / mu * cubature(inner_inner_integral,ndim,fdim, np.array([0]), np.array([s]),args=(x,mu,),abserr=1e-10, relerr=1e-10)[0])
        else:
            return 0
 

    def outer_integral(x):
        return cubature(inner_integral,ndim,fdim, np.array([0]), np.array([1]),args=(x,),abserr=1e-09, relerr=1e-09)[0]
        
    result=cubature(outer_integral,ndim,fdim, np.array([-3]), np.array([3]),abserr=1e-08, relerr=1e-08)[0]
    return 0.5 * result / (1 - beta_esc)
    
taush_min_tau = np.logspace(-25, 2, 100)
taush_min_tau=np.hstack(([0],taush_min_tau))
K_beta=np.zeros_like(taush_min_tau)
K_st=np.zeros_like(taush_min_tau)
#with open('output_K_50_100_range-10(2).txt', 'w') as file:
#    for i in range(len(taush_min_tau)):
#        K_beta[i]=core_function(taush_min_tau[i])
#        file.write(f"{taush_min_tau[i]},{K_beta[i]}\n")
#        file.flush()
#        print("K_b",K_beta[i])
#        print("i",i)

taush_min_tau,K_beta=np.loadtxt('output_K_100_100_range-25.txt', unpack=True)

taush=np.linspace(0,3,1000) #0-3
taush = np.hstack((taush,[10]))
S = B_st* np.ones_like(taush)  #initial value

S_new=copy.deepcopy(S)
const = epsilon_star*B_st 

def get_S(itau):
    #return np.interp(tau,taush,S, rigth=S[-1])
    return np.interp(np.abs(itau),taush,S, right=0) # with cloud boundary


def int_integral(taus,tau):
    return np.exp(np.interp(np.abs(taus-tau),taush_min_tau,np.log(K_beta))) * get_S(taus)
iter = 0
while True:
    for i in range(0,len(taush)):
        ta_min = taush[i] - 10.
        ta_min = np.max([0, ta_min])
        S_new[i] = one_min_e*cubature(int_integral,ndim,fdim, np.array([ta_min]), np.array([taush[i]+10]),args=(taush[i],),abserr=1e-30, relerr=1e-10)[0]+const#0.01 for gamma=50 [0]!!
        #quad(int_integral,-1,1,args=(taush[i],),epsabs=1.49e-30, epsrel=1.49e-12, limit=100000)[0]
        #print("integ",S[i], S_new[i])
    print("tolerance",np.max(np.abs((S_new - S)/S)))
    if np.max(np.abs((S_new - S)/S)) < tolerance:
        S=copy.deepcopy(S_new)
        break

    S=copy.deepcopy(S_new)

print(f"The solution for s is {S}")
with open('output_S.txt', 'w') as file:
    for x, i in zip(taush, S):
        file.write(f"{x},{i}\n")

#taush,S=np.loadtxt('output_S.txt', unpack=True)
x_arr=np.linspace(-400,100,1000)
I=np.zeros_like(x_arr)
@njit
def in_integral(t,x):
    return phi(x+gamma*1*t)

def S_phi(tau, integral, x):
    return get_S(tau)*np.exp(-1*integral)*phi(x+gamma*tau)
def outer_int(tau,x):
    integral=cubature(in_integral, ndim,fdim,np.array([0]), tau,args=(x,),abserr=1.49e-30, relerr=1.49e-10)[0]
    #print("S",x+gamma*tau,phi(x+gamma*tau),integral)
    #print("int",np.interp(tau,taush,S)*np.exp(-1*integral)*phi(x+gamma*tau))
    #return np.interp(tau,taush,S)*np.exp(-1*integral)*phi(x+gamma*tau)
    return S_phi(tau, integral, x)
    
for x_i in range(len(x_arr)):
    if x_arr[x_i]<0:
        t_min = (np.abs(x_arr[x_i]) - 5.) / gamma
        t_min = np.max([0, t_min])
        t_max = (np.abs(x_arr[x_i]) + 5.) / gamma
        I[x_i]=cubature(outer_int,ndim,fdim,np.array([t_min]),np.array([t_max]),args=(x_arr[x_i],),abserr=1.49e-30, relerr=1.49e-10)[0]
    else:
        I[x_i]=cubature(outer_int,ndim,fdim,np.array([0]),np.array([10]),args=(x_arr[x_i],),abserr=1.49e-30, relerr=1.49e-10)[0]
#for i in I:
        #print("I",i) 
with open('output_I.txt', 'w') as file:
    for x, i in zip(x_arr, I):
        file.write(f"{x},{i}\n")

print("saved to output.txt")
plt.plot(x_arr,I)
plt.xlabel("(v-v0)/$\Delta v$")
plt.ylabel("I(0,1,x)")
plt.savefig("I_tau_100_fint.png",dpi=300)
plt.show()
#data = np.column_stack((x, I))
#np.savetxt('output.txt', data, delimiter=' ')

