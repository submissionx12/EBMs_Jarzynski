import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
import torch.autograd as autograd
# import seaborn as sns
from torch import optim
import scipy 
import os
from functorch import jacfwd, jacrev, vmap, make_functional
if(th.cuda.is_available()):
    device = th.device("cuda:0")
else:
    device = th.device("cpu")

import datetime
now = datetime.datetime.now()



#author: Davide Carbone


#print current date time
print(now)    

    
th.backends.cudnn.benchmark = True
th.set_default_dtype(th.float64)


#Type of algorithm: 0 is Jarzynski corrected (alg. 1), 1 is PCD, 2 is CD
algo=[0,1,2]    

#Random seed
seed=0
th.manual_seed(seed)

#Dimension 
dim=50

#List of numbers of walkers 
n_walk=th.tensor([10000],device=device)

#list of learning rates for z
factors=th.tensor([1/8],device=device)

#time step for ULA
dt=0.1
#List of time horizons, rescaled wrt learning rates
tmax =(1e2/factors)

#List of numbers of iterations
n_iter = th.ceil(tmax/dt).int()

#coefficient in front of noise in ULA
sdt  = th.tensor(np.sqrt(2*dt),device=device)

#number of modes of student
n_studs=2

#number of mode of teacher (*** keep 2 in the present version ***)
n_teachers=2

#number of inner cicles of ULA (+ weight update in algo 1)
equil=1

#multiplicative factor to change ratio of learning rates for z and means
mult=1

#standard deviations used as threshold for resampling
std1=th.tensor([0.1,0.2,0.4,0.8,1],device=device)

#learning rate for means
alph = factors*0.2

#multiplicative factor for GD step for means
abdt = alph*dt

#clone learning rate for z
alphz = factors.clone()

#multiplicative factor for GD step for z
zdt = alphz*dt*mult

#density, potential, derivatives wrt x, a and z
# a: vector of means of shape (n mode, dimension) 
# z: vector of z of shape (n mode)
# x: vector of shape (batch size, dimension)
def rho(z,a,x):    
    return th.sum(th.exp(-0.5*th.linalg.vector_norm((x.unsqueeze(1)-a),axis=-1)**2-z),axis=-1)

def U(z,a,x):
    return -th.log(rho(z,a,x))

def dUdx(z,a,x):
    return x-(th.einsum("ij,jk->ki",th.exp(-0.5*th.linalg.vector_norm((x.unsqueeze(1)-a),axis=-1)**2-z),a)/rho(z,a,x)).transpose(0,1)

def dUda(z,a,x):
    return (th.einsum("ij,ijk->jki",th.exp(-0.5*th.linalg.vector_norm((x.unsqueeze(1)-a),axis=-1)**2-z),(a-x.unsqueeze(1)))/rho(z,a,x))


def dUdz(z,a,x):
    return (th.exp(-0.5*th.linalg.vector_norm((x.unsqueeze(1)-a),axis=-1)**2-z)).transpose(0,1)/rho(z,a,x)


#function for Jarzynski weights update
def logalpha1(x,y,z,a):
    return U(z,a,x)+0.5*th.sum((y-x)*dUdx(z,a,x),axis=-1)+0.25*dt*th.sum((dUdx(z,a,x))**2,axis=-1)


#name of the folder where output data are stored
foldername='tmax'+str(tmax[-1].item())+'_lr_'+str(mult)+'t_GMM'+str(seed)+'nteach_'+str(n_teachers)+'nstuds_'+str(n_studs)+'_equil_'+str(equil)+'factors'+str(factors[0].item())

if not os.path.exists(foldername): # if not exists, create one
    os.makedirs(foldername)    

#Parameters of the teacher
a0=th.zeros((n_teachers,dim),device=device)
#first mode
a0[0,0]=-10
#second mode
a0[1,0]=6

z0=th.zeros((n_teachers,),device=device)
#z for the second mode
z0[1] = th.tensor(-np.log(3),device=device)
#mass of the first mode
p0 = 1/(1+th.exp(-z0[1]))

#number of data points 
n_teach=int(1e4)
#number of data points in the first mode
nra=th.floor(n_teach*p0).int()

#sample data points from teacher
x0a=MultivariateNormal(a0[0],th.eye(dim,device=device)).sample((nra,)) 
x0b=MultivariateNormal(a0[1],th.eye(dim,device=device)).sample((n_teach-nra,))

x0=th.cat((x0a,x0b),0)



#Normalization constant of gaussian in dimension d
Cnorm=th.tensor(np.sqrt(2*np.pi)**dim,device=device) 


ones=th.ones(n_teach,device=device)


#function for simulating 
# nstat: number of repetitions of each setup for collecting statistics
# n_iter: number of iterations for every algorithm
# abdt_: learning rate for means
# zdt_: learning rate for z
# n_walk: number of walkers
# std1: standard deviation to be used for resampling step in algo 1
# algo: algorithm to be used
# foldername: name of the folder where to save data
# index: numerical auxiliary variable to be prepended to each simulation output files
def stat(nstat,n_iter,abdt_,zdt_,n_walk,std1,algo,foldername,index):
    
    #create string for file name relative to present simulation
    filename=foldername+'/'+str(index)+'nstat_'+str(nstat)+"_abdt_"+str(abdt_.item())[:6]+"_ztd_"+str(zdt_.item())+"_nwalk_"+str(n_walk.item())+"_std1_"+str(std1.item())+"_algo_"+str(algo)
    
    #Declare variables for collecting stats
    pstat=th.zeros((nstat,n_iter,n_studs),device=device)
    astat=th.zeros((nstat,n_iter,n_studs,dim),device=device)
    CEte_stat=th.zeros((nstat,n_iter),device=device)
    CEt_stat=th.zeros((nstat,n_iter),device=device)
    
    #Multiply by 10 the learning rate if CD and instantiate local variables for learning rates
    if algo==2:
        abdtloc=abdt_.clone()*10
        zdtloc=zdt_.clone().repeat(n_studs)*10
    else:
        abdtloc=abdt_.clone()
        zdtloc=zdt_.clone().repeat(n_studs)
    #Put the learning rate of the first z to zero to avoid global blow up of energy
    zdtloc[0]=0
    

    #cycle for collecting stats (n_iter iterations with same hyperparameters and save mean and maximum dispersion error)
    for j in range(nstat):

        #initialization of means
        a=th.zeros((n_iter,n_studs,dim),device=device)


        #initialization of means, slightly perturbed from identically zero
        a[0]=th.normal(0,1,size=(n_studs,dim),device=device)*0.01
        #greater along the direction of alignment of the wells
        a[0,0,0]=-0.1
        a[0,1,0]=0.1

        #Inizialization of z (same mass for all the modes)
        z = th.zeros((n_iter,n_studs),device=device)
        z[0,0]=0

        #Initialization of walkers for Jarzynski
        #equal proportion in first two modes, just for numerical reasons since the all the modes are near zero
        nraini=th.floor(n_walk*0.5).int()
        if algo==0:
            foldername='Jarz_'
            logw = th.zeros((n_walk),device=device)


            xa=MultivariateNormal(a[0,0],th.eye(dim,device=device)).sample((nraini,)) 
            xb=MultivariateNormal(a[0,1],th.eye(dim,device=device)).sample((n_walk-nraini,))

            x=th.cat((xa,xb),0)

        # Multinomial sampling from data as initial condition, if PCD or CD 
        elif algo==1:
            foldername='PCD_'

            x=x0[th.multinomial(ones,n_walk,replacement=True)]
        else: 
            foldername="CD_"
            x=x0[th.multinomial(ones,n_walk,replacement=True)]

        #Number of iteration between every reset to data for CD
        nreset=4
        #List to save the log of analytical partition function
        logZt = th.zeros((n_iter),device=device)
        #Value of log Z at time 0
        logZt[0]+= th.log(Cnorm*th.sum(th.exp(-z[0])))
        #store initial log Z in an auxiliary variable for resampling step in Jarzynski
        logZt0 = logZt[0]

        #constant part of KL that depends just on data
        C0 = -th.mean(U(z0,a0,x0))-th.log(Cnorm*th.sum(th.exp(-z0))) 
        #list to save the analytical KL
        CEt = th.zeros((n_iter),device=device)
        #KL at time 0
        CEt[0]+=logZt[0]+th.mean(U(z[0],a[0],x0))+C0

        #clone list to save estimated KL through Jarzynski
        CEte = CEt.clone()

        #List to save the coefficient of variation (CV) of Jarzynski weights
        varian=th.zeros((n_iter,))
        #noise buffer for Langevin step
        noise = th.randn((n_walk,dim), device=device)    
        #auxiliary variable to save walkers at previous time step to update Jarzynski weights
        xo=x.clone()    
        #counter for number of resampling occurred in Jarzynski algo
        count=0

        #for cycle for time evolution
        for i in range(n_iter-1):

            #derivatives wrt parameters computed in data points and walkers
            dUai = dUda(z[i],a[i],x)
            dUzi = dUdz(z[i],a[i],x)
            dUai0 = dUda(z[i],a[i],x0)
            dUzi0 = dUdz(z[i],a[i],x0) 

            #Jarzynski corrected algorithm
            if algo==0:
                #CV of the weights normalized on mean
                varian[i]=(2*logw).exp().mean()/(logw.exp().mean())**2-1
                #normalized weights to compute weighted averages
                ppi = th.exp(logw)/th.sum(th.exp(logw))
                #step of GD
                a[i+1] = a[i] + abdtloc*(th.sum(dUai*ppi,axis=-1)-th.mean(dUai0,axis=-1))               
                z[i+1] = z[i] + zdtloc*(th.sum(dUzi*ppi,axis=-1)-th.mean(dUzi0,axis=-1))
                #Langevin step for "equil" iterations
                for l in range(equil):
                    #store previous position of walkers
                    xo = x
                    #Langevin step
                    x-= dt*dUdx(z[i],a[i],x) - sdt*noise.normal_(0,1) 
                    #weights update
                    logw = logw - logalpha1(x,xo,z[i+1],a[i+1]) + logalpha1(xo,x,z[i],a[i])
                
                #save estimated log Z from Jarzynski weights
                logZt[i+1] = logZt0+th.log(th.mean(th.exp(logw))) 
                #save estimated KL  
                CEt[i+1] = logZt[i+1]+th.mean(U(z[i+1],a[i+1],x0))+C0
                #Resampling step 
                if varian[i]>std1:
                    #Increment of counter
                    count=count+1
                    #Multinomial weighted resampling
                    x=x[th.multinomial(logw.exp(),n_walk,replacement=True)]
                    #Put all log weights to 0
                    logw=logw*0
                    #Store previous log Z to trace it 
                    logZt0 = logZt[i+1]
            else:
                #GD step if PCD or CD
                a[i+1] = a[i] + abdtloc*(th.mean(dUai,axis=-1)-th.mean(dUai0,axis=-1))
                z[i+1] = z[i] + zdtloc*(th.mean(dUzi,axis=-1)-th.mean(dUzi0,axis=-1))
                #Langevin step for "equil" iterations
                for l in range(equil):
                    x-= dt*dUdx(z[i],a[i],x) - sdt*noise.normal_(0,1)   
                #Reset to data if CD every nreset iterations
                if algo==2 and i%nreset==0:
                    x=x0[th.multinomial(ones,n_walk,replacement=True)]
            #store analytical log Z
            CEte[i+1] = th.log(Cnorm*th.sum(th.exp(-z[i+1])))+th.mean(U(z[i+1],a[i+1],x0))+C0
                
        #compute relative mass of the weights via Softmax
        p = th.nn.functional.softmax(-z,dim=1)
        #store results (p, means, KL) of one iteration with fixed hyperparameters
        pstat[j]=p
        astat[j]=a

        CEte_stat[j]=CEte
        if algo==0:                
            CEt_stat[j]=CEt

    #Store mean and maximum dispersion error of n_iter iterations in .npz archives for p, means and KL. Store CV of the weights and final weights for last iteration.
    pout_mean=pstat.mean(axis=0)
    pout_error=th.abs(th.max(pstat,axis=0)[0]-th.min(pstat,axis=0)[0])/2
    aout_mean=astat.mean(axis=0)
    aout_error=th.abs(th.max(astat,axis=0)[0]-th.min(astat,axis=0)[0])/2
    CEte_mean=CEte_stat.mean(axis=0)
    CEte_error=th.abs(th.max(CEte_stat,axis=0)[0]-th.min(CEte_stat,axis=0)[0])/2
    if algo==0:
        CEt_mean=CEt_stat.mean(axis=0)
        CEt_error=th.abs(th.max(CEt_stat,axis=0)[0]-th.min(CEt_stat,axis=0)[0])/2
        np.savez(filename,pout_mean.cpu(),pout_error.cpu(),aout_mean.cpu(),aout_error.cpu(),CEte_mean.cpu(),\
                 CEte_error.cpu(),CEt_mean.cpu(),CEt_error.cpu(),logw.cpu(),varian.cpu())

    else:    
        np.savez(filename,pout_mean.cpu(),pout_error.cpu(),aout_mean.cpu(),aout_error.cpu(),CEte_mean.cpu(),CEte_error.cpu())



#number of iterations for stats at fixed hyperparameters        
n_stat=5  
#auxiliary variable to count number of simulations 
index=0
for n_walk_i in n_walk:
    for k in range(len(factors)):
        for algo_i in algo:
            if algo_i==0:
                for std1_i in std1:
                    stat(n_stat,n_iter[k],abdt[k],zdt[k],n_walk_i,std1_i,algo_i,foldername,index)
            else:
                stat(n_stat,n_iter[k],abdt[k],zdt[k],n_walk_i,std1_i,algo_i,foldername,index)
            index=index+1

#datetime for elapsing running time
now = datetime.datetime.now()

print(now)    
