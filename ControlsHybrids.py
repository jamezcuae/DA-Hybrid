# %% 
# 0. The basics
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# %% 
# ===========================================================================
# 1. Generate the truth (without transient)
from misc_funs import natrun
Nx = 12
tmax = 14; dt = 0.025
print('***generating nature run***')
x,t,ut,ug0 = natrun(Nx,tmax)
Nsteps = np.size(t); print ('Nsteps', Nsteps)

# Plot the trajectory
nrow=3; ncol=4
limp=13

plt.figure()
for jpl in range(Nx):
 plt.subplot(nrow,ncol,jpl+1)
 plt.plot(t,ut[:,jpl],'k') #ut has dimn time(201) by xdimn (20)
 plt.xlabel('time')
 plt.ylabel('x['+str(jpl+1)+']')
 plt.ylim([-limp,limp])
 plt.grid(True) #show grid
del jpl
plt.subplots_adjust(top=0.955,bottom=0.08,left=0.11,right=0.9,hspace=0.465,wspace=0.345)

cmap_0 = plt.cm.get_cmap("BrBG")
plt.figure()
plt.contourf(x,t,ut,cmap=cmap_0,vmin=-10,vmax=10,extend='both')
plt.colorbar()
plt.xlabel('grid points',fontsize=14)
plt.ylabel('time',fontsize=14)
plt.title('Hovmoller plot',fontsize=14)

#%%
# ========================================================================
# 2. Observations
from miscobs import getHR, genobs
# Select the period of observations (in model steps) and the grid
# Gridobs: 'all','1010', 'landsea'
period_obs = 2
gridobs = '1010' 
stdobs = 1

Nx_obs, loc_obs, H, R, Rsq, invR = getHR(gridobs,Nx,stdobs)

loc_nobs = np.setdiff1d(x,loc_obs)
locs = [loc_obs,loc_nobs]
# y
seed = 1
print('***generating observations***')
tobs,yobs = genobs(dt,ut,Nsteps,Nx_obs,H,period_obs,Rsq,seed)


# plot
plt.figure()
for jpl in range(Nx):
 plt.subplot(nrow,ncol,jpl+1)
 plt.plot(t,ut[:,jpl],'k')
 plt.xlabel('time')
 plt.ylabel('x['+str(jpl+1)+']')
 plt.ylim([-limp,limp])
 plt.grid(True)
del jpl
for jpl in range(Nx_obs):
 plt.subplot(nrow,ncol,loc_obs[jpl]+1)
 plt.autoscale(False) 
 plt.scatter(tobs,yobs[:,jpl],20,'r')
 plt.ylim([-limp,limp])
del jpl
plt.subplots_adjust(top=0.955,bottom=0.08,left=0.11,right=0.9,hspace=0.465,wspace=0.345)


# %%
#===========================================================================
# 3. Variational data assimilation
from rmse_spread import rmse_spread 
from misc_funs import getBc
Bc,Bc_sq = getBc(Nx)
# plot
plt.figure()
lim = 2
print('***show climatological B***')
my_cmap = matplotlib.cm.get_cmap('BrBG')
plt.imshow(np.array(Bc),interpolation="nearest",cmap=my_cmap,vmin=-lim,vmax=lim)
plt.colorbar()
plt.xlabel('grid points',fontsize=14)
plt.ylabel('grid points',fontsize=14)
plt.title('climatological B',fontsize=14)


#%%
# --------------------------------------------------------------
# 3.1. As a base for coparison, do a simple 3DVar and 4DVar
from var3dfile import var3d
from var4dfile import var4d

print('*** compute 3DVar and 4DVar solutions ***')
ub3,ua3 = var3d(ug0,t,x,H,yobs,period_obs,gridobs,Bc_sq,invR)        
obsperwin = 2
ub4,ua4 = var4d(ug0,t,x,H,yobs,period_obs,obsperwin,gridobs,Bc_sq,invR)

rmseb3 = np.empty((Nsteps,2));     rmsea3 = np.empty((Nsteps,2));     
rmseb4 = np.empty((Nsteps,2));     rmsea4 = np.empty((Nsteps,2));

for job in range(2):
 rmseb3[:,job] = rmse_spread(ut[:,locs[job].astype(int)], \
                ub3[:,locs[job].astype(int)],None,1)
 rmsea3[:,job] = rmse_spread(ut[:,locs[job].astype(int)], \
                ua3[:,locs[job].astype(int)],None,1)
 rmseb4[:,job] = rmse_spread(ut[:,locs[job].astype(int)], \
                ub4[:,locs[job].astype(int)],None,1)
 rmsea4[:,job] = rmse_spread(ut[:,locs[job].astype(int)], \
                ua4[:,locs[job].astype(int)],None,1)
del job

plt.figure()
for jpl in range(Nx):
 plt.subplot(nrow,ncol,jpl+1)
 plt.plot(t,ut[:,jpl],'-k',label='truth')
 plt.plot(t,ub3[:,jpl],'-c',label='bgd 3DV') 
 plt.plot(t,ua3[:,jpl],'-m',label='ana 3DV')
 plt.plot(t,ub4[:,jpl],'-b',label='bgd 4DV')
 plt.plot(t,ua4[:,jpl],'-r',label='ana 4DV') 
 plt.xlabel('time')
 plt.ylabel('x['+str(jpl+1)+']')
 plt.ylim([-limp,limp])
 plt.grid(True)
 if jpl==Nx-1:
  plt.legend()
del jpl
for jpl in range(Nx_obs):
 plt.subplot(nrow,ncol,loc_obs[jpl]+1)
 plt.autoscale(False) 
 plt.scatter(tobs,yobs[:,jpl],20,'k')
 plt.ylim([-limp,limp])
del jpl
plt.subplots_adjust(top=0.955,bottom=0.08,left=0.11,right=0.9,hspace=0.465,wspace=0.345)



# 3.2. Compare 3DVar and SC-4DVar
plt.figure()
title_txt = ['observed variables','unobserved variables']
for job in range(2):
 plt.subplot(1,2,job+1)
 plt.plot(t,rmseb3[:,job],'-c.',label='bgd-3DVar')
 plt.plot(t,rmsea3[:,job],'-m.',label='ana-3DVar')
 plt.plot(t,rmseb4[:,job],'-b.',label='bgd-4DVar')
 plt.plot(t,rmsea4[:,job],'-r.',label='ana-4DVar')
 plt.legend()
 plt.title(title_txt[job])
 plt.xlabel('time')
 plt.ylabel('RMSE')
del job


# %%
#=============================================================================
# 4. Ensemble data assimilation
#import etkf16; reload(etkf16)
from etkf16 import getlocmat
lam = 2;  # localisation halfwidth
loctype = 1 #(Gaspari-Cohn)
Lxx = getlocmat(Nx,Nx,np.eye(Nx),lam,loctype) # get the localisation matrix
Lxy = getlocmat(Nx,Nx_obs,H,lam,loctype) # get the localisation matrix
loc_cmap = matplotlib.cm.get_cmap('gray_r')
print('***generate localisation matrix***')

fsize = 14
plt.figure()
plt.subplot(1,2,1)
plt.pcolor(np.flipud(Lxx),cmap=loc_cmap)
plt.tick_params(labelsize=fsize)  
plt.title('localisation in model space',fontsize=14)
plt.subplot(1,2,2)
plt.pcolor(np.flipud(Lxy),cmap=loc_cmap)
plt.tick_params(labelsize=fsize)  
plt.title('localisation in model/obs space',fontsize=14)
plt.subplots_adjust(top=0.955,bottom=0.08,left=0.11,right=0.8,hspace=0.465,wspace=0.345)
cax = plt.axes([0.85, 0.3, 0.025, 0.4])
cb = plt.colorbar(cax=cax)    



#%%
# ----------------------------------------------------------------------------
# 4.1 Do a LETKF and analyse the ensemble covariances
from etkf16 import etkf_l96
M = 10 #ensemble size
Ubkf,ubkf,Uakf,uakf = etkf_l96(ug0,t,x,M,Nx_obs,H,R,yobs,period_obs,lam,Lxy)
print('***Use LETKF***')

plt.figure()
for jpl in range(Nx):
 plt.subplot(nrow,ncol,jpl+1)
 plt.plot(t,ut[:,jpl],'-k',label='truth')
 plt.plot(t,Ubkf[:,jpl,:],'-c',label='bgd ens')
 plt.plot(t,Uakf[:,jpl,:],'-m',label='ana ens')
 plt.plot(t,ubkf[:,jpl],'-b',label='bgd mean')
 plt.plot(t,uakf[:,jpl],'-r',label='ana mean')
 plt.grid(True)
 plt.xlabel('time')
 plt.ylabel('x['+str(jpl+1)+']')
 plt.ylim([-limp,limp])
 if jpl==Nx-1:
  plt.legend()
del jpl
for jpl in range(Nx_obs):
 plt.subplot(nrow,ncol,loc_obs[jpl]+1)
 plt.autoscale(False) 
 plt.scatter(tobs,yobs[:,jpl],20,'k')
 plt.ylim([-limp,limp])
del jpl
plt.subplots_adjust(top=0.955,bottom=0.08,left=0.11,right=0.9,hspace=0.465,wspace=0.345)


# Compute the RMSE and spread
rmsebkf = np.empty((Nsteps,2));     rmseakf = np.empty((Nsteps,2));
for job in range(2):
 rmsebkf[:,job] = rmse_spread(ut[:,locs[job].astype(int)], \
                ubkf[:,locs[job].astype(int)],None,1)
 rmseakf[:,job] = rmse_spread(ut[:,locs[job].astype(int)], \
                uakf[:,locs[job].astype(int)],None,1)
del job

# 4.2. Do some comparisons in terms of RMSE
plt.figure()
title_txt = ['observed variables','unobserved variables']
for job in range(2):
 plt.subplot(1,2,job+1)
 plt.plot(t,rmseb3[:,job],'-c.',label='bgd-3DVar')
 plt.plot(t,rmsea3[:,job],'-m.',label='ana-3DVar')
 plt.plot(t,rmseb4[:,job],'-b.',label='bgd-4DVar')
 plt.plot(t,rmsea4[:,job],'-r.',label='ana-4DVar')
 plt.plot(t,rmsebkf[:,job],'-y.',label='bgd-LETKF')
 plt.plot(t,rmseakf[:,job],'-g.',label='ana-LETKF')
 plt.title(title_txt[job])
 plt.xlabel('time')
 plt.ylabel('RMSE')
 plt.legend()
del job


#%%
# ----------------------------------------------------------------------
# 5. Now let's compare the climatological Bc with some sample Bc's
nsample = 3
ind = np.arange(period_obs,(nsample+1)*period_obs,period_obs)
Pbs_kf = np.empty((Nx,Nx,nsample))
LPbs_kf = np.empty((Nx,Nx,nsample))
for j in range(nsample):
 aux = np.squeeze(Ubkf[ind[j],:,:])
 aux = np.cov(aux,ddof=1)
 Pbs_kf[:,:,j] = aux
 LPbs_kf[:,:,j] = Lxx*aux
del j
print('***compare climatological B with sample B***')

plt.figure()
lim = 1
my_cmap = matplotlib.cm.get_cmap('BrBG_r')
plt.subplot(nsample,3,1)
plt.imshow(np.array(Bc),interpolation="nearest",cmap=my_cmap,vmin=-lim,vmax=lim)
plt.title('Bc')
plt.colorbar()   
for j in range(nsample):
 plt.subplot(nsample,3,2+(j*3))   
 plt.imshow(np.array(Pbs_kf[:,:,j]),interpolation="nearest",cmap=my_cmap,vmin=-lim,vmax=lim)
 if j==0:
  plt.title('Pb')
 plt.colorbar()
 plt.subplot(nsample,3,3+(j*3))   
 plt.imshow(np.array(LPbs_kf[:,:,j]),interpolation="nearest",cmap=my_cmap,vmin=-lim,vmax=lim)
 if j==0:
  plt.title('Schur(L,Pb)')
 plt.colorbar()   
del j
plt.subplots_adjust(top=0.955,bottom=0.08,left=0.11,right=0.9,hspace=0.465,wspace=0.345)


#%%
###########################################################################
# 6 Hybrid DA part 1
lam = 1.5; loctype = 1
Lxx = getlocmat(Nx,Nx,np.eye(Nx),lam,loctype) 
Lxy = getlocmat(Nx,Nx_obs,H,lam,loctype) 

#import h4Dkf;   reload(h4Dkf)
from h4Dkf import etkf4DVar
loch = 1


# 6.1 Hybrid DA part 1
M = 10
beta = [0.8,0.4]
obsperwin = 2

print('***etkf4DVar***')
ubh,uah,Uaenh,uaenh = etkf4DVar(ug0,t,x,R,invR,H,yobs,period_obs,obsperwin,\
                           gridobs,Nx_obs,Bc,Bc_sq,lam,Lxx,Lxy,loch,M,beta)

plotens = 0

plt.figure()
for jpl in range(Nx):
 plt.subplot(nrow,ncol,jpl+1)
 plt.plot(t,ut[:,jpl],'-k',label='truth')
 plt.plot(t,ubh[:,jpl],'-b',label='bgd hyb')
 plt.plot(t,uah[:,jpl],'-r',label='ana hyb')
 if plotens!=0:
  plt.plot(t,Uaenh[:,jpl,:],'-y')
  plt.plot(t,uaenh[:,jpl],'-g')
 plt.ylim([-limp,limp])
 plt.grid(True)
 if jpl==Nx-1:
  plt.legend()   
del jpl
for jpl in range(Nx_obs):
 plt.subplot(nrow,ncol,loc_obs[jpl]+1)
 plt.autoscale(False) 
 plt.scatter(tobs,yobs[:,jpl],20,'grey')
 plt.ylim([-limp,limp])
del jpl
plt.subplots_adjust(top=0.955,bottom=0.08,left=0.11,right=0.9,hspace=0.465,wspace=0.345)


# RMSE
rmsebh = np.empty((Nsteps,2));     rmseah = np.empty((Nsteps,2));
for job in range(2):
 rmsebh[:,job] = rmse_spread(ut[:,locs[job].astype(int)], \
                np.squeeze(ubh[:,locs[job].astype(int)]),None,1)
 rmseah[:,job] = rmse_spread(ut[:,locs[job].astype(int)], \
                np.squeeze(uah[:,locs[job].astype(int)]),None,1)
del job

plt.figure()
title_txt = ['observed variables','unobserved variables']
for job in range(2):
 plt.subplot(1,2,job+1)
 plt.plot(t,rmsebh[:,job],'-c.',label='bgd-4dVar-LETKF')
 plt.plot(t,rmseah[:,job],'-m.',label='ana-4dVar-LETKF')
 plt.title(title_txt[job])
 plt.xlabel('time')
 plt.ylabel('RMSE')
 plt.legend()
del job


#%%
# 7. Hybrid DA part 2: Avoiding the TLM and adjoint
# 7.1, Evolution of covariances by two ways
from transmat import transmat_l96
from misc_funs import evolcov, covfamrun

print('*** evolving the covariance matrices ***')
uref, tmat, seed = transmat_l96(ug0,t,x) # compute the TL matrix linearised 
                                            # about the background traj
                                            
lags = 5 
Bt, B0t = evolcov(Bc,tmat,Nx,lags)

M = 10 # number of ensemble members
Ufam,Pbt,Pb0t = covfamrun(ug0,Nx,lags,Bc_sq,M)

lim = 2

for jpl in range(2):
 if jpl==0:
  Bplot = Bt;   Pbplot = Pbt
 if jpl==1:
  Bplot = B0t;  Pbplot = Pb0t 
 plt.figure() 
 my_cmap = matplotlib.cm.get_cmap('BrBG_r')
 for jlags in range(0,lags,1):
  if jpl==0:   
   title_text = 'Cov(t='+str(jlags)+')'
  if jpl==1:
   title_text = 'Cov(0,t='+str(jlags)+')'

  plt.subplot(5,lags,1+jlags)
  plt.imshow(np.array(Bplot[:,:,jlags]),cmap=my_cmap,vmin=-lim,vmax=lim)   
  plt.title('exact ' +title_text)
  
  plt.subplot(5,lags,1+2*lags+jlags)
  plt.imshow(np.array(Pbplot[:,:,jlags]),cmap=my_cmap,vmin=-lim,vmax=lim)  
  plt.title('raw ens '+ title_text)   
  
  plt.subplot(5,lags,1+4*lags+jlags)
  plt.imshow(np.array(Lxx*Pbplot[:,:,jlags]),cmap=my_cmap,vmin=-lim,vmax=lim)  
  plt.title('loc ens' + title_text)     
  
 del jlags 
 plt.subplots_adjust(top=0.955,bottom=0.08,left=0.11,right=0.9,hspace=0.465,wspace=0.345)
del jpl


#%%
# ---------------------------------------------------------------------------
# 7.2. SC-4DEnVar
obsperwin = 2;
lam = 1.5; loctype = 1
Lxx = getlocmat(Nx,Nx,np.eye(Nx),lam,loctype) # get the localisation matrix
Lxy = getlocmat(Nx,Nx_obs,H,lam,loctype) # get the localisation matrix


# 3.3. 4DENVAR-state-sc
#import inc4DenV; reload(inc4DenV)
print('***perform SC4DEnVar***')
from inc4DenV import envar
M = 10;  locenvar = 1; 
ua4Den,ub4Den,Uaen4Den,uaen4Den,UFr4Den = envar(ug0,t,x,R,invR,H,yobs,\
       period_obs,obsperwin,gridobs,Nx_obs,Bc_sq,lam,Lxx,Lxy,locenvar,M)
                        
    
plt.figure()
for jpl in range(Nx):
 plt.subplot(nrow,ncol,jpl+1)
 plt.plot(t,ut[:,jpl],'-k',label='truth')
 plt.plot(t,ub4Den[:,jpl],'-b',label='bgd 4DenV')
 plt.plot(t,ua4Den[:,jpl],'-r',label='ana 4DenV')
 plt.title('x['+str(jpl+1)+']')
 plt.ylim([-limp,limp])
 plt.grid(True)
 if jpl==Nx-1:
  plt.legend()   
del jpl
for jpl in range(Nx_obs):
 plt.subplot(nrow,ncol,loc_obs[jpl]+1)
 plt.autoscale(False) 
 plt.scatter(tobs,yobs[:,jpl],20,'grey')
 plt.ylim([-limp,limp])
del jpl
plt.subplots_adjust(top=0.955,bottom=0.08,left=0.11,right=0.9,hspace=0.465,wspace=0.345)


# RMSE
rmseb4Den = np.empty((Nsteps,2));     rmsea4Den = np.empty((Nsteps,2));

for job in range(2):
 rmseb4Den[:,job] = rmse_spread(ut[:,locs[job].astype(int)], \
                np.squeeze(ub4Den[:,locs[job].astype(int)]),None,1)
 rmsea4Den[:,job] = rmse_spread(ut[:,locs[job].astype(int)], \
                np.squeeze(ua4Den[:,locs[job].astype(int)]),None,1)
del job

plt.figure()
title_txt = ['observed variables','unobserved variables']
for job in range(2):
 plt.subplot(1,2,job+1)
 plt.plot(t,rmseb4Den[:,job],'-c.',label='bgd-4dEnVar')
 plt.plot(t,rmsea4Den[:,job],'-m.',label='ana-4dEnVar')
 plt.title(title_txt[job])
 plt.xlabel('time')
 plt.ylabel('RMSE')
 plt.legend()
del job
















