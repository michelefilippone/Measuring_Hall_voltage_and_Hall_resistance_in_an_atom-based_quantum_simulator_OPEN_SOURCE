from __future__ import division
import itertools
import datetime
import numpy as np 
import scipy as sp 
import scipy.integrate
import math
import cmath
import time
import random
from scipy import interpolate
import os
from scipy import special
import sys

from FUNCTIONS_2_LEG_LADDER import *


pi=math.pi
I=1j
inf=np.inf

###############################
#NON-INTERACTING CODE PBC
###############################
def specplus(k,chi,N,tau):
    return -(np.cos(k)*np.cos(chi/2./N)+( (np.sin(k)*np.sin(chi/2./N))**2.+tau**2.)**.5)

def specminus(k,chi,N,tau):
    return -(np.cos(k)*np.cos(chi/2./N)-( (np.sin(k)*np.sin(chi/2./N))**2. +tau**2.)**.5)

def vplus(k,chi,N,tau):
    return np.sin(k)*(np.cos(chi/2./N)-np.cos(k)*np.sin(chi/2./N)**2./( ( (np.sin(k)*np.sin(chi/2./N))**2.+tau**2.)**.5) ) 

def vminus(k,chi,N,tau):
    return np.sin(k)*(np.cos(chi/2./N)+np.cos(k)*np.sin(chi/2./N)**2./(( (np.sin(k)*np.sin(chi/2./N))**2.+tau**2.)**.5) )
    
def coeff_imbalance(k,chi,N,tau):
	return np.sin(k)*np.sin(chi/2./N)/(np.sin(k)**2.*np.sin(chi/2./N)**2+tau**2.)**.5
	 
def ground_state_energy(phi,chi,N,tau,Num):
    shift=int(phi/2./pi+chi/4./pi)
    idx=np.arange(0,N)
    kappas=2.*pi*idx/N+phi/N+chi/2/N
    ep=specplus(kappas,chi,N,tau).tolist()
    em=specminus(kappas,chi,N,tau).tolist()
    datay=ep+em
    datay.sort()
    return np.sum([ datay[x] for x in range (Num)])

def ground_state_current(phi,chi,N,tau,Num):
    shift=int(phi/2./pi+chi/4./pi)
    idx=np.arange(0,N)
    kappas=2.*pi*idx/N+phi/N+chi/2/N
    ep=specplus(kappas,chi,N,tau).tolist()
    em=specminus(kappas,chi,N,tau).tolist()
    vp=vplus(kappas,chi,N,tau).tolist()
    vm=vminus(kappas,chi,N,tau).tolist()
    energies=np.array(ep+em)
    velocities=np.array(vp+vm)
    velocities_sorted=velocities[np.argsort(energies)]
    return -np.sum([velocities_sorted[x] for x in range (Num)])/N

def ground_state_dens_imbalance(phi,chi,N,tau,Num):
    shift=int(phi/2./pi+chi/4./pi)
    idx=np.arange(0,N)
    kappas=2.*pi*idx/N+phi/N+chi/2/N
    ep=specplus(kappas,chi,N,tau).tolist()
    em=specminus(kappas,chi,N,tau).tolist()
    coeff_plus=coeff_imbalance(kappas,chi,N,tau)
    coeff_minus=-coeff_imbalance(kappas,chi,N,tau)
    coeffs=coeff_plus.tolist()+coeff_minus.tolist()
    energies=np.array(ep+em)
    coefficienti=np.array(coeffs)
    coefficienti_sorted=coefficienti[np.argsort(energies)]
    return -np.sum([coefficienti_sorted[x] for x in range (Num)])/N




###############################
#NON-INTERACTING CODE OBC
#SIMULATES HAMILTONIAN OF THE FORM
#
#H=-tpara \sum_{j,sigma}[e^{I*chi/2}c*_{j,sigma}c_{j+1,sigma}+h.c.]-tperp\sum_{j}[c*_{j,up}c*_{j,down}+h.c.]
#+mu\sum_{j,sigma} j c*_{j,sigma}c_{j,sigma}
#
#with Open Boundary Conditions (OBC). For OBC the model is gauge equivalent to  ( c_{j,sigma} -> e^{-I sigma j chi /2}c_{j,sigma} ) 
#
#H=-tpara \sum_{j,sigma}[c*_{j,sigma}c_{j+1,sigma}+h.c.]-tperp\sum_{j}[e*{I chi j}c*_{j,up}c_{j,down}+h.c.]
#+mu\sum_{j,sigma} j c*_{j,sigma}c_{j,sigma}
#
#This last version is more practical to run in DMRG (I guess...)
###############################


#BUILDS THE HAMILTONIAN with L rungs
def Hamiltonian_Free_OBC(L,tpara,tperp,mu,chi,W,Ey=0):
    pot=np.diag([x for x in range (L)]*2)
    confinement=np.diag([(x-L/2.)**2 for x in range (L)]*2)
    hop_intra_up_right=np.diag(np.ones(L-1).tolist()+np.zeros(L).tolist(),1)
    hop_intra_up_left=np.diag(np.ones(L-1).tolist()+np.zeros(L).tolist(),-1)
    hop_intra_down_right=np.diag(np.zeros(L).tolist()+np.ones(L-1).tolist(),1)
    hop_intra_down_left=np.diag(np.zeros(L).tolist()+np.ones(L-1).tolist(),-1)
    hop_inter_up=np.diag(np.ones(L),L)
    hop_inter_down=np.diag(np.ones(L),-L)
    transverse_field=np.diag(np.ones(L).tolist()+(-np.ones(L)).tolist())
    ham=-mu*pot-tpara*(np.exp(I*chi/2.)*hop_intra_up_right+np.exp(-I*chi/2.)*hop_intra_up_left)\
        -tpara*(np.exp(-I*chi/2.)*hop_intra_down_right+np.exp(I*chi/2.)*hop_intra_down_left)\
        -tperp*(hop_inter_up+hop_inter_down)+W*confinement+Ey*transverse_field
    return ham


#Returns the total current operator
def current_free_obc_op(L,tpara,chi):
    hop_intra_up_right=np.diag(np.ones(L-1).tolist()+np.zeros(L).tolist(),1)
    hop_intra_up_left=np.diag(np.ones(L-1).tolist()+np.zeros(L).tolist(),-1)
    hop_intra_down_right=np.diag(np.zeros(L).tolist()+np.ones(L-1).tolist(),1)
    hop_intra_down_left=np.diag(np.zeros(L).tolist()+np.ones(L-1).tolist(),-1)
    j=-tpara*(I*np.exp(I*chi/2.)*hop_intra_up_right-I*np.exp(-I*chi/2.)*hop_intra_up_left)\
      -tpara*(I*np.exp(-I*chi/2.)*hop_intra_down_right-I*np.exp(I*chi/2.)*hop_intra_down_left)     
    return j

#Returns the total polarization operator along the transverse direction 'y'
def polarization_free_obc_op(L):
    py=np.diag(np.ones(L).tolist()+(-np.ones(L)).tolist())
    return py
    

#Returns the Ground State energy of the ground state for L rungs and M particles 
def GS_energy_OBC(L,M,tpara,tperp,mu=0,chi=0,W=0):
    return np.sum([sp.linalg.eigh(Hamiltonian_Free_OBC(L,tpara,tperp,mu,chi,W))[0][x] for x in range (0,M)])    


def fermi_factor(e,T,chem_pot):
	return 1./(1.+np.exp((e-chem_pot)/T))

#Returns the the average value of the current j and the transverse polarization of py as a funtion of time after a 
#quench of the electric field (controlled by mu), on the ground state (which is calculated for mu=0)
def evolve_free_obc(L,M,tpara,tperp,mu,chi,dt,tmax,W=0,T=0,chem_pot=0,Ey=0):
	[spec0,eigvec0]=sp.linalg.eigh(Hamiltonian_Free_OBC(L,tpara,tperp,0,chi,W,Ey))
	eig0mat=np.matrix(eigvec0)
	[spec,eigvec]=sp.linalg.eigh(Hamiltonian_Free_OBC(L,tpara,tperp,mu,chi,W,Ey))
	j_op=np.matrix(current_free_obc_op(L,tpara,chi))
	py_op=np.matrix(polarization_free_obc_op(L))
	times=np.arange(0,tmax,dt)
	list_j=[]
	list_py=[]
	if T==0:
		for t in times:
			evol_op=np.matrix(Evolution_Operator(t,spec,eigvec))
			jt=evol_op.H*j_op*evol_op
			pyt=evol_op.H*py_op*evol_op
			j_mat=eig0mat.H*jt*eig0mat
			py_mat=eig0mat.H*pyt*eig0mat
			list_j.append(np.sum([j_mat[x,x] for x in range (M)]))
			list_py.append(np.sum([py_mat[x,x] for x in range (M)]))
	#FINITE TEMPERATURE
	else:
		for t in times:
			evol_op=np.matrix(Evolution_Operator(t,spec,eigvec))
			jt=evol_op.H*j_op*evol_op
			pyt=evol_op.H*py_op*evol_op
			j_mat=eig0mat.H*jt*eig0mat
			py_mat=eig0mat.H*pyt*eig0mat
			list_j.append(np.sum([j_mat[x,x]*fermi_factor(spec0[x],T,chem_pot) for x in range (2*L)]))
			list_py.append(np.sum([py_mat[x,x]*fermi_factor(spec0[x],T,chem_pot) for x in range (2*L)]))
	return [times,list_j,list_py]
	
def evolve_free_obc_new(L,M,tpara,chi,spec0,eigvec0,spec,eigvec,dt,tmax,T=0,chem_pot=0):
	eig0mat=np.matrix(eigvec0)
	j_op=np.matrix(current_free_obc_op(L,tpara,chi))
	py_op=np.matrix(polarization_free_obc_op(L))
	times=np.arange(0,tmax,dt)
	list_j=[]
	list_py=[]
	if T==0:
		list_evol_op=[np.matrix(Evolution_Operator(t,spec,eigvec))*eig0mat for t in times]    
		list_jt=[U.H*j_op*U for U in list_evol_op]
		list_pyt=[U.H*py_op*U for U in list_evol_op]
		list_j=[np.sum([j_mat[x,x] for x in range (M)]) for j_mat in list_jt]        
		list_py=[np.sum([py_mat[x,x] for x in range (M)]) for py_mat in list_pyt]        
	#FINITE TEMPERATURE
	else:
		list_evol_op=[np.matrix(Evolution_Operator(t,spec,eigvec))*eig0mat for t in times]    
		list_jt=[U.H*j_op*U for U in list_evol_op]
		list_pyt=[U.H*py_op*U for U in list_evol_op]
		list_j=[np.sum([j_mat[x,x]*fermi_factor(spec0[x],T,chem_pot) for x in range (2*L)]) for j_mat in list_jt]        
		list_py=[np.sum([py_mat[x,x]*fermi_factor(spec0[x],T,chem_pot) for x in range (2*L)]) for py_mat in list_pyt]        
	return [times,list_j,list_py]
	
	
	
def number_of_particles_vs_chemical_potential(spec,T,chem_pot):
	weighted_summands=[1./(np.exp((x-chem_pot)/T)+1.) for x in spec]
	N=np.sum(weighted_summands)
	return N

##############################################################
#FINITE TEMPERATURE MICROCANONICAL ENSEMBLE
##############################################################

def spec_non_int(tx,ty,k,chi,s):
    return -2*tx*np.cos(k)*np.cos(chi/2.)+s*((2.*tx*np.sin(k)*np.sin(chi/2.))**2.+ty**2.)**(0.5)

def dffe(tx, ty, k, chi, s):
    return 2.*tx*(np.cos(k)*np.cos(chi/2.)\
                  + 2.*s*(tx*ty**2.*np.cos(2.*k)*np.sin(chi/2.)**2.-4.*tx**3.*np.sin(k)**4.*np.sin(chi/2)**4.)\
                  /(4.*tx**2.*np.sin(k)**2.*np.sin(chi/2.)**2.+ty**2.)**(3./2.)
                 )

def dnfe(tx, ty, k, chi, s):
    return 2*s*tx*ty**2.*np.cos(k)*np.sin(chi/2)/(4.*tx**2.*np.sin(k)**2.*np.sin(chi/2.)**2.+ty**2.)**(3./2.)

#return occupation basis with L sites and N particles (remind we have two rungs)
def occupation_basis(L,N):
    return list(place_ones(2*L,N))

#This function produces the single-body observables for all the single-particle configurations
def configuration_observable(o_list,occupation_list):
    observables=[]
    for l in occupation_list:
        observables.append(np.dot(o_list,l))
    return observables

#This function gives the (not-normalized) thermal average of a single-particle observable
def config_thermal_average(beta,config_energy,config_obs):
    thermal_prefactors=[np.exp(-beta*x) for x in config_energy]
    return np.dot(thermal_prefactors,config_obs)
    

#Hall Response : PH 
def PH(beta,tx,ty,chi,L,N,s='mute'):
    if s!='mute':   print('Calculate configurations...')
    configs=occupation_basis(L,N)
    if s!='mute': print('Calculate single particle energies and derivatives...')
    e_list=[spec_non_int(tx,ty,2*np.pi*n/L,chi,s) for n in range (0,L) for s in [-1,1]]
    dffe_list=[dffe(tx,ty,2*np.pi*n/L,chi,s) for n in range (0,L) for s in [-1,1]]
    dnfe_list=[dnfe(tx,ty,2*np.pi*n/L,chi,s) for n in range (0,L) for s in [-1,1]]
    if s!='mute': print('Calculate configuration sums...')
    config_energies=configuration_observable(e_list,configs)
    config_dffe=configuration_observable(dffe_list,configs)
    config_dnfe=configuration_observable(dnfe_list,configs)
    if s!='mute': print('Calculate PH...')
    numerator=config_thermal_average(beta,config_energies,config_dnfe)
    denominator=config_thermal_average(beta,config_energies,config_dffe)
    return numerator/denominator

