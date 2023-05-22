import numpy as np
import matplotlib
matplotlib.use('Agg')
import sys
import matplotlib.pyplot as plt
from matplotlib import pyplot
import math as ma
import AltEllipFncs as apf
import os
from scipy.interpolate import interp1d

# Passing the eccentricity and Mach number at perihelion
ecc0 = float(sys.argv[1])
Mp = float(sys.argv[2])

# We import the grid and redefine all of the parameters
grid = apf.Grid(Mp,ecc0)

r       =   grid.r
theta   =   grid.theta
phi     =   grid.phi
T       =   grid.T
Ntheta  =   grid.Ntheta
Nr      =   grid.Nr
Nphi    =   grid.Nphi
A       =   grid.A
subsize =   grid.subsize

# Now we want to look at the convergence in box size. Therefore we introduce an array L over the range of box sizes
L = np.arange(subsize-1,Nr+subsize-1,subsize)    
H = [r[x] for x in L]

print(H[len(H)-1])

# Next we want to create a more densely sampled time domain. This is because it is expensive to evaluate many points on an orbit. To 
# avoid this we can interpolate over a more densely sampled time domain 
InterpTime = np.arange(T[0],T[len(T)-1],0.001)
ephi = [apf.EPhi(t,ecc0) for t in InterpTime]
F = [apf.TrueAnomaly(t,ecc0) for t in InterpTime]

# Initialize arrays
adot = []
edot = []

# Now we begin the first loop over Box Size. This iterates each time for a slightly bigger box.
for j in range(0,len(H)):

# Initializing arrays again. We want these to wipe for each independent box size
    Radi = np.zeros(len(T))
    Azi = np.zeros(len(T))

# Now we import the forces and collect them in an array Azi (Azimuthal). Each element in this array is for a different time
    for i in range(0,len(T)):
         NAME = 'BoxConv%gt_%gecc_%gMp_.txt' %(T[i],ecc0,Mp)
         with open("/groups/astro/davidon/GDF_Python/SteadyStates/BoxRes/ecc%g/"%(ecc0) + NAME,'r') as ap:
            Azi[i] = np.genfromtxt(ap,delimiter=',')[0+2*j]

# Same thing for the Radial component of the force
    for i in range(0,len(T)):
        NAME = 'BoxConv%gt_%gecc_%gMp_.txt' %(T[i],ecc0,Mp)                    
        with open("/groups/astro/davidon/GDF_Python/SteadyStates/BoxRes/ecc%g/"%(ecc0) + NAME,'r') as ap:
            Radi[i] = float(np.genfromtxt(ap,delimiter=',')[1+2*j])

# Now we need to interpolate these forces. We want a better covering of the orbit over time
    FAZI = interp1d(T,Azi,kind='cubic') #Maybe spline? Or even try a Fourier series?
    FRAD = interp1d(T,Radi,kind='cubic')

# For which we can evaluate the interpolated functions over this domian.
    InterpAzi = np.array([FAZI(t) for t in InterpTime])
    InterpRad = np.array([FRAD(t) for t in InterpTime])


    ADOT = np.mean((2/(np.sqrt(1-ecc0**2)))*(-InterpRad*ecc0*np.sin(F)-InterpAzi*(1+ecc0*np.cos(F))))
    adot.append(ADOT)

    EDOT = np.mean((np.sqrt(1-ecc0**2))*(-InterpRad*np.sin(F)-InterpAzi*(np.cos(F)+np.cos(ephi))))
    edot.append(EDOT)



plt.figure()
plt.plot(H,adot, color='purple',label='adot')
plt.plot(H,edot, color='orange',label='edot')
plt.legend(frameon=False)
plt.xlabel('Box Size (Rp)')
plt.title('OrbitalElements-BoxSize%gecc_%gMp'%(ecc0,Mp))
plt.savefig('OrbitalElements-BoxSize%gecc_%gMp.png'%(ecc0,Mp))

#DIFFA = np.zeros(BoxResSize)
#DIFFE =np.zeros(BoxResSize)
#for n in range(1,BoxResSize):
#    DIFFA[n] = np.abs((adot[n]-adot[n-1])/(adot[n-1]))
#    DIFFE[n] = np.abs((edot[n]-edot[n-1])/edot[n-1])

#plt.figure()
#plt.plot(L,DIFFA, color='purple',label='dadot/da')
#plt.plot(L,DIFFE, color='orange',label='dedot/de')
#plt.title('OrbitalElements-Change%gecc_%gMp'%(ecc0,Mp))
#plt.legend(frameon=False)
#plt.ylim([-0.1,0.1])
#plt.axhline(y=DIFFA[BoxResSize-1],color='purple',linestyle='dashed')
#plt.axhline(y=DIFFE[BoxResSize-1],color='orange',linestyle='dashed')
#plt.axhline(y=0,color='gray')
#plt.text(3.2,DIFFA[BoxResSize-1]-0.01,'%g'%(np.round(DIFFA[BoxResSize-1],3)),color='purple')
#plt.text(3.2,DIFFE[BoxResSize-1]-0.01,'%g'%(np.round(DIFFE[BoxResSize-1],3)),color='orange')
#plt.savefig('OrbitalElements-Change%gecc_%gMp.png'%(ecc0,Mp))
