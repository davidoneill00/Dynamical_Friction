import numpy as np
import matplotlib
matplotlib.use('Agg')
import sys
import matplotlib.pyplot as plt
from matplotlib import pyplot
import math as ma
import EllipFncs as epf
import os
from scipy.interpolate import interp1d

# Passing the eccentricity and Mach number at perihelion

ecc0 = float(sys.argv[1])
Mp = float(sys.argv[2])

# We import the grid

grid = epf.Grid(Mp,ecc0)

# Now we want to look at the convergence in box size. Therefore we introduce an array L over the range of box sizes

L = np.arange(grid.subsize-1,grid.Nr+grid.subsize-1,grid.subsize)    
H = [grid.r[x] for x in L]

# Next we want to create a more densely sampled time domain. This is because it is expensive to evaluate many points on an orbit. To 
# avoid this we can interpolate over a more densely sampled time domain 

InterpTime = np.arange(grid.T[0],grid.T[len(grid.T)-2],0.001)
ephi = [epf.EPhi(t,ecc0) for t in InterpTime]
F = [epf.TrueAnomaly(t,ecc0) for t in InterpTime]

# Initialize arrays

adot = []
edot = []

# Now we begin the first loop over Box Size. This iterates each time for a slightly bigger box.

for j in range(0,len(H)):

# Initializing arrays again. We want these to wipe for each independent box size

    Radi = np.zeros(len(grid.T))
    Azi = np.zeros(len(grid.T))

# Now we import the forces and collect them in an array Azi (Azimuthal). Each element in this array is for a different time

    for i in range(0,len(grid.T)):
         NAME = 'BoxConv%gt_%gecc_%gMp_.txt' %(grid.T[i],ecc0,Mp)
         with open("/groups/astro/davidon/GDF_Python/SteadyStates/BoxRes/ecc%g/"%(ecc0) + NAME,'r') as ap:
            Azi[i] = np.genfromtxt(ap,delimiter=',')[0+2*j]

# Same thing for the Radial component of the force

    for i in range(0,len(grid.T)):
        NAME = 'BoxConv%gt_%gecc_%gMp_.txt' %(grid.T[i],ecc0,Mp)                    
        with open("/groups/astro/davidon/GDF_Python/SteadyStates/BoxRes/ecc%g/"%(ecc0) + NAME,'r') as ap:
            Radi[i] = float(np.genfromtxt(ap,delimiter=',')[1+2*j])

# Now we need to interpolate these forces. We want a better covering of the orbit over time

    FAZI = interp1d(grid.T,Azi,kind='cubic') 
    FRAD = interp1d(grid.T,Radi,kind='cubic')

# For which we can evaluate the interpolated functions over this domian.

    InterpAzi = np.array([FAZI(t) for t in InterpTime])
    InterpRad = np.array([FRAD(t) for t in InterpTime])

    ADOT = np.trapz((2/np.sqrt(1-ecc0**2))*(-InterpRad*ecc0*np.sin(F)-InterpAzi*(1+ecc0*np.cos(F))),InterpTime,axis=0)
#    ADOT = np.mean((2/np.sqrt(1-ecc0**2))*(-InterpRad*ecc0*np.sin(F)-InterpAzi*(1+ecc0*np.cos(F))))
    adot.append(ADOT)
    EDOT = np.trapz((np.sqrt(1-ecc0**2))*(-InterpRad*np.sin(F)-InterpAzi*(np.cos(F)+np.cos(ephi))),InterpTime, axis=0)
#    EDOT = np.mean((np.sqrt(1-ecc0**2))*(-InterpRad*np.sin(F)-InterpAzi*(np.cos(F)+np.cos(ephi))))
    edot.append(EDOT)

Orbital_Elements = [adot[len(adot)-1],edot[len(edot)-1]]

SavingName = 'adot%gecc_%gMp_%gA_%gNr_%gNtheta_.txt' %(ecc0,Mp,grid.A,grid.Nr,grid.Ntheta)
with open(os.path.join('/groups/astro/davidon/GDF_Python/SteadyStates/ecc%g/' %(ecc0),SavingName), "w") as file:
    np.savetxt(file, Orbital_Elements, fmt="%f")
file.close()


plt.figure()
plt.plot(H,adot, color='purple',label='adot')
plt.plot(H,edot, color='orange',label='edot')
plt.legend(frameon=False)
plt.xlabel('Box Size (Rp)')
plt.title('OrbitalElements-BoxSize%gecc_%gMp'%(ecc0,Mp))
plt.savefig('OrbitalElements-BoxSize%gecc_%gMp.png'%(ecc0,Mp))





