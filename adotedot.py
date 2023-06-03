import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import sys
import matplotlib.pyplot as plt
from matplotlib import pyplot
import math as ma
import EllipFncs as epf
import matplotlib.cm as cm
import os
from scipy.interpolate import interp2d
rc_fonts = {"text.usetex": True,}
plt.rcParams.update(rc_fonts)


# We want to plot the changing orbital elements over some grid:

ecc0 = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
Mp = np.array([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])

# Now we ant to import our data

adot=np.zeros([len(ecc0),len(Mp)])
edot=np.zeros([len(ecc0),len(Mp)])

for j in range(1,len(ecc0)):
    for m in range(1,len(Mp)):
        grid = epf.Grid(Mp[m],ecc0[j])
        SavingName = 'adot%gecc_%gMp_%gA_%gNr_%gNtheta_.txt' %(ecc0[j],Mp[m],grid.A,grid.Nr,grid.Ntheta)
        with open("/groups/astro/davidon/GDF_Python/SteadyStates/ecc%g/"%(ecc0[j]) + SavingName,'r') as ap:
            Orbital_Elements = np.array(np.genfromtxt(ap,delimiter=None))
        ap.close()

        adot[j][m] = Orbital_Elements[0]
        edot[j][m] = Orbital_Elements[1]

# All non-zero eccentricity adot, edots have been imported. Now we will do the same for the Kim and Kim adot.

Ts =np.arange(2,3,0.001)
kkadot = np.zeros([len(ecc0),len(Mp)])
kkedot = np.zeros([len(ecc0),len(Mp)])
EPHI = [epf.EPhi(t,0) for t in Ts]
FS   = [epf.TrueAnomaly(t,0) for t in Ts]
for k in range(0,len(ecc0)):
    for j in range(0,len(Mp)):
        Azi = np.array([epf.PolarKK(t,ecc0[k],Mp[j])[0] for t in Ts])
        Radi= np.array([epf.PolarKK(t,ecc0[k],Mp[j])[1] for t in Ts])
        kkadot[k][j] = np.mean((2/(np.sqrt(1-ecc0[k]**2)))*(-Radi*ecc0[k]*np.sin(FS)-Azi*(1+ecc0[k]*np.cos(FS))))
        kkedot[k][j] = np.mean((np.sqrt(1-ecc0[k]**2))*(-Radi*np.sin(FS)-Azi*(np.cos(FS)+np.cos(EPHI))))
        
# Since both agree at eccentricity 0 we can include the zero eccentricity values in our grid

adot[:][0] = kkadot[:][0]
edot[:][0] = kkedot[:][0] 

print('adot',adot)
print('edot',edot)

# We want to see where the Kim & Kim approximation holds and where it breaks down

DifferenceInAdot = adot - kkadot
DifferenceInEdot = edot - kkedot

# Plots 1 & 2 showing the difference in Kim & Kim proxy to our findings

plt.figure()
plt.xlabel('Eccentricity', fontsize=20)
plt.ylabel('Mach', fontsize=20)
mpl.pyplot.xticks(fontsize=16)
mpl.pyplot.yticks(fontsize=16)
plt.title(r'Difference in $\dot{a}/a$ to Kim and Kim',fontsize=25)
plt.contourf(ecc0,Mp,np.absolute(DifferenceInAdot.T),100)
plt.colorbar()
plt.savefig('aDotDiff.png')

plt.figure()
plt.xlabel('Eccentricity', fontsize=20)
plt.ylabel('Mach', fontsize=20)
mpl.pyplot.xticks(fontsize=16)
mpl.pyplot.yticks(fontsize=16)
plt.title(r'Difference in $\dot{e}$ to Kim and Kim',fontsize=25)
plt.contourf(ecc0,Mp,np.absolute(DifferenceInEdot.T),100)
plt.colorbar()
plt.savefig('eDotDiff.png')



# Now we can interpolate over our data to give a smoother covering

AdotInterpFunc = interp2d(Mp,ecc0,adot,kind = 'linear')
EdotInterpFunc = interp2d(Mp,ecc0,edot,kind = 'linear')

NEWECC = np.linspace(min(ecc0)+0.001,max(ecc0)-0.001,5000)
NEWMACH= np.linspace(min(Mp)+0.001,max(Mp)-0.001,5000)

ADOTRESULT = AdotInterpFunc(NEWMACH,NEWECC)
EDOTRESULT = EdotInterpFunc(NEWMACH,NEWECC)


cmap1 = mpl.cm.turbo(np.linspace(-1.2, 0.3, 200))
cmap2 = mpl.cm.afmhot_r(np.linspace(0, 0.9, 200))

cmap1 = mpl.colors.ListedColormap(cmap1[158:, :-1])
cmap2 = mpl.colors.ListedColormap(cmap2[:, :-1])


colors = np.vstack((cmap2.colors, cmap1.colors))
my_cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors)

plt.figure()
plt.contourf(NEWECC, NEWMACH, ADOTRESULT.T, cmap=my_cmap, levels=np.linspace(-2, 0.4, 200))
plt.colorbar(ticks = [-2.65,-2.32,-1.99,-1.66,-1.32,-1,-0.66, -0.33,0,0.33])
plt.title(r'Change in Semi-Major axis $\dot{a}/a$',fontsize = 15)
plt.xlabel('Eccentricity',fontsize = 15)
plt.ylabel(r'$M_p$',fontsize=18)
plt.xticks(np.arange(0,1,0.1))
plt.savefig('adot.png')









cmap1a = mpl.cm.magma_r(np.linspace(-4, 1, 200))
cmap2a = mpl.cm.afmhot(np.linspace(0, 0.9, 200))

cmap1a = mpl.colors.ListedColormap(cmap1a[158:, :-1])
cmap2a = mpl.colors.ListedColormap(cmap2a[:, :-1])


colors = np.vstack((cmap1a.colors, cmap2a.colors))
my_cmapa = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors)

plt.figure()
plt.contourf(NEWECC, NEWMACH, EDOTRESULT.T,500, cmap = my_cmapa)
plt.colorbar()
plt.title(r'Change in Eccentricity $\dot{e}$',fontsize=15)
plt.xlabel('Eccentricity',fontsize = 15)
plt.ylabel(r'$M_p$',fontsize=16)
plt.xticks(np.arange(0,1,0.1))
plt.savefig('edot.png')
