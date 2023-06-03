import numpy as np
import matplotlib
matplotlib.use('Agg')
import sys
import matplotlib.pyplot as plt
from matplotlib import pyplot
import math as ma
import EllipFncs as epf
from scipy.interpolate import interp1d
import os
rc_fonts = {"text.usetex": True,}
plt.rcParams.update(rc_fonts)
IncludeLegend = False

# Initialise parameters and create the grid

ecc0 = float(sys.argv[1])
Mp = float(sys.argv[2])

grid = epf.Grid(Mp,ecc0)

T = np.append(grid.T,grid.T[0]+1)
print(T)
# Now we will import the forces found from our force computational files

Radi = np.zeros(len(T))
Azi = np.zeros(len(T))
for i in range(0,len(T)-1):
    NAME ='Polar%gt_%gecc_%gMp_%gA_%gNr_%gNtheta_.txt' %(T[i],ecc0,Mp,grid.A,grid.Nr,grid.Ntheta)
    with open("/groups/astro/davidon/GDF_Python/SteadyStates/ecc%g/"%(ecc0) + NAME,'r') as ap:
        Azi[i] = np.genfromtxt(ap,delimiter=',')[0]

for i in range(0,len(T)-1):
    NAME ='Polar%gt_%gecc_%gMp_%gA_%gNr_%gNtheta_.txt' %(T[i],ecc0,Mp,grid.A,grid.Nr,grid.Ntheta)
    with open("/groups/astro/davidon/GDF_Python/SteadyStates/ecc%g/"%(ecc0) + NAME,'r') as ap:
        Radi[i] = float(np.genfromtxt(ap,delimiter=',')[1])

Azi[len(grid.T)] = Azi[0]
Radi[len(grid.T)] = Radi[0]
# Performing an interpolation over the data points

print(Azi)

FAZI = interp1d(T,Azi,kind='linear') 
FRAD = interp1d(T,Radi,kind='linear')

InterpTime = np.arange(T[0],T[len(T)-1]-0.001,0.001)
InterpAzi = np.array([FAZI(t) for t in InterpTime])
InterpRad = np.array([FRAD(t) for t in InterpTime])


#Next we want to compare to Kim and Kim. We don't need to change the soundspeed at all because the only place it shows up is in the 1/c2 prefactor. However if we use the same phi, the velocity is quite high so we need to normalise it such that the velocity at pericenter is Mp.

KK_ForceRad = [epf.PolarKK(t,ecc0,Mp)[1] for t in InterpTime]
KK_ForceAzi = [epf.PolarKK(t,ecc0,Mp)[0] for t in InterpTime]
O99_ForceRad= [epf.PolarO99(t,ecc0,Mp)[1] for t in InterpTime]
O99_ForceAzi= [epf.PolarO99(t,ecc0,Mp)[0] for t in InterpTime]

plt.figure()
ax = plt.figure().add_subplot(111)
line1, = plt.plot(InterpTime,InterpRad,color='black',linestyle='dashed',linewidth = 2)
line2, = plt.plot(InterpTime,InterpAzi,color='black',linewidth = 2,label = '  This work')
line3, = plt.plot(InterpTime,KK_ForceRad,color='teal',linestyle='dashed',linewidth = 2)
line4, = plt.plot(InterpTime,KK_ForceAzi,color='teal',linewidth = 2,label = '  Kim proxy')
line5, = plt.plot(InterpTime,O99_ForceAzi,color='peru',linewidth = 2,label = '  Ostriker proxy')
line6, = plt.plot(InterpTime,O99_ForceRad,color='peru',linewidth = 2,linestyle='dashed')
#plt.gca().set_aspect('equal')
ratio = 1.0
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
plt.tight_layout()
plt.xlabel(r'Time $t$',fontsize =25)
#plt.xticks(['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'])
plt.ylabel(r'$F/[{4\pi{G^2}M^2\rho_0}/c^2]$', fontsize=25)
matplotlib.pyplot.xticks(fontsize=16)
matplotlib.pyplot.yticks(fontsize=16)
plt.title('Ecc = %g, Mp = %g'%(ecc0,Mp),fontsize=25)

if IncludeLegend == True:
    legend = plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=4)
    for text in legend.get_texts():
        text.set_ha('left')
        plt.subplots_adjust(bottom=0.2)

plt.savefig('ForceTime%gMp_%gecc_.png' %(Mp,ecc0),dpi=300, bbox_inches = "tight")

