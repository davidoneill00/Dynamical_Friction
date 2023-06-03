import numpy as np
import matplotlib
matplotlib.use('Agg')
import sys
import matplotlib.pyplot as plt
from matplotlib import pyplot
import math as ma
import EllipFncs as epf
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D
from scipy.interpolate import interp1d
import os
rc_fonts = {"text.usetex": True,}
plt.rcParams.update(rc_fonts)


# Initialise parameters and create the grid
ecc0 = [0.1,0.5,0.9]
Mp = [0.5,4] 

plt.figure()
fig, axes = plt.subplots(2, 3,figsize=(10,8))


legend_handles = []
legend_labels = [] 
for p in range(0,len(ecc0)):
    for q in range(0,len(Mp)):

        e = ecc0[p]
        m = Mp[q]

        grid = epf.Grid(m,e)
        T = np.append(grid.T,grid.T[0]+1)

# Now we will import the forces found from our force computational files

        Radi = np.zeros(len(T))
        Azi = np.zeros(len(T))
        for i in range(0,len(T)-1):
            NAME ='Polar%gt_%gecc_%gMp_%gA_%gNr_%gNtheta_.txt' %(T[i],e,m,grid.A,grid.Nr,grid.Ntheta)
            with open("/groups/astro/davidon/GDF_Python/SteadyStates/ecc%g/"%(e) + NAME,'r') as ap:
                Azi[i] = np.genfromtxt(ap,delimiter=',')[0]

        for i in range(0,len(T)-1):
            NAME ='Polar%gt_%gecc_%gMp_%gA_%gNr_%gNtheta_.txt' %(T[i],e,m,grid.A,grid.Nr,grid.Ntheta)
            with open("/groups/astro/davidon/GDF_Python/SteadyStates/ecc%g/"%(e) + NAME,'r') as ap:
                Radi[i] = float(np.genfromtxt(ap,delimiter=',')[1])

        Azi[len(grid.T)] = Azi[0]
        Radi[len(grid.T)] = Radi[0]

# Performing an interpolation over the data points

        FAZI = interp1d(T,Azi,kind='cubic') 
        FRAD = interp1d(T,Radi,kind='cubic')

        InterpTime = np.arange(T[0],T[len(T)-1]-0.001,0.001)
        InterpAzi = np.array([FAZI(t) for t in InterpTime])
        InterpRad = np.array([FRAD(t) for t in InterpTime])

        KK_ForceRad = [epf.PolarKK(t,e,m)[1] for t in InterpTime]
        KK_ForceAzi = [epf.PolarKK(t,e,m)[0] for t in InterpTime]
        O99_ForceRad= [epf.PolarO99(t,e,m)[1] for t in InterpTime]
        O99_ForceAzi= [epf.PolarO99(t,e,m)[0] for t in InterpTime]

        print('Loop Finished')

        TCK = 1.5

        line1, = axes[q, p].plot(InterpTime,InterpAzi,color='black',linewidth = TCK,label = r'This work $F_\theta$')
        line2, = axes[q, p].plot(InterpTime,O99_ForceAzi,color='peru',linewidth = TCK,label = r'Ostriker proxy $F_\theta$')
        line3, = axes[q, p].plot(InterpTime,KK_ForceAzi,color='royalblue',linewidth = TCK,label = r'Kim proxy $F_\theta$')
        line4, = axes[q, p].plot(InterpTime,InterpRad,color='black',linestyle='dashed',linewidth = TCK,label = r'This work $F_\r$')
        line5, = axes[q, p].plot(InterpTime,O99_ForceRad,color='peru',linewidth = TCK,linestyle='dashed',label = r'Ostriker proxy $F_r$')
        line6, = axes[q, p].plot(InterpTime,KK_ForceRad,color='royalblue',linestyle='dashed',linewidth = TCK,label = r'Kim proxy $F_r$')
        axes[q, p].set_title('Ecc = %g, Mp = %g'%(e,m),fontsize=15)

        if q == 0:
            axes[q,p].set_xticks([])
        else:
            axes[q,p].set_xticklabels(['0','0.2','0.4','0.6','0.8','1'])
            axes[q,p].set_xlabel(r'Orbital Time $t/P$')
            axes[q,p].set_ylim(-0.5,2)
        if p == 1:
            axes[q,p].set_yticks([])
        elif p==2:
            axes[q,p].set_yticks([])
        else:
            axes[q,p].set_ylabel(r'$F/[{4\pi{G^2}M^2\rho_0}/c^2]$', fontsize=15)


legend_handles = [line1, line2, line3, line4, line5, line6]
legend_labels = [r'This work $F_\theta$',r'Ostriker proxy $F_\theta$',r'Kim proxy $F_\theta$',r'This work $F_\r$',r'Ostriker proxy $F_r$',r'Kim proxy $F_r$']

#plt.legend(legend_handles,legend_labels)

fig.subplots_adjust(bottom=0.16)
fig.legend(legend_handles, legend_labels, loc='lower center', ncol=3)
plt.savefig('ForceTime.png', dpi=300)#, bbox_inches="tight")#,pad_inches=0.2)

