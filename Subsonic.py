import numpy as np
import sys
import EllipFncs as epf
import time
import os
import os.path
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import itertools
import math
from numpy import asarray
from numpy import savetxt


# Defining our eccentricity and Mach number at perihelion

ecc0 = float(sys.argv[1])
Mp = float(sys.argv[2])
c0 = ((2*np.pi)/Mp)*np.sqrt((1+ecc0)/(1-ecc0))

# Importing the previous grid and redefining all of the parameters 

grid = epf.Grid(Mp,ecc0)
overstart = time.time()
Plotting = False

if Plotting == True:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import pyplot


print('ecc',ecc0,'Mp',Mp,'Ntheta',grid.Ntheta,'A',grid.A,'Nr',grid.Nr)


# Alpha, now extended to an array domain                                                                            
def Alphap(params):
    r = params[0]
    theta=params[1]
    phi=params[2]
    time = params[3]
    x = np.cos(epf.EPhi(time,ecc0))+r*np.cos(theta)*np.sin(phi)
    y = np.sqrt(1-ecc0**2)*np.sin(epf.EPhi(time,ecc0))+ r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(phi)
    return epf.Alpha(time,x,y,z,ecc0,c0)

# Simply the x component of the vector dr/r^2                                                                 
def distfactorp(params):
    r = params[0]
    theta=params[1]
    phi=params[2]
    time = params[3]
    return np.sin(phi)**2*np.cos(theta)

# And the y component of the vector dr/r^2                                                                              
def distfactorq(params):
    r = params[0]
    theta=params[1]
    phi=params[2]
    time = params[3]
    return np.sin(phi)**2*np.sin(theta)


# We begin the multiprocessing pool
if __name__ == '__main__':
    p = Pool()

# We need to evaluate the distance factors first. If the file already exists, we will import it

    Gridx = 'Intgx%gA_%gNr_%gNtheta_.txt' %(grid.A,grid.Nr,grid.Ntheta)
    Gridy = 'Intgy%gA_%gNr_%gNtheta_.txt' %(grid.A,grid.Nr,grid.Ntheta)

    if os.path.isfile('/lustre/astro/davidon/Storage/AlphaFiles/Grid/'+ Gridx) == True:
        with open("/lustre/astro/davidon/Storage/AlphaFiles/Grid/" + Gridx,'r') as ap:
            INTGX = np.array(np.genfromtxt(ap,delimiter=',').reshape(grid.Nr,grid.Ntheta,grid.Nphi))
        ap.close()
        with open("/lustre/astro/davidon/Storage/AlphaFiles/Grid/" + Gridy,'r') as ap:
            INTGY = np.array(np.genfromtxt(ap,delimiter=',').reshape(grid.Nr,grid.Ntheta,grid.Nphi))
        ap.close()

# The file has not been found. Therefore we will use the pool to compute it now
    
    else:
        INTGX = []
        INTGY = []

        distfactorgrid = list(itertools.product(grid.r,grid.theta,grid.phi))

# The evaluations of the functions over the grid

        INTGX.append(p.map(distfactorp,distfactorgrid))                                                                              
        INTGY.append(p.map(distfactorq,distfactorgrid))                                                                            

        Intgx = np.array(INTGX).reshape(grid.Nr,grid.Ntheta,grid.Nphi)                                                                
        Intgy = np.array(INTGY).reshape(grid.Nr,grid.Ntheta,grid.Nphi) 
        
# Saving the files to a specific directory
        with open(os.path.join('/lustre/astro/davidon/Storage/AlphaFiles/Grid/', Gridx),"w") as file:
            np.savetxt(file, INTGX, fmt="%f")
        file.close()

        with open(os.path.join('/lustre/astro/davidon/Storage/AlphaFiles/Grid/', Gridy),"w") as file:
            np.savetxt(file, INTGY, fmt="%f")
        file.close()

# Option whether or not we want to plot. Usually going to be turned off
        if Plotting == True:
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))                                                              
            im =ax.contourf(grid.theta, grid.r, Intgx[:,:,grid.Nphi-1],200,vmin = -1,vmax = .5)                                       
            fig.colorbar(im)                                                                                                          
            plt.savefig('Intgx%gA_%gNr_%gNtheta_.png' %(grid.A,grid.Nr,grid.Ntheta))                                                                      

        if Plotting == True:                                                                                                          
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))                                                               
            im =ax.contourf(grid.theta, grid.r, Intgy[:,:,grid.Nphi-1],200,vmin = -1,vmax = .5)                                       
            fig.colorbar(im)                                                                                                          
            plt.savefig('Intgy%gA_%gNr_%gNtheta_.png' %(grid.A,grid.Nr,grid.Ntheta))                 

# Now the time independent grid has either been imported or computed. We can look at the time dependent 
# density perturbations Alpha

    for t in grid.T:
        ALPHAP = []

# We define the parameter lists which will be evaluated over. We can then pass paramlist to the pool
        paramlist = list(itertools.product(grid.r,grid.theta,grid.phi,[t]))

# The multiprocessing happens right here to evaluate the density perturbations over our grid
        ALPHAP.append(p.map(Alphap,paramlist))
        ALPHA = np.array(ALPHAP).reshape([grid.Nr,grid.Ntheta,grid.Nphi])

# Now we save our results. This path may be altered depending on preferences
        NAME = 'Alpha%gt_%gecc_%gMp_%gA_%gNr%gNtheta_.txt' %(t,ecc0,Mp,grid.A,grid.Nr,grid.Ntheta)
        with open(os.path.join('/lustre/astro/davidon/Storage/AlphaFiles/AlphaFiles/ecc%g/'%(ecc0),NAME), "w") as file:
            np.savetxt(file, ALPHAP, fmt="%f")
            file.close()

        if Plotting == True:
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
            im =ax.contourf(grid.theta, grid.r, np.log10(ALPHA[:,:,grid.Nphi-1]),200,vmin = -1,vmax = .5)
            fig.colorbar(im)
            plt.savefig('Alpha%gt_%gecc_%gMp_.png' %(t,ecc0,Mp))


# Now we multiply the distance factor times the density perturbations to give the integrand of the force 
        Intgi = np.multiply(Intgx,ALPHA)
        Intgj = np.multiply(Intgy,ALPHA)


# Now we have found alpha and the distance factors. This is everything needed to find the total force.
# However now we will evaluate the force on subgrids (smaller spheres) to ensure the force is convergent

        force = []
        SubGrid = np.arange(grid.subsize,grid.Nr+grid.subsize,grid.subsize)
        for sub in SubGrid:
            Reducedr     = grid.r[0:sub]
            ReducedIntgi = Intgi[0:sub,:,:]
            ReducedIntgj = Intgj[0:sub,:,:]

# Integrating the force on the reduced radial grid
            ReducedGDFi = 2*np.trapz(np.trapz(np.trapz(ReducedIntgi, Reducedr, axis=0), grid.theta, axis=0), grid.phi, axis=0)
            ReducedGDFj = 2*np.trapz(np.trapz(np.trapz(ReducedIntgj, Reducedr, axis=0), grid.theta, axis=0), grid.phi, axis=0)

# Some minor manipulation with the force to provide it in the correct radial/azimuthal basis
            ReducedFNorm = np.sqrt(ReducedGDFi**2+ReducedGDFj**2)/(4*np.pi)
            Reducedthisthing = np.arctan2(ReducedGDFj,ReducedGDFi)

            if Reducedthisthing<0:
                ReducedForceAngle = Reducedthisthing+2*np.pi
            else:
                ReducedForceAngle= Reducedthisthing

            Tan = np.arctan2(np.cos(epf.EPhi(t,ecc0))*np.sqrt(1-ecc0**2),-np.sin(epf.EPhi(t,ecc0)))
            if Tan<0:
                T = Tan+2*np.pi
            else:
                T= Tan


            ReducedEffAngle = ReducedForceAngle-T
            ReducedForce = [-ReducedFNorm*np.cos(ReducedEffAngle),ReducedFNorm*np.sin(ReducedEffAngle)]

# The main result: Row is the force evaluated. It lists [Azimuthal, Radial]
            ReducedRow = [epf.ForceRadAzi(epf.EPhi(t,ecc0),ecc0,ReducedForce[1],ReducedForce[0])[1],epf.ForceRadAzi(epf.EPhi(t,ecc0),ecc0,ReducedForce[1],ReducedForce[0])[0]]
            force.append(ReducedRow)

# Now we have the force from each subgrid
        force = np.array(force).reshape(2*len(SubGrid))
# We save it here. Now in principle this may be plotted to show the convergence.
        NAME1 = 'BoxConv%gt_%gecc_%gMp_.txt' %(t,ecc0,Mp)
        with open(os.path.join('/groups/astro/davidon/GDF_Python/SteadyStates/BoxRes/ecc%g/' %(ecc0),NAME1), "w") as file:
            np.savetxt(file, force, fmt="%f")
            file.close()

        print('Finished BoxConvergence on time%g' %(t))


# Just to be sure, we evaluate the force on the entire grid. This should be equal to the final component
# of the BoxConv file which we have already saved
        GDFi = 2*np.trapz(np.trapz(np.trapz(Intgi, grid.r, axis=0), grid.theta, axis=0), grid.phi, axis=0)
        GDFj = 2*np.trapz(np.trapz(np.trapz(Intgj, grid.r, axis=0), grid.theta, axis=0), grid.phi, axis=0) 

        FNorm = np.sqrt(GDFi**2+GDFj**2)/(4*np.pi)
        thisthing = np.arctan2(GDFj,GDFi)
    
        if thisthing<0:
            ForceAngle = thisthing+2*np.pi
        else:
            ForceAngle= thisthing

        Tan = np.arctan2(np.cos(epf.EPhi(t,ecc0))*np.sqrt(1-ecc0**2),-np.sin(epf.EPhi(t,ecc0)))
        if Tan<0:
            T = Tan+2*np.pi
        else:
            T= Tan

        EffAngle = ForceAngle-T
        Force = [-FNorm*np.cos(EffAngle),FNorm*np.sin(EffAngle)]   #THIS IS PERP PAR

        Row = [epf.ForceRadAzi(epf.EPhi(t,ecc0),ecc0,Force[1],Force[0])[1],epf.ForceRadAzi(epf.EPhi(t,ecc0),ecc0,Force[1],Force[0])[0]]

# We now save the entire force to a specific directory:
        NAME1 = 'Polar%gt_%gecc_%gMp_%gA_%gNr_%gNtheta_.txt' %(t,ecc0,Mp,grid.A,grid.Nr,grid.Ntheta)
        with open(os.path.join('/groups/astro/davidon/GDF_Python/SteadyStates/ecc%g/' %(ecc0),NAME1), "w") as file:
            np.savetxt(file, Row, fmt="%f")
            file.close()

        print('Force is',Row)
            
overend = time.time()
endtotal = overend-overstart
print('Total time is', endtotal)


