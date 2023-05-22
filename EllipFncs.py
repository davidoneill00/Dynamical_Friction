import numpy as np
from scipy.misc import derivative
import math
import time
import sys
import scipy.optimize as sco
from scipy.interpolate import interp1d

#Define a number of parameters and the grid:

class Grid():
    def __init__(self, Mp, ecc):
        if Mp <= 0.5:
            self.A = 27
            self.T = np.arange(4,5,0.1)
            self.Ntheta = 120
            self.Nr=3200
        elif Mp <= 1:
            self.A = 27
            self.T = np.arange(8,9,0.1)
            self.Ntheta= 120
            self.Nr=3200
        elif Mp < 2:
            self.A = 0   #Need to define these
            self.T = 0
            self.Ntheta= 160
            self.Nr=3500
        elif Mp >= 2:
            self.A = 20
            self.T = np.arange(int(Mp),int(Mp)+1,0.1)
            self.Ntheta= 160
            self.Nr=2000

        self.Nphi = self.Ntheta//4
        self.theta = np.linspace(0,2*np.pi,self.Ntheta)
        self.phi = np.linspace(0,np.pi/2,self.Nphi)
        self.subsize = 50
        self.r = np.zeros(self.Nr)
        for i in range(1,self.Nr):
            self.r[0] = 0.1
            self.r[i] = self.r[i-1]*(1+2*np.pi/(self.Ntheta*self.A))


#Solving Kepler's equation:     phi - e * sin(phi) = 2 * pi * t
def f(phi, t, ecc):
    return phi-ecc*np.sin(phi)-2*np.pi*t


#The eccentric anomaly
def EPhi(t,ecc):
    res = sco.newton(f, 2,args=(t,ecc,),tol=0.00001,maxiter=50)
    return res

#Distance function
def FullEllipDistXY(x,y,z,phip,ecc):
    x1 = np.cos(phip)
    y1 = np.sqrt(1-ecc**2)*np.sin(phip)
    dist = np.sqrt((x-x1)**2+(y-y1)**2+z**2)
    return dist

#Now we want to solve the retarded Green's function equation: phi (t - d/c) - phi' = 0
def RootFunction(phip,t,x,y,z,ecc,c):
    output = EPhi(t-FullEllipDistXY(x,y,z,phip,ecc)/c,ecc)-phip
    return output

#Root finding. We employ a numerical bisection method:
def rootsearch(f,a,b,dx,t,x,y,z,ecc,c):
    x1 = a; f1 = f(a,t,x,y,z,ecc,c)
    x2 = a + dx; f2 = f(x2,t,x,y,z,ecc,c)
    while f1*f2 > 0.0:
        if x1 >= b:
            return None,None
        x1 = x2; f1 = f2
        x2 = x1 + dx; f2 = f(x2,t,x,y,z,ecc,c)
    return x1,x2

def bisect(f,x1,x2,t,x,y,z,ecc,c,switch=0,epsilon=1e-05):
    f1 = f(x1,t,x,y,z,ecc,c)
    if f1 == 0.0:
        return x1
    f2 = f(x2,t,x,y,z,ecc,c)
    if f2 == 0.0:
        return x2
    if f1*f2 > 0.0:
        print('Root is not bracketed')
        return None
    n = int(math.ceil(math.log(abs(x2 - x1)/epsilon)/math.log(2.0)))
    for i in range(n):
        x3 = 0.5*(x1 + x2); f3 = f(x3,t,x,y,z,ecc,c)
        if (switch == 1) and (abs(f3) >abs(f1)) and (abs(f3) > abs(f2)):
            return None
        if f3 == 0.0:
            return x3
        if f2*f3 < 0.0:
            x1 = x3
            f1 = f3
        else:
            x2 =x3
            f2 = f3
    return (x1 + x2)/2.0

# A set of coordinates centered on the current position of the perturber
def CenteredCoords(t,x,y,z,ecc):
    return [x-np.cos(EPhi(t,ecc)),y-np.sqrt(1-ecc**2)*np.sin(EPhi(t,ecc))]

# The distance ot a given point from the current position of the perturber
def CenteredDist(t,x,y,z,ecc):
    return  np.sqrt(z**2+(CenteredCoords(t,x,y,z,ecc))[0]**2+(CenteredCoords(t,x,y,z,ecc))[1]**2)

# Define criterion for the seperation in roots. Below d = 0.1 we set ths high to speed up evalutation
# of roots. We do this because we puncture a sphere of radius 0.1 anyway, so it has no contribution to force.
def rootsep(t,x,y,z,ecc):
    d=CenteredDist(t,x,y,z,ecc)
    if d<=0.1:
        rtsep = 50.0
    else:
        rtsep = d/20.0
    return rtsep


def roots(f,a,b,t,x,y,z,ecc,c):
    arr = []
    while 1:
        x1,x2 = rootsearch(f,a,b,rootsep(t,x,y,z,ecc),t,x,y,z,ecc,c)
        if x1 != None:
            a = x2
            root = bisect(f,x1,x2,t,x,y,z,ecc,c,1)
            if root != None:
                pass
                arr.append(root)
                
        else:
            return arr

# Straightforward lagged time function. Outputs the time at which a sound wave was emitted from
# the perturber provided it is at (x,y,z) at time t 
def LaggedTime(t, x, y,z, phip, ecc,c):
    return t - FullEllipDistXY(x, y,z, phip, ecc)/c          

# Now we can find the value of the eccentric anomaly at the retarded time (rts). This is generally
# an array for supersonic trajcetories. We also output the number of roots  
def ERootValues(t,x,y,z,ecc,c):
    UPLim = EPhi(t,ecc)   
    DOWNLim = 0
    vary = EPhi(LaggedTime(t,x,y,z,np.arctan2(y,x)-np.pi,0,c),ecc)
    if vary>0:
        DOWNLim = vary
    
    rts = roots(RootFunction,DOWNLim,UPLim,t,x,y,z,ecc,c)
    return rts, len(rts)

# A wrapped derivative function
def Deriv(func, var=0, point=[]):
    args = point[:]
    def wraps(pp):
        args[var] = pp
        return func(*args)
    return derivative(wraps, point[var], dx=1e-05)

# We differentiate the distance function with respect to phip. Here, phip is the Eccentric anomaly value
# at the retarded time. 
def EMetricDeriv(x,y,z,phip,e):
    if (phip==np.float64):
        return Deriv(FullEllipDistXY,3,[x,y,z,phip,e])
    else:
        res = [Deriv(FullEllipDistXY,3,[x,y,z,phip[i],e]) for i in range(len(phip))]
        return res

# The eccentric anomaly evaluated at the retarded time
def PhiValTL(t,x,y,z,phip,ecc,c):
    return EPhi(LaggedTime(t,x,y,z,phip,ecc,c),ecc)

# The derivative of the eccentric anomaly evaluated at the retarded time with respect to phip
def FullDerPhi(t,x,y,z,phip,ecc,c):
    if (phip==np.float64):
        return 1-Deriv(PhiValTL,4,[t,x,y,z,phip,ecc,c])
    else:
        res = [1-Deriv(PhiValTL,4,[t,x,y,z,phip[i],ecc,c]) for i in range(len(phip))]
    return res

# The main function. This gives the dimensionless density perturbation at a given t,x,y,z. The Mach number
# is implicit in this equation. We can define the sound speed as c = 2 *( pi / Mp )*sqrt((1+e)/(1-e))

def Alpha(t,x,y,z,ecc,c):
    Val=ERootValues(t,x,y,z,ecc,c)[0]
    A=np.sum((1/((FullEllipDistXY(x,y,z, Val, ecc)*np.abs(FullDerPhi(t, x, y,z, Val, ecc,c))))))
    if CenteredDist(t,x,y,z,ecc)<0.1:
        A=0
    return A 


def TrueAnomaly(phi,ecc):
    return phi+2*np.arctan2(ecc*np.sin(phi),(1+np.sqrt(1-ecc**2)-ecc*np.cos(phi)))

# Gives the angle of the tangent plane for some eccentric anomaly angle and eccentricity                                              
def Tangent(angle,ecc):
    return np.arctan2(np.cos(angle)*np.sqrt(1-ecc**2),-np.sin(angle))

# Given a perpendicular/parallel force we want to decompose it into radial and azimuthal components                                   
def ForceRadAzi(angle,ecc,fperp,fpar):
    delta = Tangent(angle,ecc) - TrueAnomaly(angle,ecc)- np.arctan2(fperp,fpar)
    f = np.sqrt(fperp**2+fpar**2)
    return [f*np.cos(delta),f*np.sin(delta)]




    
