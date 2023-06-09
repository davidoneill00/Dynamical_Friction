import numpy as np
from scipy.misc import derivative
import math 
import scipy.optimize as sco

#Define a number of parameters and the grid:

class Grid():
    def __init__(self, Mp, ecc):
        if Mp <= 0.5:
            if ecc == 0:
                self.T = np.array([6])
            elif ecc<=0.4:
                self.T = np.arange(6,7,0.05)
            elif ecc<=0.7:
                self.T = np.array([6,6.025,6.05,6.075,6.1,6.15,6.2,6.3,6.35,6.4,6.5,6.6,6.65,6.7,6.8,6.85,6.9,6.925,6.95,6.975])
            elif ecc<=0.9:
                self.T = np.array([6,6.005,6.01,6.015,6.02,6.025,6.03,6.06,6.09,6.15,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.85,6.91,6.94,6.97,6.975,6.98,6.985,6.99,6.995])

            self.A = 5.3
            self.Ntheta= 180
            self.Nr=1000
            


        elif Mp <= 1:
            if ecc == 0:
                self.T = np.array([11])
            elif ecc<=0.4:
                self.T = np.arange(11,12,0.05)
            elif ecc<=0.5:
                self.T = np.array([11,11.025,11.05,11.075,11.1,11.15,11.2,11.3,11.35,11.4,11.5,11.6,11.65,11.7,11.8,11.85,11.9,11.925,11.95,11.975])   
            elif ecc<=0.7:
                self.T = np.array([11,11.005,11.01,11.015,11.02,11.025,11.03,11.06,11.09,11.15,11.2,11.3,11.4,11.5,11.6,11.7,11.8,11.85,11.91,11.94,11.97,11.975,11.98,11.985,11.99,11.995])
            elif ecc<=0.9:
                self.T = np.array([11,11.001,11.002,11.003,11.004,11.01,11.02,11.04,11.06,11.08,11.1,11.2,11.3,11.4,11.5,11.6,11.7,11.8,11.9,11.92,11.94,11.96,11.98,11.99,11.996,11.997,11.998,11.999])

            self.A = 5.3
            self.Ntheta= 180
            self.Nr=1000

        elif Mp <= 2:
            t0 = int(Mp)+20
            if ecc == 0:
                self.T = np.array([t0])
            elif ecc<=0.4:
                self.T = np.arange(t0,t0+1,0.05)
            elif ecc<=0.5:
                self.T = np.array([t0,t0+0.025,t0+0.05,t0+0.075,t0+0.1,t0+0.15,t0+0.2,t0+0.3,t0+0.35,t0+0.4,t0+0.5,t0+0.6,t0+0.65,t0+0.7,t0+0.8,t0+0.85,t0+0.9,t0+0.925,t0+0.95,t0+0.975])
            elif ecc<=0.7:
                self.T = np.array([t0,t0+0.005,t0+0.01,t0+0.015,t0+0.02,t0+0.025,t0+0.03,t0+0.06,t0+0.09,t0+0.15,t0+0.2,t0+0.3,t0+0.4,t0+0.5,t0+0.6,t0+0.7,t0+0.8,t0+0.85,t0+0.91,t0+0.94,t0+0.97,t0+0.975,t0+0.98,t0+0.985,t0+0.99,t0+0.995])
            elif ecc<=0.9:
                self.T = np.array([t0,t0+0.001,t0+0.002,t0+0.003,t0+0.004,t0+0.01,t0+0.02,t0+0.04,t0+0.06,t0+0.08,t0+0.1,t0+0.2,t0+0.3,t0+0.4,t0+0.5,t0+0.6,t0+0.7,t0+0.8,t0+0.9,t0+0.92,t0+0.94,t0+0.96,t0+0.98,t0+0.99,t0+0.996,t0+0.997,t0+0.998,t0+0.999])

            self.Ntheta= 180
            self.A = 5.3
            self.Nr=1000

        elif Mp <= 6 :
            t0 = int(Mp)+1
            if ecc == 0:
                self.T = np.array([t0])
                self.A = 8
                self.Nr=1000
            elif ecc<=0.4:
                self.T = np.arange(t0,t0+1,0.05)
                self.A = 8
                self.Nr=1000
            elif ecc<=0.5:
                self.T = np.array([t0,t0+0.025,t0+0.05,t0+0.075,t0+0.1,t0+0.15,t0+0.2,t0+0.3,t0+0.35,t0+0.4,t0+0.5,t0+0.6,t0+0.65,t0+0.7,t0+0.8,t0+0.85,t0+0.9,t0+0.925,t0+0.95,t0+0.975])
                self.A = 8
                self.Nr=1000
            elif ecc<=0.7:
                self.T = np.array([t0,t0+0.005,t0+0.01,t0+0.015,t0+0.02,t0+0.025,t0+0.03,t0+0.06,t0+0.09,t0+0.15,t0+0.2,t0+0.3,t0+0.4,t0+0.5,t0+0.6,t0+0.7,t0+0.8,t0+0.85,t0+0.91,t0+0.94,t0+0.97,t0+0.975,t0+0.98,t0+0.985,t0+0.99,t0+0.995])   
                self.A = 8
                self.Nr=1000
            elif ecc<=0.9:
                self.T = np.array([t0,t0+0.001,t0+0.002,t0+0.003,t0+0.004,t0+0.01,t0+0.02,t0+0.04,t0+0.06,t0+0.08,t0+0.1,t0+0.2,t0+0.3,t0+0.4,t0+0.5,t0+0.6,t0+0.7,t0+0.8,t0+0.9,t0+0.92,t0+0.94,t0+0.96,t0+0.98,t0+0.99,t0+0.996,t0+0.997,t0+0.998,t0+0.999])
                self.A = 9
                self.Nr= 1200
            self.Ntheta= 180

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

# We include the fitting function defined by Kim and Kim 2007. However this is in tangential, perpendicular components

def DF(M):
    if M<4.4:
        if M<1.1:
            kkra = 10**(3.51*M-4.22)
            if M<1:
                if M<0.1:
                    kkaz = (0.5*math.log((1+M)/(1-M))-M)/(M**2)
                else:
                    kkaz = (0.7706*math.log((1+M)/(1.0004-0.9185*M))-1.4703*M)/(M**2)
            else:
                kkaz = (math.log(3300*(M-0.71)**5.72*M**(-9.58)))/M**2
        else:
            kkra = (0.5*math.log(9.33*M**2*(M**2-0.95)))/M**2
            kkaz = (math.log(3300*(M-0.71)**5.72*M**(-9.58)))/M**2
    else:
        kkra = 0.3
        kkaz = math.log(10/(0.11*M+1.65))/M**2
    return [kkaz,kkra]

# We also include the force from the Ostiker model

def O99(M):
    FITTINGFACTOR = math.log(20)  #Vt = 2Rp gave best agreement to Kim and Kim so I'll use that instead of their fit                     
    x1 = 0.92
    x2 = 1.02
    if M<=x1:
        return [(0.5*math.log((1+M)/(1-M))-M)/(M**2),0]
    elif M>=x2:
        return [(0.5*math.log(1-1/(M**2))+FITTINGFACTOR)/(M**2),0]
    else:
        y1 = O99(x1)[0]
        y2 = O99(x2)[0]
        return [y1+(M-x1)*(y2-y1)/(x2-x1),0]

# Now we want to be able to transform the above to a mock elliptical orbit. We paramaterise the velocity from an elliptical orbit    

def O99Elliptical(t,ecc,Mp):
    vel = 2*np.pi*np.sqrt((1+ecc*np.cos(EPhi(t,ecc)))/(1-ecc*np.cos(EPhi(t,ecc))))
    c = (2*np.pi/Mp)*np.sqrt((1+ecc)/(1-ecc))
    return [O99(vel/c)[0],O99(vel/c)[1]]


def KKElliptical(t,ecc,Mp):
    vel = 2*np.pi*np.sqrt((1+ecc*np.cos(EPhi(t,ecc)))/(1-ecc*np.cos(EPhi(t,ecc))))
    c = (2*np.pi/Mp)*np.sqrt((1+ecc)/(1-ecc))
    return [DF(vel/c)[0],DF(vel/c)[1]]

# And now we decompose the force into the radial/azimuthal basis. This gives us a good comparison for the computed forces      

def PolarO99(t,ecc,Mp):
    fperp =O99Elliptical(t,ecc,Mp)[1]
    fpar = O99Elliptical(t,ecc,Mp)[0]
    Tangent = np.arctan2(np.cos(EPhi(t,ecc))*np.sqrt(1-ecc**2),-np.sin(EPhi(t,ecc)))
    TrueAnomaly = EPhi(t,ecc)+2*np.arctan2(ecc*np.sin(EPhi(t,ecc)),(1+np.sqrt(1-ecc**2)-ecc*np.cos(EPhi(t,ecc))))
    delta = Tangent-TrueAnomaly-np.arctan2(fperp,fpar)
    f = np.sqrt(fperp**2+fpar**2)
    return [f*np.sin(delta),f*np.cos(delta)]    


def PolarKK(t,ecc,Mp):
    if Mp == 0:
        return [0,0]
    else:
        fperp =KKElliptical(t,ecc,Mp)[1]
        fpar = KKElliptical(t,ecc,Mp)[0]
        Tangent = np.arctan2(np.cos(EPhi(t,ecc))*np.sqrt(1-ecc**2),-np.sin(EPhi(t,ecc)))
        TrueAnomaly = EPhi(t,ecc)+2*np.arctan2(ecc*np.sin(EPhi(t,ecc)),(1+np.sqrt(1-ecc**2)-ecc*np.cos(EPhi(t,ecc))))
        delta = Tangent-TrueAnomaly-np.arctan2(fperp,fpar)
        f = np.sqrt(fperp**2+fpar**2)
        return [f*np.sin(delta),f*np.cos(delta)]   #Azimuthal,Radial    
    
