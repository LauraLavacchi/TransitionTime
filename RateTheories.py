import numpy as np
from scipy import integrate
import cmath as ma


###############
# Grote-Hynes #
###############
def GetLambdaNeqGHExp(wNeq,gamma,tau_exp):
    coeff = np.array([tau_exp,1,gamma-wNeq**2*tau_exp,-wNeq**2])
    roots = np.roots(coeff)
    return np.max(roots)

def GH(gamma,
          VNeq,
          wNeq,
          w0,
          tau_exp):
    lNeq = GetLambdaNeqGHExp(wNeq,gamma,tau_exp)
    return w0/(2*np.pi)*lNeq/wNeq*np.exp(-VNeq)

##################
# Carmeli-Nitzan #
##################
def CN(gamma,
          VNeq,
          w0,
          tau_exp):
    PreFac = gamma*VNeq/(2*(1+w0**2*tau_exp**2))
    return PreFac*np.exp(-VNeq)


#######################
# GH + Carmeli-Nitzan #
#######################
def GHCN(gamma,
          VNeq,
          wNeq,
          w0,
          tau_exp):
    GH_ = GH(gamma,VNeq,wNeq,w0,tau_exp)
    CN_ = CN(gamma,VNeq,w0,tau_exp)
    return 1./(1./GH_ + 1./CN_)


###########
# Kramers #
###########
def Kramers(gamma,w0,wNeq,vNeq):
    # note that the gamma from Best is the same as from Haenggi,
    # i.e. it is the friction with the mass already factored out
    return (np.sqrt( (gamma/2)**2 + wNeq**2 ) - gamma/2 )*w0/(2*np.pi*wNeq)*np.exp(-vNeq)
# Inverse function:
def kGamma(rate,w0,wNeq,vNeq):
    TermA = w0*wNeq/(2*np.pi*rate)*np.exp(-vNeq)
    TermB = 2*np.pi*wNeq*rate/w0*np.exp(vNeq)
    return TermA-TermB

#####################
# Mel'nikov-Meshkov #
#####################
# (see also
#  http://localhost:8888/notebooks/yoshi-scratch/MSM/new/analysis/Alkane/MD/N4/MelnikovMeshkov.ipynb
# )
def A_func(tm,td,S1_integral):
    Arg = 2*np.sqrt(2)*np.sqrt(td/tm)*S1_integral
    func = lambda x: np.log(1 - np.exp( -Arg * (x**2+1./4.) )  )/(x**2 + 1./4.)
    return np.exp( integrate.quad(func, -np.inf, np.inf)[0]/(2*np.pi))

# This is for calculating the number S1 for a quartic potential
def S1_integral(VNeq):
    func = lambda x: np.sqrt(  -VNeq*( (x**2-1)**2 - 1 )  )
    return integrate.quad(func,-np.sqrt(2.),0.)[0]

def MM(tm, # ps
       td, # ps
	VNeq, # in units of kT
	wNeq, # 1/ps
	w0, # 1/ps
	S1_integral):
    A = A_func(tm,td,S1_integral)
    Fac1 = w0/(2*np.pi)
    Fac2 = np.sqrt(1 + 1./(4*tm**2*wNeq**2)) - 1./(2*tm*wNeq)
    Fac3 = A*np.exp(-VNeq)
    return Fac1*Fac2*Fac3


##########################
# Pollak-Grabert-Haenggi #
##########################
# based on eq. (4.6)
def GetLambdaNeqPGH(wNeq,Gamma,Alpha):
    # see http://docs.scipy.org/doc/numpy/reference/generated/numpy.roots.html
    # coefficient arrays goes from highest to lowest
    p = np.array([Alpha*Gamma,
                 1,
                 Gamma*(1-Alpha*wNeq**2),
                 -wNeq**2])
    roots = np.roots(p)
    for i,e in enumerate(roots):
        if e.real > 0:
            if abs(e) != abs(e.real):
                raise RuntimeError('GetLambdaNeq: Root complex!')
            #print(e)
            return e.real


# based on eqs. (4.12)
def GetEpsilon(Gamma,Alpha,lNeq):
    return Gamma/(2*lNeq*( 1 + Alpha*Gamma*lNeq )**2 )

# based on eq. (4.20)
#def GetTp(lNeq,l0):
#    y = (lNeq**2-l0**2)/(l0**2+lNeq**2)
#    # Note that np.arccos yields a value in (0,pi), but we
#    # want a value in (pi,2*pi)
#    # http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.arccos.html
#    return (2*np.pi-np.arccos(y))/l0
#def GetTp(lNeq,l0):
#    y = -2*l0*lNeq/(l0**2+lNeq**2)
#    # Note that np.arccos yields a value in (0,pi), but we
#    # want a value in (pi,2*pi)
#    # http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.arccos.html
#    bla = np.arcsin(y)+2*np.pi
#    return bla/l0
def GetTp(lNeq,l0):
    x = (lNeq**2-l0**2)/(l0**2+lNeq**2)
    y = -2*l0*lNeq/(l0**2+lNeq**2)
    # Note that np.arctan2 yields a value in (-pi,pi), but we
    # want a value in (pi,2*pi)
    bla = (np.arctan2(y,x))%(2*np.pi)
    return bla/l0

# based on eqs. (4.22), (4.23)
def ZetaSig(Alpha,Gamma,lNeq,wNeq):
    Zeta = 0.5*(lNeq + 1./(Alpha*Gamma))
    Sig = np.sqrt( Zeta**2 - wNeq**2/(Alpha*Gamma*lNeq) + 0.j )
    return Zeta, Sig

# based on eq. (4.27)
def R(z,l0,lNeq,tp):
    #if z == np.inf:
    #    return 0
    Fac=l0**2+z**2
    # EDIT: the last z**2 in the following line was added because of the comment in the KSR ref.
    Term1 = l0**4*(lNeq-z)**2*(np.exp(-tp*z)-1)/(lNeq**2*Fac**2 * z**2)
    Term2 = tp*( 1/z + z*(l0**2+lNeq**2) / (2*lNeq**2*Fac) )
    Term3 = 2*l0**2*(l0**2+z*lNeq)/(lNeq**2*Fac**2)
    Term4 = 3*z/(lNeq*Fac)
    return Term1+Term2+Term3+Term4

# based on eq. (4.25)
def GetDeltaE(vNeq,
              w0,wNeq,
              l0,lNeq,
              Alpha,Gamma,
              e):#,tp): # e = \epsilon
    Zeta, Sig = ZetaSig(Alpha,Gamma,lNeq,wNeq)
    tp = GetTp(lNeq,l0)
    assert tp*l0/(np.pi) < 2 and tp*l0/(np.pi) >= 1, "The angle for tp calc. is not in desired interval!"
    #print('tp*l0=',tp*l0)
    #print('Zeta =',Zeta,'Sigma =', Sig)
    #
    Factor1 = vNeq*w0**2/wNeq**2*(lNeq**4*(l0**2+lNeq**2))/(l0**4)
    Term1 = 0.5*(1+2*e)*( R(Zeta+Sig,l0,lNeq,tp) + R(Zeta-Sig,l0,lNeq,tp) )
    Term2 = -0.5*( (1+2*e)*Zeta - lNeq )/Sig *(R(Zeta+Sig,l0,lNeq,tp) - R(Zeta-Sig,l0,lNeq,tp))
    Term3 = -R(lNeq,l0,lNeq,tp)
    #if ((Factor1*(Term1+Term2+Term3)) < 0):
        #print(Zeta,Sig)
        #print(Term1,Term2,Term3)
        #print(R(Zeta+Sig,l0,lNeq,tp),R(Zeta-Sig,l0,lNeq,tp),R(lNeq,l0,lNeq,tp))
        #print(Term1,Term2,Term3)
        #print(tp*l0)#(np.exp(-tp*lNeq)-1))
    return (Factor1*(Term1+Term2+Term3)).real

# based on eq. (3.33)
def getFT(lNeq,wNeq,delta):
    Function = lambda y: np.log(1-np.exp(-delta*(1+y**2)/4))/(1+y**2)
    Integral = integrate.quad(Function, -np.inf, np.inf)[0]
    return lNeq/wNeq*np.exp(Integral/np.pi)

# based on (3.32)
def PGH(Gamma,
                  Alpha,
                  vNeq, # in units of k_B T
                  wNeq,
                  w0,
       double_well=False):
    lNeq = GetLambdaNeqPGH(wNeq,Gamma,Alpha)
    Epsilon = GetEpsilon(Gamma,Alpha,lNeq)
    # for l0, see eqs. (2.20), (4.15)
    l0 = np.sqrt((w0**2+wNeq**2)/(1+Epsilon)-lNeq**2)
    delta = GetDeltaE(vNeq,w0,wNeq,l0,lNeq,Alpha,Gamma,Epsilon)
    #print('eps=',Epsilon,'Lambda0 =',l0,'deltaE=',delta)
    #print('lNeq=',lNeq)
    #if delta.real < 0:
        #print(delta)
        #print(Gamma/wNeq,delta)
        #print(delta)
        #delta=abs(delta)
        #print('gamma=',Gamma,'alpha=',Alpha)
        #print('delta=',delta,'lNeq=',lNeq)
    if double_well:
        fTA = getFT(lNeq,wNeq,delta)
        fTB = getFT(lNeq,wNeq,2*delta)
        fT = fTA**2/fTB
    else:
        fT = getFT(lNeq,wNeq,delta)
    return w0/(2*np.pi)*fT*np.exp(-vNeq), fT/(lNeq/wNeq), delta



###########################
# Krishnan-Singh-Robinson #
###########################

# based on eq. (36)
def GetTp_KSR(l0P,eS,eC):
    x = (eC**2-eS**2)/(eC**2+eS**2)
    y = -2*eS*eC/(eS**2+eC**2)
    # Note that np.arctan2 yields a value in (-pi,pi), but we
    # want a value in (0,2*pi)
    bla = (np.arctan2(y,x))%(2*np.pi)
    return bla/l0P

# based on eqs. (38), (39)
def ThetaSig(Alpha0,Gamma0,lNeqP,wNeq):
    Theta = 0.5*(lNeqP + 1./(Alpha0*Gamma0))
    Sig = np.sqrt( Theta**2 - wNeq**2/(Alpha0*Gamma0*lNeqP) + 0.j )
    return Theta, Sig

# based on eqs. (34), (35)
def etaSetaC(lp,lNeqP,l0P,Epsilon,EpsilonP):
    u00 = 1./np.sqrt(Epsilon+1)
    u00P = 1./np.sqrt(EpsilonP+1)
    eS = l0P * u00P * lp / (u00 * lNeqP**2)
    eC = 1/lNeqP**2 * ((l0P**2 + lNeqP**2) - l0P**2 * u00P/u00)
    return eS, eC

# based on eq. (41)
def R_KSR(z,l0P,eC,eS,tp):
    #if z == np.inf:
    #    return 0

    Term1 = tp/z
    Term2 = - eC* np.sin(l0P * tp)/(z*l0P)
    Term3 = eS/(l0P*z)
    Term4 = - eS * np.cos(l0P*tp)/(l0P*z)
    Fac6 = (z*eS-l0P*eC)/(z**2+l0P**2)
    Term6 = Fac6 * (1/l0P - np.cos(l0P*tp)/l0P + eC *np.cos(2*l0P *tp)/(4*l0P)-eC/(4*l0P)+eS*tp/2 -eS * np.sin(2*l0P*tp)/(4*l0P))
    Fac7 = (z*eC+l0P*eS)/(z**2+l0P**2)
    Term7 = Fac7 * (-np.sin(l0P*tp)/l0P + eC*tp/2 + eC*np.sin(2*l0P*tp)/(4*l0P)+eS*np.cos(2*l0P*tp)/(4*l0P)-eS/(4*l0P))
    Fac8 = (z*eC+l0P*eS)/(z**2+l0P**2)-1/z
    Term8 = Fac8 * (1/z - np.exp(-z*tp)/z + (eS*l0P-eC*z)/(z**2+l0P**2)+ (-eS*l0P + z*eC)/(z**2+l0P**2)*np.exp(-z*tp)*np.cos(l0P*tp)-(l0P*eC+z*eS)/(z**2+l0P**2)*np.exp(-z*tp)*np.sin(l0P*tp))
    return Term1+Term2+Term3+Term4+Term6+Term7+Term8

# based on eq. (37)
def GetDeltaE_KSR(vNeq,
              w0,wNeq,
              l0P,lNeqP,lp,
              Alpha0,Gamma0,
              Epsilon, EpsilonP):
    Theta, Sig = ThetaSig(Alpha0,Gamma0,lNeqP,wNeq)
    #Theta, Sig = ThetaSig(Alpha0,Gamma0,lp,wNeq)
    eS,eC = etaSetaC(lp,lNeqP,l0P,Epsilon,EpsilonP)
    tp = GetTp_KSR(l0P,eS,eC)
    assert tp*l0P/(np.pi) < 2 and tp*l0P/(np.pi) >= 0, "The angle for tp calc. is not in desired interval!"
    Factor1 = vNeq*w0**2/wNeq**2*(lNeqP**4*(l0P**2+lNeqP**2))/(l0P**4)
    Term1 = 0.5*(1+2*EpsilonP)*( R_KSR(Theta+Sig,l0P,eC,eS,tp) + R_KSR(Theta-Sig,l0P,eC,eS,tp) )
    Term2 = -0.5*( (1+2*EpsilonP)*Theta - lNeqP )/Sig *(R_KSR(Theta+Sig,l0P,eC,eS,tp) - R_KSR(Theta-Sig,l0P,eC,eS,tp))
    Term3 = -R_KSR(lNeqP,l0P,eC,eS,tp)
    return (Factor1*(Term1+Term2+Term3)).real


# based on (3.32) in PGH ref.
def KSR(Gamma0,
        Alpha0,
        GammaNeq,
        AlphaNeq,
        vNeq, # in units of k_B T
        wNeq,
        w0,
       double_well=False, printEps=False):
    lNeq = GetLambdaNeqPGH(wNeq,GammaNeq,AlphaNeq)
    lNeqP = GetLambdaNeqPGH(wNeq,Gamma0,Alpha0)
    # These are not defined in the paper!
    Epsilon = GetEpsilon(GammaNeq,AlphaNeq,lNeq)
    EpsilonP = GetEpsilon(Gamma0,Alpha0,lNeqP)
    if printEps:
        print("Epsilon: ", Epsilon, "EpsilonP:", EpsilonP)
    # for l0P, see eq. (33)
    # Further note that Epsilon = -1 + 1/u00P**2 as given by eq. (2.20) in PGH ref.
    l0P =  np.sqrt((w0**2+wNeq**2)/(1+EpsilonP)-lNeqP**2)
    delta = GetDeltaE_KSR(vNeq,w0,wNeq,l0P,lNeqP, lNeq, Alpha0,Gamma0, Epsilon, EpsilonP)
    # taken from PGH
    if double_well:
        fTA = getFT(lNeq,wNeq,delta)
        fTB = getFT(lNeq,wNeq,2*delta)
        fT = fTA**2/fTB
    else:
        fT = getFT(lNeq,wNeq,delta)
    return w0/(2*np.pi)*fT*np.exp(-vNeq), fT/(lNeq/wNeq), delta, fT,

###############################
# Kappler (multi-exponential) #
###############################

def doubleExpKapplerMFPT(td,tm_over_td,gammas,tMs_over_td,U0,thres=10e7):
    tOD=0.0
    tED=0.0
    gamma0=np.sum(gammas)
    for gamma,tM_over_td in zip(gammas,tMs_over_td):
        tOD+=gamma/gamma0*np.exp(U0)/U0*(np.pi/(2*np.sqrt(2))*1/(1+10*U0*tM_over_td)+np.sqrt(U0*tm_over_td))
        tED+=1./(gamma0/gamma*np.exp(U0)/U0*(tm_over_td+4*U0*tM_over_td**2+np.sqrt(U0*tm_over_td)))
    tMFP = (tOD+1./tED)*td
    if tMFP > thres:
        return np.inf
    else:
        return tMFP
