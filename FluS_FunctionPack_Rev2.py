# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:34:47 2019

@authors: Romulo Rodrigues de Carvalho
          

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

""" 
Define all the functions that will be needed for the calculation of the petroelastic model.
These functions are related to oil properties, gas properties, brine properties, and properties of the aggregate mineral constituents of the rock matrix.
Note to software developers: the code was written in a stucutured logic. All functions needed for it to run are made available in this file alone.
    
"""
msTofts = 3.28084 # m/s to ft/s
ftsToms = 1/3.28084 # m/s to ft/s
PsiToGPa = 1/145038 # psi to GPa
GPatoPsi = 145038 # GPa to psi
gcm3Tolbft3 = 62.4279606 # gcm3 to lb/ft3
lbft3Togcm3 = 1/62.4279606 # g/cm3  to lb/ft3 
SCFSTBtoLL = 1/5.61459275059329 # SCF/BBL to L/L
PsiToMPa = 6.894757*0.001
MPaToPsi = 1/PsiToMPa
LLtoSCFSTB = 5.61459275059329
MWair = 28.97 # lbm/lbmol

"""

    SECTION 1.1: EQUATIONS FOR GASES FROM BATZLE AND WANG (1992)

"""

def GasCriticalProperties_BW(SGG):
    
    """
    Calculates critical/pseucritical pressure and temperature in natural gas.
    
    Description
    ----------  
              
    Parameters
    ----------  
    Tpc: array_like (float, ndim=1)
       Vector containing the value of pseudocritical temperature in degree Rankine.
       Vector containing the value of pseudocritical pressure in psia.
              
    Returns
    -------
    gamma: array_like (float, ndim=1)
           The index of pVec that contains the first p greather than the critical p.  
    """
    
    Tpc = 94.72 + 170.75*SGG # output temperature is in Kelvin
    Ppc = 4.892 - 0.4048*SGG # output in MPa
    
    Tpc_gas = Tpc*1.8     # changes from Kelvin to Rankine
    Ppc_gas = 145.038*Ppc # changes from MPa to psia
       
    return Tpc_gas, Ppc_gas

def ZfactorGas_BW(Tpr,Ppr):
    
    E = 0.109*((3.85 - Tpr)**2)*np.exp(-(0.45 + 8*((0.56 - 1/Tpr)**2) )*((Ppr**(1.2))/Tpr) ) 
    Z_gas = (0.03 + 0.00527*((3.5 - Tpr)**3) )*Ppr + (0.642*Tpr - 0.007*(Tpr**4) - 0.52) + E
       
    return Z_gas

def CgtGas_BW(Ppr,Tpr,Z,P):
    
    F = -1.2*((Ppr**0.2)/Tpr)*( 0.45 + 8*((0.56 - 1/Tpr)**2) )*np.exp(- (0.45 + 8*(0.56 - 1/Tpr)**2)*((Ppr**1.2)/Tpr)) # fraction, unitless
    
    DZDPpr = 0.03 + 0.00527*(3.5 - Tpr)**3 + 0.109*((3.85 - Tpr)**2)*F # fraction, unitless
    
    Cgt_gas = (1/P)*(1 - (Ppr/Z)*DZDPpr) # psia^-1
    
    return Cgt_gas


def YGas_BW(Ppr):
    
    Y_gas = 0.85 + 5.6/(Ppr + 2) + 27.1/((Ppr + 3.5)**2) - 8.7*np.exp(-0.65*(Ppr + 1)) # fraction, unitless
    
    return Y_gas


def KGas_BW(Cgt,Yo):
    
    Cgs = Cgt/Yo
    
    k_gas = Cgs**-1   
       
    return k_gas

def RhoGas_BW(P, T, MWgas,Z):
    
    R = 10.7316 # [ft3*psi]/[lbmol*DegR]
    rho_gas = (P*MWgas)/(Z*R*T) # lbm/ft3   
       
    return rho_gas


def VpGas_BW(K_gas,rho_gas):
    
    multiplier = np.sqrt( 32.174049/((1/12)**2) )

    Vp_gas = multiplier*(K_gas/rho_gas)**(1/2)
       
    return Vp_gas # ft/s

def GasElasticProperties_BW(SGG,P,TR):
    
    MWair = 28.97 # lbm/lbmol
    
    TpcGas, PpcGas = GasCriticalProperties_BW(SGG) # returns Rankine and psi
    
    TprGas = TR/TpcGas 
    
    PprGas = P/PpcGas    
    
    ZfactorGas = ZfactorGas_BW(TprGas,PprGas)    
    
    CgtGas = CgtGas_BW(PprGas,TprGas, ZfactorGas,P) # returns psia**-1   
    
    YGas = YGas_BW(PprGas) # returns unitless   
    
    KGas = KGas_BW(CgtGas,YGas) # returns psia   
    
    MWgas = SGG*MWair    
    
    RhoGas = RhoGas_BW(P, TR, MWgas, ZfactorGas)   # returns lbm/ft3
    
    VpGas = VpGas_BW(KGas, RhoGas) # returns ft/s
    
    return VpGas, RhoGas, KGas 


"""

    SECTION 1.2: EQUATIONS FOR OILS FROM BATZLE AND WANG (1992)

"""

def Bo_BW(Rs,SGG,API,T):
            
    """
    
    The BW equations are in metric units. Here, we will need to convert inputs, which are in Field Units, to Metric Units 
    
    T: Rankine
    Rs: SCF/STB
    SGG: unitless
    API: API degree
    
    """ 
    SCFSTBtoLL = 1/5.61459275059329 # SCF/BBL to L/L
        
    RsM = Rs*SCFSTBtoLL        
    TC = (T - 491.67)*(5/9)   
        
    SGO = 141.5/(API + 131.5)        
    Main = 2.4*RsM*np.sqrt(SGG/SGO) + TC + 17.8
        
    Bo = 0.972 + 0.00038*(Main)**1.175 # Bo, nevertheless, is unitless in this equation, since it is a M3/M3 ratio.
        
    return Bo # M3/M3



def RhoDeadOil_BW(P,T,API):
        
    """
    
    The BW equations are in metric units. Here, we will need to convert inputs, which are in Field Units, to Metric Units 
    
    P: psi
    T: Rankine
    Rs: SCF/STB
    SGG: unitless
    API: API degree
    
    """ 
        
    PMPa = P*PsiToMPa    
    TC = (T - 491.67)*(5/9)        
    SGO = 141.5/(API + 131.5)  
        
    rhoop = SGO + (0.00277*PMPa - 1.71*(10**-7)*PMPa**3 )*(SGO - 1.15)**2
        
    RhoDeadOil = rhoop/(0.972 + 3.81*(10**-4)*(TC + 17.78)**1.175)
                
    return RhoDeadOil
    
    
    
def RhoLiveOil_BW(P,T,API,Rs,SGG, Bo):
    
    """
    
    The BW equations are in metric units. Here, we will need to convert inputs, which are in Field Units, to Metric Units 
    
    P: psi
    T: Rankine
    Rs: SCF/STB
    SGG: unitless
    API: API degree
    Bo: M3/M3
    
    """ 
    SCFSTBtoLL = 1/5.61459275059329
    PMPa = P*PsiToMPa
    TC = (T - 491.67)*(5/9)  
    RsM = Rs*SCFSTBtoLL   
       
    SGO = 141.5/(API + 131.5)  
 
    rhos = (SGO + 0.0012*SGG*RsM)/Bo  # saturation density
        
    RhoLiveOil =  (rhos + (0.00277*PMPa - 1.71*(10**-7)*PMPa**3 )*(rhos - 1.15)**2) / (0.972 + 3.81*(10**-4)*(TC + 17.78)**1.175)
                        
    return RhoLiveOil     # g/cm3
      
def VpDeadOil_BW(P,T,API): # 
        
    """
    
    The BW equations are in metric units. Here, we will need to convert inputs, which are in Field Units, to Metric Units 
    
    P: psi
    T: Rankine
    Rs: SCF/STB
    SGG: unitless
    API: API degree
    Bo: M3/M3
    
    """ 
    
    PMPa = P*PsiToMPa
    TC = (T - 491.67)*(5/9)        
    SGO = 141.5/(API + 131.5)  
                       
    VpDeadOil = 2096*np.sqrt(SGO/(2.6 - SGO)) -3.7*TC + 4.64*PMPa + 0.0115*(np.sqrt((18.33/SGO) - 16.97) -1 )*TC*PMPa # results in m/s
                    
    return VpDeadOil # m/s
    
    
def VpLiveOil_BW(Rs, P, T, Bo, API): # 
        
    """
    
    The BW equations are in metric units. Here, we will need to convert inputs, which are in Field Units, to Metric Units 
    
    P: psi
    T: Rankine
    Rs: SCF/STB
    SGG: unitless
    API: API degree
    Bo: M3/M3
    
    """ 
    SCFSTBtoLL = 1/5.61459275059329
    PMPa = P*PsiToMPa
    TC = (T - 491.67)*(5/9)        
    SGO = 141.5/(API + 131.5)
    RsM = Rs*SCFSTBtoLL     
        
    rhops = (SGO/Bo)*(1 + 0.001*RsM)**-1
        
    """ The output velocity is in m/s"""
    Vp = 2096*np.sqrt(rhops/(2.6 - rhops)) -3.7*TC + 4.64*PMPa + 0.0115*(np.sqrt((18.33/rhops) - 16.97) - 1)*TC*PMPa 
       
                   
    return Vp   # m/s


def KOil_BW(Vp, rhoo): # 
        
    """ The BW equations are in metric units. Here, we will need to convert inputs, which are in Field Units, to Metric Units """ 

    Ko_BW = rhoo*(Vp**2)*(10**-6) # in GPa
               
    return Ko_BW 



def LiveOilElasticProperties_BW(P, T, API, Rs, SGG):
    
    """
    
    The BW equations are in metric units. Here, we will need to convert inputs, which are in Field Units, to Metric Units 
    
    P: psi
    T: Rankine
    Rs: SCF/STB
    SGG: unitless
    API: API degree
    
    
    """ 
    
    # PMPa = P*PsiToMPa
    # TC = (T - 491.67)*(5/9)        
    # RsM = Rs*SCFSTBtoLL 
    
    Bo = Bo_BW(Rs,SGG,API,T) # M3/M3 or BBL/STB [or unitless]
    
    Rho_oil = RhoLiveOil_BW(P,T,API,Rs,SGG, Bo) # g/cm3
    
    Vp_oil = VpLiveOil_BW(Rs, P,T, Bo,API) # m/s
    
    K_oil = KOil_BW(Vp_oil, Rho_oil) # GPa
    
    VpOil = Vp_oil*msTofts #ft/s
    RhoOil = Rho_oil*gcm3Tolbft3 # lbm/ft3
    KOil = K_oil*GPatoPsi # psi
        
                   
    return VpOil, RhoOil, KOil



def DeadOilElasticProperties_BW(P,T,API):
    
    """
    
    The BW equations are in metric units. Here, we will need to convert inputs, which are in Field Units, to Metric Units 
    
    P: psi
    T: Rankine
    Rs: SCF/STB
    SGG: unitless
    API: API degree    
    
    """     
    
    # PMPa = P*PsiToMPa
    # TC = (T - 491.67)*(5/9)
       
    Rho_oil = RhoDeadOil_BW(P,T,API) # g/cm3
    
    Vp_oil = VpDeadOil_BW(P,T,API) # m/s
    
    K_oil = KOil_BW(Vp_oil, Rho_oil) # GPa
    
    VpOil = Vp_oil*msTofts # ft/s
    RhoOil = Rho_oil*gcm3Tolbft3 # lbm/ft3
    KOil = K_oil*GPatoPsi # psi
                   
    return VpOil, RhoOil, KOil


"""

    SECTION 1.3: EQUATIONS FOR BRINE FROM BATZLE AND WANG (1992)

"""

def BrineElasticProperties_BW(P,T,S):
    
    """
    
    The BW equations are in metric units. Here, we will need to convert inputs, which are in Field Units, to Metric Units 
    
    P: psi
    T: Rankine
    Rs: SCF/STB
    SGG: unitless
    API: API degree
    
    
    """ 
    
    PMPa = P*PsiToMPa
    TC = (T - 491.67)*(5/9) 
    
    
    """ Table 2 from Batzle and Wang (1992) """
    
    w = np.zeros([5,4])
    
    w[0,0] = 1402.85         
    w[0,2] = 3.437*10**(-3)     
    w[1,0] = 4.871           
    w[1,2] = 1.739*10**(-4)
    w[2,0] = -0.04783        
    w[2,2] = -2.135*10**(-6) 
    w[3,0] = 1.487*10**(-4)   
    w[3,2] = -1.455*10**(-8) 
    w[4,0] = -2.197*10**(-7)  
    w[4,2] = 5.230*10**(-11) 
    w[0,1] = 1.524           
    w[0,3] = -1.197*10**(-5) 
    w[1,1] = -0.0111         
    w[1,3] = -1.628*10**(-6)
    w[2,1] = 2.747*10**(-4)   
    w[2,3] = 1.237*10**(-8) 
    w[3,1] = -6.503*10**(-7)  
    w[3,3] = 1.327*10**(-10) 
    w[4,1] = 7.987*10**(-10)  
    w[4,3] = -4.614*10**(-13)

    v_w = 0

    for i in np.arange(0,5,1):
        for j in  np.arange(0,4,1):   
            
            v_w = w[i,j]*(TC**(i))*(PMPa**(j)) + v_w
        
    Vp_brine = v_w + S*(1170 - 9.6*TC + 0.055*(TC**2) - 8.5*(10**-5)*(TC**3) + 2.6*PMPa - 0.0029*TC*PMPa - 0.0476*(PMPa**2)) + (S**1.5)*(780 - 10*PMPa + 0.16*(PMPa**2)) - 1820*(S**2) # m/s

    rho_w = 1 + (10**-6)*(-80*TC - 3.3*(TC**2) + 0.00175*(TC**3) + 489*PMPa - 2*TC*PMPa + 0.016*(TC**2)*PMPa - 1.3*(10**-5)*(TC**3)*PMPa - 0.333*(PMPa**2) - 0.002*TC*(PMPa**2))
    
    Rho_brine = rho_w + 0.668*S + 0.44*(S**2) + (10**-6)*S*( 300*PMPa - 2400*PMPa*S + TC*(80 + 3*TC - 3300*S - 13*PMPa + 47*PMPa*S))
    
    K_brine = Rho_brine*(Vp_brine**2)*(10**-6) # Data will be output in GPa 
     
             
    VpBrine = Vp_brine*msTofts #ft/s
    RhoBrine = Rho_brine*gcm3Tolbft3 # lbm/ft3
    KBrine = K_brine*GPatoPsi # psi
    
    
    return VpBrine, RhoBrine, KBrine 



"""

    SECTION 2.1: EQUATIONS FOR GAS FROM CARVALHO AND MORAES (2020)

"""


def TpcPpcGas_CM(SGG, gastype, yH2S, yCO2, yN2):
        
    """
    
    In order to calculate the pseudocritical properties of a certain gas, the main inputs are:
        
        1 - Specific gravity of gas (SGG)
        2 - The gas type: a gas may be either: 
            - (gastype = 1) An associated gas, which is usually the case for gas coming out of solution
            - (gastype = 2) A condensate gas, which is usually a heavier gas and characteristic of specific reservoirs
            By default, when no further information is available, assume (gastype = 1)
    
    If further information on the mole fractions of the non-hydrocarbon components in the gas is available, also inform:
        3 - yH2S: the mole fraction of sulfur
        4 - yCO2: the mole fraction of carbon dioxide
        5 - yN2: the mole fraction of nitorgren
        If the values for yH2S, yCO2, and yN2 are not known, set parameter to zero and automatically the function will yield 
        values for pseudocritical properties solely based on specific gravity    
    
    """

    # Definition of molecular weight of air, according to
    MWair = 28.97 # lbm/lbmol according to Harris (1984)
    
    # Definition of the molecular weight of the most common non-hydrocarbon components in natural gases
    MWH2S = 34.076 # lbm/lbmol
    MWCO2 = 44.01  # lbm/lbmol
    MWN2 = 28.0134 # lbm/lbmol
    
    # Definition of pseudocritial pressures and temperatures of the most common non-hydrocarbon components in natural gases
    PcH2S = 1305.3 # psia
    PcCO2 = 1070 # psia
    PcN2 = 492.52 # psia

    TcH2S = 671.6 # DEG R
    TcCO2 = 547.4308 # DEG R
    TcN2 = 227.146 # DEG R 
    
    # Calculation of the mole fraction of the hydrocarbon components       
    yHC = 1  - yH2S - yCO2 - yN2
       
    # Calculation of the pseudocritical properties per gas type according to Sutton and Hamman (2009)
    # TpcHC and PpcHC stand for pseudocritical properties calculated assuming that the gas is composed purely of hydrocarbons    
    
    # Associated gas equations
    if (gastype == 1):
    
        TpcHC = 120.1 + 429*SGG - 62.9*(SGG**2)  # output temperature is in Rankine
        PpcHC = 671.1 + 14*SGG - 34.3*(SGG**2)   # output in psia
          
    # Gas condensate gas equations
    if (gastype == 2):
    
        TpcHC = 164.3 + 357.7*SGG  -67.7*(SGG**2) # output temperature is in Rankine
        PpcHC = 744 - 125.4*SGG + 5.9*(SGG**2) # output in psia
        
    # Further arrangements are implemented for correcting for the non-hydrocarbon components
    # If the information on them is not available, the values for the hydrocarbon phase will collapse to the values calculated for TpcHC and PpcHC
    
    Ppcstar = yHC*PpcHC + yH2S*PcH2S + yCO2*PcCO2 + yN2*PcN2

    Tpcstar = yHC*TpcHC + yH2S*TcH2S + yCO2*TcCO2 + yN2*TcN2

    E = 120*( (yCO2 + yH2S)**0.9 - (yCO2 + yH2S)**1.6) + 15*(yH2S**0.5 - yH2S**4)

    TpcGas = Tpcstar - E

    PpcGas = Ppcstar*(Tpcstar - E)/(Tpcstar + yH2S*(1-yH2S)*E)
      
    return TpcGas, PpcGas, yHC

#
def ZfactorGasdf_CM(Tpr,Ppr):
    
    """
    
    The z-factor stands for a volumetric property of the matter. It is calculated as a function of three properties:
      
    - Pseudoreduced pressure
    - Pseudoreduced temperature    
    - Reduced density  
        
    """
    
    A1 = 0.3265
    A2 = -1.0700
    A3 = -0.5339
    A4 = 0.01569
    A5 = -0.05165
    A6 = 0.5475
    A7 = -0.7361
    A8 = 0.1844
    A9 = 0.1056
    A10 = 0.6134
    A11 = 0.7210
    
    c1 = A1 + (A2/Tpr) + (A3/(Tpr**3)) + (A4/(Tpr**4)) + (A5/(Tpr**5)) 
    c2 = A6 + (A7/Tpr) + (A8/(Tpr**2)) 
    c3 = (A7/Tpr) + (A8/(Tpr**2))   
    
    Zmin = 0*np.ones(len(Ppr))
    
    Zmax = 30*np.ones(len(Ppr))
    Zmid = np.zeros(len(Ppr))
    rhoR = np.zeros(len(Ppr))
    Zcheck = np.zeros(len(Ppr))
    
    for i in range(len(Ppr)):    
    
        while ( abs(Zmax[i] - Zmin[i]) > 10**-4 ):         
            
            Zmid[i] = (Zmax[i] + Zmin[i])/2     
            rhoR[i] = 0.27*Ppr[i]/(Zmid[i]*Tpr[i])                 
            Zcheck[i] = -Zmid[i] + 1 + (c1[i]*rhoR[i]) + (c2[i]*rhoR[i]**2) - (A9*c3[i]*rhoR[i]**5) + ((A10/Tpr[i]**3)*(1+(A11*rhoR[i]**2))*rhoR[i]**2*np.exp(-A11*rhoR[i]**2))          
                      
            if (Zcheck[i] < 0):  
                    
                Zmax[i] = Zmid[i]     
                
            else: 
                
                Zmin[i] = Zmid[i] 
         
    return Zmin, rhoR

""" for one point only (NO DATAFRAMES) """

def ZfactorGaspt_CM(Tpr,Ppr):
    
    """
    
    The z-factor stands for a volumetric property of the matter. It is calculated as a function of three properties:
      
    - Pseudoreduced pressure
    - Pseudoreduced temperature    
    - Reduced density  
        
    """
    
    A1 = 0.3265
    A2 = -1.0700
    A3 = -0.5339
    A4 = 0.01569
    A5 = -0.05165
    A6 = 0.5475
    A7 = -0.7361
    A8 = 0.1844
    A9 = 0.1056
    A10 = 0.6134
    A11 = 0.7210
    
    c1 = A1 + (A2/Tpr) + (A3/(Tpr**3)) + (A4/(Tpr**4)) + (A5/(Tpr**5)) 
    c2 = A6 + (A7/Tpr) + (A8/(Tpr**2)) 
    c3 = (A7/Tpr) + (A8/(Tpr**2))   
    
    Zmin = 0
    Zmax = 30   
           
    while ( abs(Zmax - Zmin) > 10**-10 ):         
            
        Zmid = (Zmax + Zmin)/2     
        rhoR = 0.27*Ppr/(Zmid*Tpr)                 
        Zcheck = -Zmid + 1 + (c1*rhoR) + (c2*rhoR**2) - (A9*c3*rhoR**5) + ((A10/Tpr**3)*(1+(A11*rhoR**2))*rhoR**2*np.exp(-A11*rhoR**2))          
                      
        if (Zcheck < 0):  
                    
            Zmax = Zmid     
                
        else: 
                
            Zmin = Zmid 
     
    ZfactorGas = Zmin
    RhoRGas = rhoR
    
    return ZfactorGas, RhoRGas

def CgtGas_CM(Tpr, Ppc, RhoR):
    
    # Use a correlation formula to calculate Cg, which is the gas isothermal
    # compressibility
    
    A1 = 0.3265
    A2 = -1.0700
    A3 = -0.5339
    A4 = 0.01569
    A5 = -0.05165
    A6 = 0.5475
    A7 = -0.7361
    A8 = 0.1844
    A9 = 0.1056
    A10 = 0.6134
    A11 = 0.7210
      
    Co= Tpr*(A1 + (A2/Tpr) + (A3/(Tpr**3)) + (A4/(Tpr**4))  +  (A5/(Tpr**5))     )
    C2= Tpr*(A6 + A7/Tpr + (A8/(Tpr**2)) )
    C3= -A9*(A7 + (A8/Tpr) )
    C4= A10/Tpr**2
    C5= A11

    DP = Tpr + 2*Co*RhoR + 3*C2*(RhoR**2) + 6*(C3*(RhoR**5)) + (C4*(RhoR**2))*( 3 + 3*(C5*(RhoR**2)) - 2*((C5*(RhoR**2))**2))*np.exp(-C5*(RhoR**2))

    Cr = 0.27/(RhoR*DP)

    CgtGas = Cr/Ppc # in psi^-1
       
    return CgtGas

def YGas_CM(TR,Tpr, Ppr, SGG, RhoR,z,yHC,yH2S,yCO2,yN2):
    
    # First we need to estimate the ideal isobaric heat capacity of pure
    # hydrocarbon gas mixture, Btu/lb-more DEG R
      
    CpH2S = -1.41982*(10**-9)*(TR**3) + 4.14115*(10**-6)*(TR**2) - 1.65889*(10**-3)*(TR) + 8.06974
        
    CpCO2 = 1.40313*(10**-9)*(TR**3) -6.00036*(10**-6)*(TR**2) + 1.13292*(10**-2)*(TR) + 4.30864
        
    CpN2 = 1.00993*(10**-10)*(TR**3) + 5.19807*(10**-7)*(TR**2) - 6.07453*(10**-4)*(TR) + 7.12153
                   
    CpHCo_Numerator = 4.20572 - 3.07077*np.log(Tpr) + 0.369082*SGG + 8.99319*(SGG**2) - 1.22268*(SGG**3)
    CpHCo_Denominator = 1 + 0.0340928*(SGG) - 0.994010*np.log(Tpr) + 0.499824*(np.log(Tpr)**2) - 0.178017*(np.log(Tpr)**3)
    CpHCo = CpHCo_Numerator/CpHCo_Denominator # BTU /DEG R-lbmol   
       
    Cpmo = CpHCo*yHC + yH2S*CpH2S + yCO2*CpCO2 + yN2*CpN2 # BTU /DEG R-lbmol
        
    # Heat capacity ratio:   
    
    #    DAK constants
    A1 = 0.3265
    A2 = -1.0700
    A3 = -0.5339
    A4 = 0.01569
    A5 = -0.05165
    A6 = 0.5475
    A7 = -0.7361
    A8 = 0.1844
    A9 = 0.1056
    A10 = 0.6134
    A11 = 0.7210
     
    Tpr = Tpr
    
    RR = 1.98588 # BTU /DEG R-lbmol

    A = (6*(A3/Tpr**3) + 12*(A4/Tpr**4)  + 20*(A5/Tpr**5))*RhoR
    
    B = (A8/Tpr**2)*(RhoR**2)
    
    C = (2/5)*A9*(A8/(Tpr**2))*(RhoR**5)
    
    D = 6*(A10/Tpr**3)*( 1/A11  - ( ((A11*(RhoR**2) + 2)*np.exp(-A11*(RhoR**2)) )/(2*A11) )   )
               
    term1 = (A2/Tpr + 3*(A3/Tpr**3) + 4*(A4/Tpr**4)  + 5*(A5/Tpr**5))*RhoR
    term2 = (A7/Tpr + 2*(A8/Tpr**2) )*(RhoR**2)
    term3 = A9*( A7/Tpr + 2*(A8/(Tpr**2)) )*(RhoR**5)
    
    E = (RhoR/0.27)*( - term1 - term2 + term3 -3*A10*(1 + A11*(RhoR**2) )*((RhoR**2)/Tpr**3)*np.exp(-A11*(RhoR**2))  + z)

    Part1 = A1 + A2/Tpr  + A3/(Tpr**3) + A4/(Tpr**4) + A5/(Tpr**5)
    Part2 = A6 + A7/Tpr + A8/(Tpr**2)
    Part3 = A7/Tpr + A8/(Tpr**2)  
    Part4 = (1 + A11*RhoR**2 + (A11**2)*(RhoR**4) )*2*A10*(RhoR/(Tpr**3) )*np.exp(-A11*(RhoR**2)) 
        
    F =  (RhoR*Tpr/0.27)*(Part1 + 2*Part2*RhoR -5*A9*Part3*(RhoR**4) + Part4) + Tpr*z/0.27

   
    Cp = RR*0.27*(Tpr/(RhoR**2))*(E**2/F) - RR*(A + B - C +D) + Cpmo - RR
    Cv = -RR*(A + B - C +D) + Cpmo - RR
    
    YGas = Cp/Cv          
    
    return YGas

def GasElasticProperties_CM(P, T, SGG, gastype, yH2S,yCO2,yN2):
  
    Tpc, Ppc, yHC =  TpcPpcGas_CM(SGG, gastype, yH2S, yCO2, yN2)

    Tpr = T/Tpc
           
    Ppr = P/Ppc
    
    if np.array(P).size>1:
                
        z_factor, rho_r = ZfactorGasdf_CM(Tpr,Ppr)
        
    else:
        
        z_factor, rho_r = ZfactorGaspt_CM(Tpr,Ppr)
    
    Cgt = CgtGas_CM(Tpr, Ppc, rho_r)
    
    R = 10.7316
    MWair = 28.97 # lbm/lbmol
    MWgas = SGG*MWair    
    
    RhoGas = (P*MWgas)/(z_factor*R*T) # lbm/ft3
    
    YGas = YGas_CM(T,Tpr, Ppr, SGG, rho_r,z_factor,yHC,yH2S,yCO2,yN2)
           
    Cgs = Cgt/YGas
    KGas = Cgs**-1  # psi
    multiplier = np.sqrt( 32.174049/((1/12)**2) )
    VpGas = multiplier*(KGas/RhoGas)**(1/2) # ft/sec
    
    return VpGas, RhoGas, KGas


"""

    SECTION 2.2: EQUATIONS FOR OIL FROM CARVALHO AND MORAES (2020)

"""

def MWOil_CM(API,Kw,Rs,SGG):
    
    if Kw == 0:
        
        kwangle = (12.3-11.4)/(70-5)
        kwintercept = 11.5
        Kw = kwangle*API + kwintercept
              
    sgo = 141.5/(API + 131.5) # specific gravity of oil
 
    mwsto = (Kw*((sgo)**0.84573)/4.5579)**6.58848 # molecular weight of the oil phase in stock tank

    xo = (1 + 7.521*(10**-6)*Rs*mwsto/API)**-1

    MWo = xo*mwsto + 28.964*(1 - xo)*SGG

    return MWo

def TpcPpcOil_CM(MWo,Ap,Bp,Cp,Dp,At,Bt,Ct,Dt):
    
    
    
    Ppc = Ap + Bp*MWo + Cp*MWo*(np.log(MWo) + Dp  ) # in [psia]
    Tpc = At + Bt*MWo + Ct*MWo*(np.log(MWo) + Dt  ) # in [DEG R]
    
    #print(Ap)
    
    
        
    return Ppc,Tpc 


def ZfactorOildf_CM(Tpr,Ppr, A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11):
    
    c1 = A1 + (A2/Tpr) + (A3/(Tpr**3)) + (A4/(Tpr**4)) + (A5/(Tpr**5)) 
    c2 = A6 + (A7/Tpr) + (A8/(Tpr**2)) 
    c3 = (A7/Tpr) + (A8/(Tpr**2))   
    
    Zmin = 0*np.ones(len(Ppr))
    
    Zmax = 30*np.ones(len(Ppr))
    Zmid = np.zeros(len(Ppr))
    rhoR = np.zeros(len(Ppr))
    Zcheck = np.zeros(len(Ppr))
    
    for i in range(len(Ppr)):    
    
        while ( abs(Zmax[i] - Zmin[i]) > 10**-4 ):         
            
            Zmid[i] = (Zmax[i] + Zmin[i])/2     
            rhoR[i] = 0.27*Ppr[i]/(Zmid[i]*Tpr[i])                 
            Zcheck[i] = -Zmid[i] + 1 + (c1[i]*rhoR[i]) + (c2[i]*rhoR[i]**2) - (A9*c3[i]*rhoR[i]**5) + ((A10/Tpr[i]**3)*(1+(A11*rhoR[i]**2))*rhoR[i]**2*np.exp(-A11*rhoR[i]**2))          
                      
            if (Zcheck[i] < 0):  
                    
                Zmax[i] = Zmid[i]     
                
            else: 
                
                Zmin[i] = Zmid[i] 
    
    ZfactorOil = Zmin
    RhoROil = rhoR           
               
         
    return ZfactorOil,RhoROil



def ZfactorOilpt_CM(Tpr,Ppr, A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11):
    
    c1 = A1 + (A2/Tpr) + (A3/(Tpr**3)) + (A4/(Tpr**4)) + (A5/(Tpr**5)) 
    c2 = A6 + (A7/Tpr) + (A8/(Tpr**2)) 
    c3 = (A7/Tpr) + (A8/(Tpr**2))   
    
    Zmin = 0
    Zmax = 30   
           
    while ( abs(Zmax - Zmin) > 10**-10 ):         
            
        Zmid = (Zmax + Zmin)/2     
        rhoR = 0.27*Ppr/(Zmid*Tpr)                 
        Zcheck = -Zmid + 1 + (c1*rhoR) + (c2*rhoR**2) - (A9*c3*rhoR**5) + ((A10/Tpr**3)*(1+(A11*rhoR**2))*rhoR**2*np.exp(-A11*rhoR**2))          
                      
        if (Zcheck < 0):  
                    
            Zmax = Zmid     
                
        else: 
                
            Zmin = Zmid 
         
    ZfactorOil = Zmin
    RhoROil = rhoR           
               
         
    return ZfactorOil,RhoROil


def CgtOil_CM(Tpr,RhoR,Ppc,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11):
        
    Co = Tpr*(A1 + (A2/Tpr) + (A3/(Tpr**3)) + (A4/(Tpr**4))  +  (A5/(Tpr**5))     )
    C2 = Tpr*(A6 + A7/Tpr + (A8/(Tpr**2)) )
    C3 = -A9*(A7 + (A8/Tpr) )
    C4 = A10/Tpr**2
    C5 = A11
    DP = Tpr + 2*Co*RhoR + 3*C2*(RhoR**2) + 6*(C3*(RhoR**5)) + (C4*(RhoR**2))*( 3 + 3*(C5*(RhoR**2)) - 2*((C5*(RhoR**2))**2))*np.exp(-C5*(RhoR**2))
    Cr = 0.27/(RhoR*DP)
    CgtOil = Cr/Ppc # psi**-1
        
    return CgtOil



def YOil_CM(Ppr, Tpr, MWo):
    
    x0 = Ppr
    x1 = Tpr
    x2 = MWo    
    YOil = -0.132172*x1 -6.67239e-05*x0*x1 -0.00106716*x1*x2 + (1.71389e-08)*x0**3 - 0.00157865*(x0*x1**2) - 2.67187e-06*x0*x1*x2 + (8.4033e-11)*x0**4 + (2.59829e-08)*x1*x0**3 + 1.35569
    
    return YOil


def VelocityOildf_CM(P, T, MWo):
    
    VpOil = np.zeros(len(MWo))    

    for i in range(len(MWo)):
        if MWo[i] < 450:
               
            VpOil[i] = P[i]*0.114227 + -8.848*T[i] + 26.2186*MWo[i] -1.67112e-06*P[i]**2 + 0.000136532*P[i]*T[i] -0.000365705*P[i]*MWo[i] + 0.00239495*T[i]**2 -0.00158273*T[i]*MWo[i] -0.0455288*MWo[i]**2 + 5230.986706351145
        
        if MWo[i] > 450:
            VpOil[i] = 0.088323*P[i] -8.73396*T[i] +4.217*MWo[i] + 7883.7420
    
    return VpOil


def VelocityOilpt_CM(P, T, MWo):
  
    
    if MWo < 450:
               
        VpOil = P*0.114227 + -8.848*T + 26.2186*MWo -1.67112e-06*P**2 + 0.000136532*P*T -0.000365705*P*MWo + 0.00239495*T**2 -0.00158273*T*MWo -0.0455288*MWo**2 + 5230.986706351145
        
    if MWo > 450:
        
        VpOil = 0.088323*P -8.73396*T +4.217*MWo + 7883.7420
    
    return VpOil


def OilElasticProperties_CM(P,T, MWo, API,Kw,Rs,SGG):
    
    
    
    A1 = 0.3265
    A2 = -1.0700
    A3 = -0.5339
    A4 = 0.01569
    A5 = -0.05165
    A6 = 0.5475
    A7 = -0.7361
    A8 = 0.1844
    A9 = 0.1056
    A10 = 0.6134
    A11 = 0.7210
    
    Ap = 768.1
    Bp = -4.919
    Cp = 1.302
    Dp = -3.366
    
    At = 135.6
    Bt = 10.864
    Ct = -2.81
    Dt = -3.366
    
            
                
    Ppc,Tpc = TpcPpcOil_CM(MWo,Ap,Bp,Cp,Dp,At,Bt,Ct,Dt)
    
    Ppr = P/Ppc
    Tpr = T/Tpc
        
    R = 10.7316 # [ft3*psi]/[lbmol*DegR]
        
    
    if np.array(P).size>1:
                
        z_factor, rho_r = ZfactorOildf_CM(Tpr,Ppr,A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11)
        
    else:
        
        z_factor, rho_r = ZfactorOilpt_CM(Tpr,Ppr,A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11)
    
        
    RhoOil = (P*MWo)/(z_factor*R*T) # % [lbm/ft3]   
    
#    multiplier = np.sqrt( 32.174049/((1/12)**2) )
    
    if np.array(MWo).size>1:
                
        VpOil =  VelocityOildf_CM(P, T, MWo)
        
    else:
        
        VpOil =  VelocityOilpt_CM(P, T, MWo)
      
            
    kOil = RhoOil*(VpOil**2)/(32.17*12**2)
    
    return VpOil, RhoOil, kOil
    


"""------------------------------------------------------------------------------------------"""


def KFluid_func(Sw, So, Sg, kw, ko, kg):

#    if kw ==0:  
#            
#        k_fluid = 1/(So/ko + Sg/kg)      # psia   
#    else:
#        
#        k_fluid = 1/(Sw/kw + So/ko + Sg/kg)
#        
#    
#    if ko ==0:  
#            
#        k_fluid = 1/(Sw/kw + Sg/kg)      # psia   
#    else:
#        
#        k_fluid = 1/(Sw/kw + So/ko + Sg/kg)
#        
#    if kg ==0:  
#            
#        k_fluid = 1/(Sw/kw + So/ko)      # psia   
#    else:
#        
#        k_fluid = 1/(Sw/kw + So/ko + Sg/kg)   
        
    k_fluid = 1/(Sw/kw + So/ko + Sg/kg)
         
    return k_fluid


def rho_fluid_func(Sw, So, Sg, rho_brine, rho_oil, rho_gas):       
            
    rho_fluid =  Sw*rho_brine + So*rho_oil + Sg*rho_gas
             
    return rho_fluid


def k_sat_func(phi,k_fluid, k_0, k_frame):       
            
    k1 = phi/k_fluid + (1-phi)/k_0 - k_frame/(k_0*k_0) 
    k_sat = k_frame + ((1 - k_frame/k_0)**2)/k1 
             
    return k_sat



def k_sat_fromdata(vp_data, vs_data, rho_sat_data):       
            
    multiplier = 32.17*(12**2)
    k_sat_data = rho_sat_data*((vp_data**2) - (vs_data**2)*(4/3))*(multiplier**-1) # psia
    m_sat_data = rho_sat_data*(vs_data**2)*(multiplier**-1)                      # psia -- shear modulus remains constant
             
    return k_sat_data, m_sat_data


def k_frame_Gfunc(k_fluid_i, k_0, k_sat_i, phi):       
            
    k1_dash = k_sat_i*( (phi*k_0/k_fluid_i) + 1 - phi) - k_0 
    k2_dash = (phi*k_0)/k_fluid_i + (k_sat_i/k_0) - 1 - phi 
    k_frame = k1_dash/k2_dash   # psia
         
    return k_frame



def k_sat_fromdata(vp_data, vs_data, rho_sat_data):       

    multiplier = 32.17*(12**2)
    k_sat_data = rho_sat_data*((vp_data**2) - (vs_data**2)*(4/3))*(multiplier**-1) # psia
    m_sat_data = rho_sat_data*(vs_data**2)*(multiplier**-1)                      # psia -- shear modulus remains constant
             
    return k_sat_data, m_sat_data





