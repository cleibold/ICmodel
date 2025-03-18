import numpy as np
from scipy.stats import pearsonr as corrcoef
from importlib import reload
#
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import root as findroot
from scipy.signal import sosfiltfilt, butter
#
from optpars import psil, psim
import pickle
flock=100e3
flow=.5e3#2.5e3
filter_order=1

## change this parameter to include Gaussian jitter of optimal parameters (Fig 7C of Manuscript: paranoise=0.2)
paranoise=0.
##

tauadapt=2

aMSO,aLSO=0,0
bMSO,bLSO=0.1,0.1
cMSO,cLSO=0,0# 5/tauadapt/fs, 10/tauadapt/fs
decay=0

with open('interpol_a_7_.pkl', 'rb') as f_d:
    a_fun=pickle.load(f_d)
    
with open('interpol_pc_7_.pkl', 'rb') as f_d:
    pc_fun=pickle.load(f_d)

def ICfun(stereo, fs, dt, winshift=2000,_flow=-1):
    global aMSO,aLSO,decay
    
    power = np.mean(stereo[0,:]**2)
    #
    #
    sig_phase=np.zeros_like(stereo[0,:])
    mso_phase=np.zeros_like(stereo[0,:])
    lso_phase=np.zeros_like(stereo[0,:])
    sig_0=np.zeros((2,stereo.shape[1]))
    #

    winlen=winshift*2
    if np.mod(winlen,2)==0:
        winlen += 1

    envelope=np.blackman(winlen)
    decay = np.exp(-winshift/(fs*tauadapt) )


    id_nyquist=int(winlen/2)
    farr=np.arange(1,id_nyquist+1)*fs/winlen
    aMSO,aLSO=np.zeros(id_nyquist),np.zeros(id_nyquist)
    
    aarr = a_fun(farr*1e-3,dt)*(1.+paranoise*np.random.randn(len(farr)))
    psic = pc_fun(farr*1e-3)*(1.+paranoise*np.random.randn(len(farr)))

    shortterm_spectrum=[]
    i0=0   
    while i0+winlen<stereo.shape[1]:
        i1=i0+winlen    
        sig_local=stereo[:,i0:i1]*envelope
        fsig_local=np.fft.fft(sig_local, axis=1)
        
        fIC=np.zeros_like(fsig_local[0,:])
        fmso=np.zeros_like(fsig_local[0,:])
        flso=np.zeros_like(fsig_local[0,:])

        #sig_0 for power normalization only
        fsig_0=np.zeros_like(fsig_local)
        fsig_0[:,1:id_nyquist+1] = fsig_local[:,1:id_nyquist+1]
        #
        #
        #print(np.mean(aMSO),np.mean(aLSO))
        fIC[1:id_nyquist+1],fmso[1:id_nyquist+1], flso[1:id_nyquist+1] = ICcircuit(fsig_local[:,1:id_nyquist+1], aarr, psic)

        id_no_phaselock=np.where(farr>flock)[0]+1
        #
        fIC[id_no_phaselock]=np.abs(np.mean(fsig_0[:,id_no_phaselock],axis=0))

        sig_phase[i0:i1] += np.real(np.fft.ifft(fIC))*2
        mso_phase[i0:i1] += np.real(np.fft.ifft(fmso))*2
        lso_phase[i0:i1] += np.real(np.fft.ifft(flso))*2

        Icin=np.abs(fIC[0:id_nyquist+1])**2

        shortterm_spectrum.append(Icin)
        
        sig_0[:,i0:i1] += np.real(np.fft.ifft(fsig_0))*2
        i0=i0+winshift




    power_phase=np.nanmean(sig_0**2)

    ic=sig_phase*np.sqrt((power/power_phase))
    if _flow>0:
        sos=butter(filter_order,_flow/fs*2,output='sos')
        ic=sosfiltfilt(sos,ic)
        
 
    return ic, mso_phase, lso_phase, np.array(shortterm_spectrum)*power/power_phase



def ICcircuit(fstereo,a,psic):
    global aMSO, aLSO, cMSO, cLSO, bMSO, bLSO, decay

    aMSO *= decay
    aLSO *= decay
    
    idneg=np.where(a<0)[0]
    
    #gammaR = 1-a*np.exp(-1j*(psil+psic))
    #gammaL = (np.exp(-1j*psim) + a*np.exp(-1j*psic))
    #gammaR[idneg] = 1 + a[idneg]
    #gammaL[idneg] = (np.exp(-1j*psim) - a[idneg] * np.exp(-1j*psil))

    #gammaR2 = np.abs(gammaR)**2
    #gammaL2 = np.abs(gammaL)**2

    #idmax=np.where(np.abs(a)>2)[0]
    #a[idmax]=np.sign(a[idmax])*2
    
    Rear=fstereo[0,:]
    Lear=fstereo[1,:]
    
    MSO  = Rear + Lear*np.exp(-1j*psim)

    LSO  = Lear*np.exp(-1j*psic) - Rear*np.exp(-1j*(psil+psic))

    iLSO = Rear - Lear*np.exp(-1j*psil)

    LSO[idneg] = iLSO[idneg]
    
    MSO=MSO*(1-aMSO+bMSO)/(1+bMSO)
    LSO=LSO*(1-aLSO+bLSO)/(1+bLSO)
    
    IC  = 0.5*(MSO + a*LSO)/np.sqrt(1+a**2)#np.sqrt(gammaR2 + gammaL2 + np.sqrt(gammaR2*gammaL2)*2)
    
    aMSO += np.abs(MSO)*(1.-aMSO)*cMSO
    aLSO += np.abs(LSO)*(1.-aLSO)*cLSO

    aMSO[aMSO>1.]=1.
    aLSO[aLSO>1.]=1.
    
    return IC, MSO, LSO


def Jeffress(stereo,noff,fs,_flow=-1):
    

    Rear=stereo[0]
    Lear=stereo[1]
    
    dur=stereo.shape[1]-np.abs(noff)

    output = Lear
    if noff>0:
        output[0:dur] += Rear[noff:]
    elif noff<0:
        output[np.abs(noff):] += Rear[0:-np.abs(noff)]
    else:
        output += Rear


    if _flow>0:
        sos=butter(filter_order,_flow/fs*2,output='sos')
        output=sosfiltfilt(sos,output)
 
  
    return output/3

def stspec(out):
    winshift=2001
    winlen=winshift*2
    if np.mod(winlen,2)==0:
        winlen += 1
        
    id_nyquist=int(winlen/2)
    envelope=np.blackman(winlen)
    i0=0
    shortterm_spectrum=[]
    while i0+winlen<out.shape[0]:
        i1=i0+winlen    
        o_local=out[i0:i1]*envelope
        fo_local=np.fft.fft(o_local)
    
        pw=np.abs(fo_local[0:id_nyquist+1])**2
        
        shortterm_spectrum.append(pw)

        i0=i0+winshift

    return np.array(shortterm_spectrum)



def rcorr(nshift,sig1,sig2,fs):

    #print(nshift)
    if nshift>=0:
        ssig1=sig1[nshift:]
        ssig2=sig2
    else:
        ssig2=sig2[-nshift:]
        ssig1=sig1
        
    #print(len(ssig),len(sig1),len(sig2),nshift)
    slen=np.min([len(ssig1), len(ssig2)])
    nas = np.isnan(ssig1[0:slen]+ssig2[0:slen])
    #print(np.sum(nas), slen)
    
    if slen>np.sum(nas)+1:
        r=corrcoef(ssig1[0:slen][~nas],ssig2[0:slen][~nas])[0]
    else:
        r=np.nan
        
    return r

def rmaxobjective(nshift,sig1,sig2,fs):
    
    r=rcorr(nshift,sig1,sig2,fs)
    
    
    #print (nshift, r)
    return -np.abs(r)
    

def rpearson(sig1,sig2,fs,_flow=-1):

    smax=int(3*fs*1e-3)

    if _flow>0:
        sos=butter(filter_order,_flow/fs*2,output='sos')
        sig1=sosfiltfilt(sos,sig1)
        sig2=sosfiltfilt(sos,sig2)
    
    shifts=np.arange(-smax,smax+1).astype(int)
    
    rarr=[]
    for nshift in shifts:
        rarr.append(rmaxobjective(nshift,sig1,sig2,fs))

    minob=np.nanmin(rarr)
    
    return -minob

