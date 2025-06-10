import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
import itdtools
import scipy.signal as scisig
from importlib import reload
import pickle
from scipy.signal import find_peaks
#
import IC
reload(IC)
#
from scipy.io import wavfile
from scipy.signal import lfilter,lfilter_zi, gammatone, sosfiltfilt, butter

winlen=4001
fs=96000
id_nyquist=int(winlen/2)
farr=np.arange(1,id_nyquist+1)*fs/winlen/1000
ncut=41#31*2

Jlat=20.
save_flag=True
tvals=np.arange(.01,.35,.02)
savestruct={}

fs, mono1 = wavfile.read('sounds/pure400.wav')
fs, mono2 = wavfile.read('sounds/noise.wav')

mono1=1.*mono1[0:int(.3*fs)]
mono2=1.*mono2[0:int(.3*fs)]



#b, a =gammatone(400, fs=fs, ftype='fir')
#zi = lfilter_zi(b, a)
#z1, _ = lfilter(b, a, mono1, zi=zi*mono1[0])
#z2, _ = lfilter(b, a, mono2, zi=zi*mono2[0])
sos=butter(2,np.array([300,500])/fs*2,output='sos', btype='bandpass')# membrane filtering
mono1n=sosfiltfilt(sos,mono1)
mono2n=sosfiltfilt(sos,mono2)


  
sos=butter(1,500/fs*2,output='sos')# membrane filtering
mono1=sosfiltfilt(sos,mono1)
mono2=sosfiltfilt(sos,mono2)

#renormalization
rms1=np.sqrt(np.nanmean(mono1n**2))
rms2=np.sqrt(np.nanmean(mono2n**2))
nrm=np.sqrt(np.nanmean(mono1**2))
mono1=mono1/nrm
mono2=mono2*rms1/rms2/nrm
   



   
 


        
 
#
itd_sig=[1.,-1.]
db_sig=np.append(-1.*np.arange(0,27,0.333),np.array(-200))
lmax=int(len(db_sig)-1)
level_sig=10**(db_sig/20)

ITDmax=.7
itd_list=np.arange(0,ITDmax,.01)

Ncells=len(itd_list)
bins0=np.arange(len(itd_list)+1)

if save_flag:
    for nl,level in enumerate(level_sig):
        leveldb=db_sig[nl]
        for mdt,dts in enumerate(itd_sig):
            key_str=str(nl)+' '+str(mdt)
            print(key_str)
            
            stereo=np.array([level*mono1+mono2,dts*level*mono1+mono2])
            
            patterns=[]
            jpatterns=[]
            
            Ppatterns=IC.IC_Filterbank_fun(1.*stereo,fs, itd_list,_flow=-1)
            Ppatterns=np.transpose(Ppatterns, (1,0,2))
    
            for nneuron, dt_neuron in enumerate(itd_list):
            
                ic,_,_,pattern=IC.ICfun(1.*stereo,fs, dt_neuron,_flow=-1)
                jeff=IC.Jeffress(1.*stereo, int(dt_neuron*fs*1e-3),fs,_flow=-1)
                jpatt=IC.stspec(jeff)             
                patterns.append(pattern)
                jpatterns.append(jpatt)

            savestruct[key_str]={'jpatterns':jpatterns,   
                                 'patterns':patterns,
                                 'Ppatterns': Ppatterns}

    with open('pkl/threshold.pkl','wb') as fd:
        pickle.dump(savestruct,fd)
else:
    with open('pkl/threshold.pkl','rb') as fd:
        savestruct=pickle.load(fd)


def addinhibition(patterns,J):
    patt=np.mean(np.array(patterns),axis=1)
    patt=patt/np.max(patt)
    lateralinhibition=np.ones((patt.shape[0],1))@np.mean(patt,axis=0).reshape(1,-1)
    lpatt=patt/(1+J*lateralinhibition)

    return lpatt, patt


def dissim(x,y):

    return np.nanmax((x-y)/y,axis=0)#/np.mean(y,axis=0)
    #return np.diff(np.max(x,axis=0)/np.max(y))
    
noise0=savestruct[str(lmax)+' 0']['patterns']
Pnoise0=savestruct[str(lmax)+' 0']['Ppatterns']
jnoise=savestruct[str(lmax)+' 0']['jpatterns']
noise, nnoise=addinhibition(noise0, Jlat)
Pnoise, Pnnoise=addinhibition(Pnoise0, Jlat)
jnoise, njnoise=addinhibition(jnoise, Jlat)
Dnoise, Dnnoise=addinhibition(noise0, 2*Jlat)
DPnoise, DPnnoise=addinhibition(Pnoise0, 2*Jlat)



s0n0=[]
s0n0P=[]
s0npi=[]
s0npiP=[]
s0n0ff=[]
s0npiff=[]

Ds0n0=[]
Ds0n0P=[]
Ds0npi=[]
Ds0npiP=[]
Ds0n0ff=[]
Ds0npiff=[]
for nl,level in enumerate(level_sig[:-1]):
    leveldb=db_sig[nl]

    s0n0.append([])
    s0n0P.append([])
    s0npi.append([])
    s0npiP.append([])
    s0n0ff.append([])
    s0npiff.append([])
    
    Ds0n0.append([])
    Ds0n0P.append([])
    Ds0npi.append([])
    Ds0npiP.append([])
    Ds0n0ff.append([])
    Ds0npiff.append([])
    for mdt,dts in enumerate(itd_sig):
        key_str=str(nl)+' '+str(mdt)
           
        patterns=savestruct[key_str]['patterns']
        Ppatterns=savestruct[key_str]['Ppatterns']
        #
        patt, npatt=addinhibition(patterns, Jlat)
        Ppatt, Pnpatt=addinhibition(Ppatterns, Jlat)
        
        snr=dissim(patt[:,:ncut],noise[:,:ncut])
        Psnr=dissim(Ppatt[:,:ncut],Pnoise[:,:ncut])
        n_snr=dissim(npatt[:,:ncut],nnoise[:,:ncut])

        Dpatt, Dnpatt=addinhibition(patterns, 2*Jlat)
        DPpatt, DPnpatt=addinhibition(Ppatterns, 2*Jlat)
        
        Dsnr=dissim(Dpatt[:,:ncut],Dnoise[:,:ncut])
        DPsnr=dissim(DPpatt[:,:ncut],DPnoise[:,:ncut])
        Dn_snr=dissim(Dnpatt[:,:ncut],Dnnoise[:,:ncut])
        
        for tval in tvals:
            pks,th=find_peaks(snr, prominence=tval, wlen=20)
            Ppks,Pth=find_peaks(Psnr, prominence=tval, wlen=20)
            pksNo,thNo=find_peaks(n_snr, prominence=tval, wlen=20)

            Dpks,Dth=find_peaks(Dsnr, prominence=tval, wlen=20)
            DPpks,DPth=find_peaks(DPsnr, prominence=tval, wlen=20)
            DpksNo,DthNo=find_peaks(Dn_snr, prominence=tval, wlen=20)


            if mdt==0:
                s0n0[-1].append(1.*(len(pks)>0))
                s0n0P[-1].append(1.*(len(Ppks)>0))
                s0n0ff[-1].append(1.*(len(pksNo)>0))

                Ds0n0[-1].append(1.*(len(Dpks)>0))
                Ds0n0P[-1].append(1.*(len(DPpks)>0))
                Ds0n0ff[-1].append(1.*(len(DpksNo)>0))
            else:
                s0npi[-1].append(1.*(len(pks)>0))
                s0npiP[-1].append(1.*(len(Ppks)>0))
                s0npiff[-1].append(1.*(len(pksNo)>0))
                Ds0npi[-1].append(1.*(len(Dpks)>0))
                Ds0npiP[-1].append(1.*(len(DPpks)>0))
                Ds0npiff[-1].append(1.*(len(DpksNo)>0))
                
        
        
        

fig=plt.figure()
ax=fig.add_subplot(3,3,1)
tmp=np.array(s0npi)
levmin=np.argmin((tmp>0),axis=0)
cont=ax.plot(tvals,db_sig[levmin], color='r')
tmp=np.array(s0n0)
levmin=np.argmin((tmp>0),axis=0)
cont=ax.plot(tvals,db_sig[levmin], color='k')
tmp=np.array(Ds0npiP)
levmin=np.argmin((tmp>0),axis=0)
cont=ax.plot(tvals,db_sig[levmin], 'r--')
tmp=np.array(Ds0n0P)
levmin=np.argmin((tmp>0),axis=0)
cont=ax.plot(tvals,db_sig[levmin], 'k--')

ax.set_title('inhibition', fontsize='10')
#ax.set_ylabel('Threshold (dB)')
ax.set_ylabel('Threshold (dB)')
ax.set_xlabel('Peak prominence')
ax.set_ylim([-27,0])
ax.set_xlim([0,0.35])
ax.set_xticks([0,0.1,0.2,0.3])
ax.set_yticks([0, -5 ,-10, -15, -20, -25])
ax.grid()

ax=fig.add_subplot(3,3,2)
tmp=np.array(s0npiff)
levmin=np.argmin((tmp>0),axis=0)
cont=ax.plot(tvals,db_sig[levmin], color='r', label='S$_\pi$N$_0$')
tmp=np.array(s0n0ff)
levmin=np.argmin((tmp>0),axis=0)
cont=ax.plot(tvals,db_sig[levmin], color='k', label='S$_0$N$_0$')
ax.set_title('w\o inhibition', fontsize='10')
#ax.set_ylabel('Threshold (dB)')
ax.set_xlabel('Peak prominence')
ax.set_ylim([-27,0])
ax.set_xlim([0,0.35])
ax.set_xticks([0, 0.1,0.2, 0.3])
ax.set_yticks([0, -5 ,-10, -15, -20, -25])
ax.grid()
#ax.legend(frameon=False)



ax=fig.add_subplot(3,3,3)
tmp=np.array(s0npiP)
levmin=np.argmin((tmp>0),axis=0)
cont=ax.plot(tvals,db_sig[levmin], color='r', label='S$_\pi$N$_0$')
tmp=np.array(s0n0P)
levmin=np.argmin((tmp>0),axis=0)
cont=ax.plot(tvals,db_sig[levmin], color='k', label='S$_0$N$_0$')
tmp=np.array(Ds0npiP)
levmin=np.argmin((tmp>0),axis=0)
cont=ax.plot(tvals,db_sig[levmin], 'r--')
tmp=np.array(Ds0n0P)
levmin=np.argmin((tmp>0),axis=0)
cont=ax.plot(tvals,db_sig[levmin], 'k--')
ax.set_title('Peripheral filters + inh.', fontsize='10')
#ax.set_ylabel('Threshold (dB)')
ax.set_xlabel('Peak prominence')
ax.set_ylim([-27,0])
ax.set_xlim([0,0.3])
ax.set_xticks([0, 0.1,0.2,0.3])
ax.set_yticks([0, -5 ,-10, -15, -20, -25])
ax.grid()
ax.legend(frameon=False)










fig.subplots_adjust(wspace=0.5)
plt.show(block=0)

#plt.savefig('fig9.pdf')

