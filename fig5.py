import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
import itdtools
import scipy.signal as scisig
from importlib import reload
import pickle
import matplotlib.cm as pltcol
colr=[pltcol.get_cmap('winter')(f) for f in [0,100, 200]]
#
import IC
reload(IC)
#
from scipy.io import wavfile
from scipy.signal import sosfiltfilt, butter

winlen=4001
fs=96000
id_nyquist=int(winlen/2)
farr=np.arange(1,id_nyquist+1)*fs/winlen/1000

n400=16
n800=32
n200=7
ncut=31*2
  
fs, mono1 = wavfile.read('sounds/pure400.wav')
fs, mono2 = wavfile.read('sounds/pure800.wav')
fs, mono3 = wavfile.read('sounds/pure200.wav')

target=1.
mono1=1.*mono1[0:int(1*fs)]
mono2=1.*mono2[0:int(1*fs)]
mono3=1.*mono3[0:int(1*fs)]
mono1=mono1/np.sqrt(np.nanmean(mono1**2))*target
mono2=mono2/np.sqrt(np.nanmean(mono2**2))*target
mono3=mono3/np.sqrt(np.nanmean(mono3**2))*target
    
sos=butter(1,500/fs*2,output='sos')    
mono1=sosfiltfilt(sos,mono1)
mono2=sosfiltfilt(sos,mono2)
mono3=sosfiltfilt(sos,mono3)


ITDmax=.3
itd_list=np.arange(0,ITDmax,.01)
itd_test=np.arange(-2.5,2.5+.05,.05)
i0=np.argmin(np.abs(itd_test))
 
Ncells=len(itd_list)
bins0=np.arange(len(itd_list)+1)


neuron_sample=np.arange(2,len(itd_list),5)

save_flag=True

if save_flag==True:
    sample_rates1=np.zeros((len(itd_test),len(itd_list),ncut))
    sample_rates2=np.zeros((len(itd_test),len(itd_list),ncut))
    sample_rates3=np.zeros((len(itd_test),len(itd_list),ncut))

    for mdt,dts in enumerate(itd_test):
        #key_str=str(mdt)
 

        stereo1=itdtools.make_mixture([mono1], -1*np.array([dts])*1e-3, fs)
        stereo2=itdtools.make_mixture([mono2], -1*np.array([dts])*1e-3, fs)
        stereo3=itdtools.make_mixture([mono3], -1*np.array([dts])*1e-3, fs)
        patterns1=[]
        patterns2=[]
        patterns3=[]
        for nneuron, dt_neuron in enumerate(itd_list):
                
            _,_,_,pattern1=IC.ICfun(1.*stereo1,fs, dt_neuron,_flow=-1)
            _,_,_,pattern2=IC.ICfun(1.*stereo2,fs, dt_neuron,_flow=-1)
            _,_,_,pattern3=IC.ICfun(1.*stereo3,fs, dt_neuron,_flow=-1)
                
            patterns1.append(pattern1)
            patterns2.append(pattern2)
            patterns3.append(pattern3)

        #
    
        patt=np.nanmean(np.array(patterns1),axis=1)
        patt=patt/np.nanmax(patt)
        lateralinhibition=np.ones((patt.shape[0],1))@np.nanmean(patt,axis=0).reshape(1,-1)
        patt=patt/(1+20*lateralinhibition)
        patt=patt/np.nanmax(patt)
        #
        sample_rates1[mdt,:,:]=patt[:,:ncut]
        #
        #
        patt=np.nanmean(np.array(patterns2),axis=1)
        patt=patt/np.nanmax(patt)
        lateralinhibition=np.ones((patt.shape[0],1))@np.nanmean(patt,axis=0).reshape(1,-1)
        patt=patt/(1+20*lateralinhibition)
        patt=patt/np.nanmax(patt)
        #
        sample_rates2[mdt,:,:]=patt[:,:ncut]
        #
        patt=np.nanmean(np.array(patterns3),axis=1)
        patt=patt/np.nanmax(patt)
        lateralinhibition=np.ones((patt.shape[0],1))@np.nanmean(patt,axis=0).reshape(1,-1)
        patt=patt/(1+20*lateralinhibition)
        patt=patt/np.nanmax(patt)
        #
        sample_rates3[mdt,:,:]=patt[:,:ncut]

    
        
    sv_str={'sr1':sample_rates1,
            'sr2':sample_rates2,
            'sr3':sample_rates3}


    with open('ff5_3.pkl','wb') as fd:
        pickle.dump(sv_str,fd)

else:
    with open('ff5_3.pkl','rb') as fd:
        sv_str=pickle.load(fd)

    sample_rates1=sv_str['sr1']
    sample_rates2=sv_str['sr2']
    sample_rates3=sv_str['sr3']





def bestITDs(mat,period):
    global itd_test

    phase=[]
    for nc in range(mat.shape[1]):
        tuning_curve=mat[:,nc]
        z=np.sum(tuning_curve*np.exp(1j*2*np.pi/period*itd_test))
        phase.append(np.angle(z))

    return np.array(phase)/2./np.pi*period
        
fig=plt.figure(figsize=(10,6))

ax=fig.add_subplot(2,3,2)
ax.plot(itd_test,sample_rates1[:,neuron_sample,n400])
ax.set_title('BF = 400 Hz', fontsize=10)
ax.set_xlabel('ITD (ms)')
ax.set_xlim([-1,1])
ax.set_ylim([.3,1.])
ax=fig.add_subplot(2,3,3)
ax.plot(itd_test,sample_rates2[:,neuron_sample,n800])
ax.set_title('BF = 800 Hz', fontsize=10)
ax.set_xlabel('ITD (ms)')
ax.set_xlim([-1,1])
ax.set_ylim([.0,.85])
ax=fig.add_subplot(2,3,1)
ax.plot(itd_test,sample_rates3[:,neuron_sample,n200])
ax.set_title('BF = 200 Hz', fontsize=10)
ax.set_xlabel('ITD (ms)')
ax.set_ylabel('Response (a.u.)')
ax.set_xlim([-1,1])
ax.set_ylim([.6,.85])

ax=fig.add_subplot(2,2,3)
idmid=50
idend=101
bi_1=bestITDs(sample_rates1[:,:,n400],1./.4)
bi_2=bestITDs(sample_rates2[:,:,n800],1./.8)
bi_3=bestITDs(sample_rates3[:,:,n200],1./.2)

[h3,b]=np.histogram(bi_3,np.arange(-.5,1.6,.15))
[h1,b]=np.histogram(bi_1,b)
[h2,b]=np.histogram(bi_2,b)
bm = (b[:-1]+b[1:])/2
db=b[1]-b[0]
ax.plot(bm,h3/np.sum(h3)/db, label='200 Hz', color=colr[0])
ax.plot(bm,h1/np.sum(h1)/db, label='400 Hz', color=colr[1])
ax.plot(bm,h2/np.sum(h2)/db, label='800 Hz', color=colr[2])
ax.legend()
ax.set_xlabel('Best ITD (ms)')
ax.set_ylabel('PDF')

m400=np.sum(bm*h1)/np.sum(h1)
s400u=bm[np.min(np.where(np.cumsum(h1)/np.sum(h1)>.9)[0])]
s400l=bm[np.max(np.where(np.cumsum(h1)/np.sum(h1)<.1)[0])]

m800=np.sum(bm*h2)/np.sum(h2)
s800u=bm[np.min(np.where(np.cumsum(h2)/np.sum(h2)>.9)[0])]
s800l=bm[np.max(np.where(np.cumsum(h2)/np.sum(h2)<.1)[0])]

m200=np.sum(bm*h3)/np.sum(h3)
s200u=bm[np.min(np.where(np.cumsum(h3)/np.sum(h3)>.9)[0])]
s200l=bm[np.max(np.where(np.cumsum(h3)/np.sum(h3)<.1)[0])]

ax=fig.add_subplot(2,2,4)
ma=np.array([m200,m400,m800])
su=np.array([s200u,s400u,s800u])
sl=np.array([s200l,s400l,s800l])
ax.plot(np.array([200,400,800]),sl, 'k--')
ax.plot(np.array([200,400,800]),ma, 'k')
ax.plot(np.array([200,400,800]),su, 'k--')
ax.plot(np.arange(200,800,2),1000/8/np.arange(200,800,2), color='grey')
ax.set_xlim([100,900])
ax.set_xlabel('Best frequency (Hz)')
ax.set_ylabel('Best ITD (ms)')
fig.subplots_adjust(wspace=.45, hspace=.5)
#fig.savefig('fig5.pdf')
plt.show(block=0)


