import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as mcmaps
import itdtools
import scipy.signal as scisig
from importlib import reload
import pickle
import matplotlib.cm as pltcol

colr=[pltcol.get_cmap('winter')(f) for f in [0, 120, 170, 210, 255]]

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

ITDmax=.3
itd_list=np.arange(0,ITDmax,.01)
itd_test=np.arange(-3,3.5+.05,.05)
i0=np.argmin(np.abs(itd_test))
 

nchan={}
bfstr=['200','300','400','600','800']
nchan['200']=7
nchan['300']=12
nchan['400']=16
nchan['600']=24
nchan['800']=32

ncut=31*2# cutoff frequency for plotting

mono={}
sample_rates={}
target=1.
sos=butter(1,500/fs*2,output='sos')    
for bf in bfstr:
    fs, mono[bf] = wavfile.read('sounds/pure'+bf+'.wav')
    mono[bf]=1.*mono[bf][0:int(1*fs)]
    mono[bf]=mono[bf]/np.sqrt(np.nanmean(mono[bf]**2))*target
    mono[bf]=sosfiltfilt(sos,mono[bf])
    sample_rates[bf]=np.zeros((len(itd_test),len(itd_list),ncut))



Ncells=len(itd_list)


neuron_sample=np.arange(3,len(itd_list)-3,5)
oicol=mcmaps['coolwarm'](np.linspace(0, 1, len(itd_list)))

save_flag=False

if save_flag==True:

        

    for mdt,dts in enumerate(itd_test):
        #key_str=str(mdt)
        stereo={}
        for bf in bfstr:
            stereo[bf]=itdtools.make_mixture([mono[bf]],
                                             -1*np.array([dts])*1e-3, fs)
            patterns=[]
            
            for nneuron, dt_neuron in enumerate(itd_list):
                
                _,_,_,patterntmp=IC.ICfun(1.*stereo[bf],fs, dt_neuron,_flow=-1)
                patterns.append(patterntmp)


            #
            patt=np.nanmean(np.array(patterns),axis=1)
            patt=patt/np.nanmax(patt)
            lateralinhibition=np.ones((patt.shape[0],1))@np.nanmean(patt,axis=0).reshape(1,-1)
            patt=patt/(1+20*lateralinhibition)
            patt=patt/np.nanmax(patt)
            #
            sample_rates[bf][mdt,:,:]=patt[:,:ncut] 


    with open('pkl/ff5_3.pkl','wb') as fd:
        pickle.dump(sample_rates,fd)

else:
    with open('pkl/ff5_3.pkl','rb') as fd:
        sample_rates=pickle.load(fd)






def bestITDs(mat,period):
    global itd_test

    itdrange=np.max(itd_test)-np.min(itd_test)
    itdmax=np.min(itd_test)+period
    idmax=max(np.where(itd_test<itdmax)[0])
    phase=[]
    
    for nc in range(mat.shape[1]):
        tuning_curve=mat[:idmax,nc]
        z=np.sum(tuning_curve*np.exp(1j*2*np.pi/period*itd_test[:idmax]))
        phase.append(np.angle(z))

    return np.array(phase)/2./np.pi*period
        
fig=plt.figure(figsize=(10,6))

npanel=1
for bf in ['200', '300', '800']:
    ax=fig.add_subplot(2,3,npanel)
    maxrate=np.max(sample_rates[bf][:,neuron_sample,nchan[bf]])
    minrate=np.mean(sample_rates[bf][50:57,:,nchan[bf]])/1.5#0#np.min(sample_rates[bf][30,neuron_sample,nchan[bf]])
    for nid in neuron_sample:
        ax.plot(itd_test,
                (sample_rates[bf][:,nid,nchan[bf]]-minrate)/(maxrate-minrate),
                label=str(itd_list[nid]),
                color=oicol[nid])
    
    ax.set_title('BF = '+bf+' Hz', fontsize=10)
    ax.set_xlabel('ITD (ms)')
    ax.set_xlim([-2.,2.])
    ax.set_ylim([0,1.1])
    if npanel==1:
        ax.legend(frameon=False)
        ax.set_ylabel('Norm Rate (a.u.)')
    npanel=npanel+1
    




ax=fig.add_subplot(2,2,3)
idmid=50
idend=101
hi={}
for bf in bfstr:
    freq=1.*int(bf)/1000
    maxrate=np.max(sample_rates[bf][:,neuron_sample,nchan[bf]])
    minrate=np.mean(sample_rates[bf][50:57,neuron_sample,nchan[bf]])/1.5
    nltun=(sample_rates[bf][:,:,nchan[bf]]-minrate)/(maxrate-minrate)
    bi=bestITDs(nltun,1./freq)*freq
    [hi[bf],b]=np.histogram(bi,np.arange(-.2,.5,.06))
    
bm = (b[:-1]+b[1:])/2
db=b[1]-b[0]

ncol=0
for bf in bfstr:
    #freq=1.*int(bf)/1000
    ax.plot(bm*freq,hi[bf]/np.sum(hi[bf]), label=bf+' Hz', color=colr[ncol])
    ncol +=1

ax.legend(frameon=False)
ax.set_xlabel('Best IPD (cyc)')
ax.set_xlim([-.2,.4])
ax.set_ylabel('Normalized count')

ax=fig.add_subplot(2,2,4)
ma=[];su=[];sl=[];fa=[];med=[]
for bf in bfstr:
    cdf=np.cumsum(hi[bf])/np.sum(hi[bf])
    m=np.sum(bm*hi[bf])/np.sum(hi[bf])
    med.append(bm[np.min(np.where(cdf>.5)[0])])

    sui=bm[np.min(np.where(cdf>.9)[0])]
    sli=bm[np.max(np.where(cdf<.1)[0])]
    fa.append(1.*int(bf))
    ma.append(m)
    su.append(sui)
    sl.append(sli)

ma=np.array(ma)
su=np.array(su)
sl=np.array(sl)
fa=np.array(fa)

ax.plot(fa,sl, 'k--')
ax.plot(fa,ma, 'k')
ax.plot(fa,su, 'k--')
ax.plot(fa,np.array(med), 'k.')
#ax.plot(np.arange(200,800,2),1000/8/np.arange(200,800,2), color='grey')
ax.plot(np.arange(200,800,2),1/8*np.ones_like(np.arange(200,800,2)), color='grey')
ax.set_xlim([100,900])
ax.set_xlabel('Best frequency (Hz)')
ax.set_ylabel('Best IPD (cyc)')
fig.subplots_adjust(wspace=.45, hspace=.5)
#fig.savefig('fig5.pdf')
plt.show(block=0)


