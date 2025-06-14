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
    
save_flag=True
tval=.05
savestruct={}
fs, mono1 = wavfile.read('sounds/pure400.wav')
fs, mono2 = wavfile.read('sounds/noise.wav')

mono1=1.*mono1[0:int(.3*fs)]
mono2=1.*mono2[0:int(.3*fs)]
Jlat=20.


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
db_sig=np.array([0,-5,-10,-15,-20,-25,-200])
lmax=int(len(db_sig)-1)
level_sig=10**(db_sig/20)

ITDmax=.7
itd_list=np.arange(0,ITDmax,.01)

Ncells=len(itd_list)
bins0=np.arange(len(itd_list)+1)

 

if save_flag==True:
    
    for nl,level in enumerate(level_sig):
        leveldb=db_sig[nl]
        for mdt,dts in enumerate(itd_sig):
            key_str=str(nl)+' '+str(mdt)
            print(key_str)
            
            stereo=np.array([level*mono1+mono2,dts*level*mono1+mono2])
            
            patterns=[]
            jpatterns=[]
            for nneuron, dt_neuron in enumerate(itd_list):
            
                ic,_,_,pattern=IC.ICfun(1.*stereo,fs, dt_neuron,_flow=-1)
                jeff=IC.Jeffress(1.*stereo, int(dt_neuron*fs*1e-3),fs,_flow=-1)
                jpatt=IC.stspec(jeff)             
                patterns.append(pattern)
                jpatterns.append(jpatt)

            savestruct[key_str]={'jpatterns':jpatterns,   
                                 'patterns':patterns}

    with open('pkl/ff4.pkl','wb') as fd:
        pickle.dump(savestruct,fd)
else:
    with open('pkl/ff4.pkl','rb') as fd:
        savestruct=pickle.load(fd)


def addinhibition(patterns):
    patt=np.mean(np.array(patterns),axis=1)
    patt=patt/np.max(patt)
    lateralinhibition=np.ones((patt.shape[0],1))@np.mean(patt,axis=0).reshape(1,-1)
    lpatt=patt/(1+Jlat*lateralinhibition)

    return lpatt, patt


def dissim(x,y):

    return np.nanmax((x-y)/y,axis=0)#/np.mean(y,axis=0)
    #return np.diff(np.max(x,axis=0)/np.max(y))
    
noise=savestruct[str(lmax)+' 0']['patterns']
jnoise=savestruct[str(lmax)+' 0']['jpatterns']
noise, nnoise=addinhibition(noise)
jnoise, njnoise=addinhibition(jnoise)


fig=plt.figure(figsize=(10,6))
figNo=plt.figure(figsize=(10,6))
nplt=1
allout={'snr':[],'jsnr':[],'n_snr':[], 'n_jsnr':[],'pks':[],'pksNo':[], 'id':[], 'th':[]}
for nl,level in enumerate(level_sig[:-1]):
    leveldb=db_sig[nl]
    for mdt,dts in enumerate(itd_sig):
        key_str=str(nl)+' '+str(mdt)
           
        patterns=savestruct[key_str]['patterns']
        jpatterns=savestruct[key_str]['jpatterns']          

        #
        patt, npatt=addinhibition(patterns)
        jpatt, njpatt=addinhibition(jpatterns)
        
        snr=dissim(patt[:,:ncut],noise[:,:ncut])
        jsnr=dissim(jpatt[:,:ncut],jnoise[:,:ncut])
        n_snr=dissim(npatt[:,:ncut],nnoise[:,:ncut])
        n_jsnr=dissim(njpatt[:,:ncut],njnoise[:,:ncut])
        pks,th=find_peaks(snr, prominence=tval, wlen=20)
        pksNo,thNo=find_peaks(n_snr, prominence=tval, wlen=20)

        allout['id'].append((leveldb,mdt))
        allout['th'].append((th,thNo))
        allout['snr'].append(snr)
        allout['jsnr'].append(jsnr)
        allout['n_snr'].append(n_snr)
        allout['n_jsnr'].append(n_jsnr)
        allout['pks'].append(pks)
        allout['pksNo'].append(pksNo)
        
        if nl in [0,1,2,4]:
            ax=fig.add_subplot(4,4,nplt)
            axNo=figNo.add_subplot(4,4,nplt)
            patt_mm=patt/np.max(patt)
            ax.imshow(patt_mm[:,:ncut].T, extent=[itd_list[0],itd_list[-1] , farr[ncut], farr[0] ], cmap='inferno', vmin=0, vmax=1, aspect=ITDmax/(farr[ncut]-farr[0]))
            patt_mm=npatt/np.max(npatt)
            axNo.imshow(patt_mm[:,:ncut].T, extent=[itd_list[0],itd_list[-1] , farr[ncut], farr[0] ], cmap='inferno', vmin=0, vmax=1, aspect=ITDmax/(farr[ncut]-farr[0]))
            ax.invert_yaxis()
            axNo.invert_yaxis()
            ax.set_xticks([0, 0.3, .6])
            axNo.set_xticks([0, 0.3, .6])
            ax.set_yticks([0.4,.8])
            axNo.set_yticks([0.4,.8])

            if nplt==1:
                ax.text(-.75,0.5,str(leveldb), rotation=90)
                axNo.text(-.75,0.5,str(leveldb), rotation=90)
                ax.text(-0.9,1.4,'SNR (dB)')
                axNo.text(-0.9,1.4,'SNR (dB)')
            if nplt==5:
                ax.text(-.75,0.5,str(leveldb), rotation=90)
                axNo.text(-.75,0.5,str(leveldb), rotation=90)
            if nplt==9:
                ax.text(-.75,0.5,str(leveldb), rotation=90)
                axNo.text(-.75,0.5,str(leveldb), rotation=90)
            if nplt==13:
                ax.text(-.75,0.5,str(leveldb), rotation=90)
                axNo.text(-.75,0.5,str(leveldb), rotation=90)
            if nplt>12:
                ax.set_xlabel('Target ITD (ms)')
                axNo.set_xlabel('Target ITD (ms)')
            if nplt==13:
                ax.set_ylabel('BF (kHz)')
                axNo.set_ylabel('BF (kHz)')
            
            ax=fig.add_subplot(4,4,nplt+1)
            axNo=figNo.add_subplot(4,4,nplt+1)
            #

            ax.plot(farr[0:ncut], (jsnr), '-', color='grey', label='Delay')
            print(th)
            ax.plot(farr[0:ncut], (snr), '--k', label='IC')
            ax.plot(farr[pks], snr[pks], '.r')

            axNo.plot(farr[0:ncut], (n_jsnr), '-', color='grey',label='Delay')
            print(thNo)
            axNo.plot(farr[0:ncut], (n_snr), '--k', label='IC')
            axNo.plot(farr[pksNo], n_snr[pksNo], '.r')
            #
            ax.set_xlim([0,1])
            ax.set_xticks([0, 0.4, .8])
            ax.set_ylim([-.5,2.5])       
            axNo.set_xlim([0,1])
            axNo.set_xticks([0, 0.4, .8])
            axNo.set_ylim([-1.5,5.5])       

            if nplt==1:
                ax.set_title('S$_0$N$_0$')
                axNo.set_title('S$_0$N$_0$')
                ax.legend(frameon=False)
                axNo.legend(frameon=False)
            if nplt==3:
                ax.set_title('S$_{\pi}$N$_0$')
                axNo.set_title('S$_{\pi}$N$_0$')
            if nplt>12:
                ax.set_xlabel('BF (kHz)')
                axNo.set_xlabel('BF (kHz)')
            if nplt==13:
                ax.set_ylabel('d$^{max}$')
                axNo.set_ylabel('d$^{max}$')
          
            nplt +=2

        
fig.subplots_adjust(wspace=.45, hspace=.5)
figNo.subplots_adjust(wspace=.45, hspace=.5)
#fig.savefig('fig7.pdf')
#figNo.savefig('fig8.pdf')
plt.show(block=0)


