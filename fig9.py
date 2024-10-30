import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
import itdtools
import scipy.signal as scisig
from importlib import reload
import pickle
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
ncut=62
    
save_flag=True
if save_flag==False:
    with open('f9.pkl','rb') as fd:
        savestruct=pickle.load(fd)


else:
    savestruct={}
    fs, mono1 = wavfile.read('sounds/female1.wav')
    fs, mono2 = wavfile.read('sounds/male2.wav')

    mono1=1.*mono1#[0:196000]
    mono2=1.*mono2#[0:196000]
    target=1#np.sqrt(np.nanmean(mono1**2))
    mono1=mono1/np.sqrt(np.nanmean(mono1**2))*target
    mono2=mono2/np.sqrt(np.nanmean(mono2**2))*target
    
    sos=butter(1,500/fs*2,output='sos')
    mono1=sosfiltfilt(sos,mono1)
    mono2=sosfiltfilt(sos,mono2)




#
itd_noise=[0.,0.4]
itd_sig=[.4,0]
db_sig=np.array([0,-3,-6,-9])
level_sig=10**(db_sig/20)

ITDmax=.7
itd_list=np.arange(0,ITDmax,.01)

Ncells=len(itd_list)
bins0=np.arange(len(itd_list)+1)

 
#nintegrate=int(fs*.005)

fig=plt.figure(figsize=(10,6))
nplt=1
for nl,level in enumerate(level_sig):
    leveldb=db_sig[nl]
    for ndt,dtn in enumerate(itd_noise):
        key_str=str(nl)+' '+str(ndt)
 
        ax=fig.add_subplot(4,4,nplt+1)

        if save_flag==True:
            stereo=itdtools.make_mixture([level*mono1, mono2], -1*np.array([itd_sig[ndt], dtn])*1e-3, fs)

            iclist=[]
            jefflist=[]
            rates=np.zeros((2,len(itd_list)))
            patterns=[]
            for nneuron, dt_neuron in enumerate(itd_list):
                
                ic,_,_,pattern=IC.ICfun(1.*stereo,fs, dt_neuron,_flow=-1)
                jeff=IC.Jeffress(1.*stereo, int(dt_neuron*fs*1e-3),fs,_flow=-1)
                
                rates[:,nneuron]=(np.nanvar(ic), np.nanvar(jeff))
                
                patterns.append(pattern)
                iclist.append(ic)
                jefflist.append(jeff)
                
            Ric=np.array(iclist)
            Rjeff=np.array(jefflist)
            r_ic = Ric/np.sqrt(np.mean(Ric**2))
            r_jeff = Rjeff/np.sqrt(np.mean(Rjeff**2))
        
        
            r1=[];r2=[];rj1=[];rj2=[]; poweric=[];powerjeff=[];idown=[]
            for n in range(0,Ncells,4):
                print(leveldb,dtn,n)
                idown.append(itd_list[n])
                r1.append(IC.rpearson(mono1,r_ic[n,:],fs,_flow=-1))
                r2.append(IC.rpearson(mono2,r_ic[n,:],fs,_flow=-1))
                rj1.append(IC.rpearson(mono1,r_jeff[n,:],fs,_flow=-1))
                rj2.append(IC.rpearson(mono2,r_jeff[n,:],fs,_flow=-1))
        
        

            savestruct[key_str]={'r1':r1,
                      'r2':r2,
                      'rj1':rj1,
                      'rj2':rj2,
                      'rates':rates,
                      'patterns':patterns}
                                          
        else:
            idown=[]
            for n in range(0,Ncells,4):

                idown.append(itd_list[n])
            r1=savestruct[key_str]['r1']
            r2=savestruct[key_str]['r2']
            rj1=savestruct[key_str]['rj1']
            rj2=savestruct[key_str]['rj2']
            rates=savestruct[key_str]['rates']
            patterns=savestruct[key_str]['patterns']
            
        #
        patt=np.mean(np.array(patterns),axis=1)
        patt=patt/np.max(patt)
        lateralinhibition=np.ones((patt.shape[0],1))@np.mean(patt,axis=0).reshape(1,-1)
        patt=patt/(1+20*lateralinhibition)
        patt=patt/np.max(patt)
        #
        ax.plot(idown, (r1), '-k')
        ax.plot(idown, (rj1), '+k')
        #
        ax.plot(idown, (r2), '-', color='grey')
        ax.plot(idown, (rj2), '+', color='grey')
        #
        #
        #r0=np.append(np.append(np.ones(2)*rates[0,0],rates[0,:]),np.ones(2)*rates[0,-1])
        tmp=np.mean(patt[:,:ncut],axis=1)
        r0=np.append(np.append(np.ones(2)*tmp[0],tmp),np.ones(2)*tmp[-1])
        icr=np.convolve(r0,np.ones(5)/5,'same')[2:-2]
        dr=(icr-np.min(icr))
        ax.plot(itd_list,dr/np.max(dr), color='green')
        dr=(rates[1,:]-np.min(rates[1,:]))
        ax.plot(itd_list,dr/np.max(dr), '+', color='olive', alpha=0.5)
        #
        ax.set_xlim([-.05, ITDmax])
        ax.set_xticks([0, 0.3, .6])
        ax.set_ylim([-.1,1.1])

        
      
        if nplt==1:
            ax.set_title('S:Female$_{0.4}$N:Male$_0$')
        if nplt==3:
            ax.set_title('S:Female$_{0}$N:Male$_{0.4}$')
        if nplt>12:
            ax.set_xlabel('ITD (ms)')
        if nplt==13:
            ax.set_ylabel('Correlation')
            ax2 = ax.twinx()
            ax2.set_ylabel('Rate', color='green', fontsize=8)   
            ax2.set_ylim([0, 1])
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels([0, 1], color='green')


        ax=fig.add_subplot(4,4,nplt)
        
        ax.imshow(patt[:,:ncut].T, extent=[itd_list[0],itd_list[-1] , farr[ncut], farr[0] ], cmap='inferno', vmin=0, vmax=1, aspect=ITDmax/(farr[ncut]-farr[0]))
        ax.invert_yaxis()
        ax.set_xticks([0, 0.3, .6])
        ax.set_ylim([0,1.5])
        ax.set_yticks([0.5,1.0,1.5])

        if nplt==1:
            ax.text(-.75,0.5,str(leveldb), rotation=90)
            ax.text(-0.9,1.4,'SNR (dB)')
        if nplt==5:
            ax.text(-.75,0.5,str(leveldb), rotation=90)
        if nplt==9:
            ax.text(-.75,0.5,str(leveldb), rotation=90)
        if nplt==13:
            ax.text(-.75,0.5,str(leveldb), rotation=90)
       
        if nplt>12:
            ax.set_xlabel('ITD (ms)')

        if nplt==13:
            ax.set_ylabel('BF (kHz)')
            
            
        nplt +=2

        
fig.subplots_adjust(wspace=.55, hspace=.5)
#fig.savefig('fig9.pdf')
plt.show(block=0)


if save_flag==True:
    with open('f9.pkl','wb') as fd:
        pickle.dump(savestruct,fd)
