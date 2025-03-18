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
IC.cMSO,IC.cLSO= 5/IC.tauadapt/fs, 10/IC.tauadapt/fs
id_nyquist=int(winlen/2)
farr=np.arange(1,id_nyquist+1)*fs/winlen/1000
ncut=62
Navg=2001# time average


fs, mono1 = wavfile.read('sounds/pure400.wav')
fs, mono2 = wavfile.read('sounds/female1.wav')
    
mono1=1.*mono1[0:(96000*2)]
mono2=1.*mono2[0:(96000*2)]
target=1
mono2=mono2/np.sqrt(np.nanmean((mono2)**2))*target
mono1=mono1/np.sqrt(np.nanmean((mono1)**2))*target

sos=butter(1,500/fs*2,output='sos')
mono1=sosfiltfilt(sos,mono1)
mono2=sosfiltfilt(sos,mono2)

save_flag=True
if save_flag==False:
    with open('pkl/f11.pkl','rb') as fd:
        savestruct=pickle.load(fd)


else:
    savestruct={}





#
itd_sig=[.1,.6]

ITDmax=.7
itd_list=np.arange(0,ITDmax,.01)

Ncells=len(itd_list)
bins0=np.arange(len(itd_list)+1)

fig=plt.figure()
for n_itd, itd in enumerate(itd_sig):
    stereo=itdtools.make_mixture([mono1], -1*np.array([itd])*1e-3, fs)
    iclist=[]
    for nneuron, dt_neuron in enumerate(itd_list):
    
        ic,mso,lso,pattern=IC.ICfun(1.*stereo,fs, dt_neuron,_flow=-1)
        #print(dt_neuron, np.nanvar(IC.aMSO))
    
        iclist.append(ic**2)

    #
    
    ax=fig.add_subplot(2,3,1+3*n_itd)
    
    tax=np.arange(len(mso))/fs
    randsamp=np.sort(np.random.permutation(len(tax))[:int(2*fs/60)])
    ax.scatter(mso[randsamp],lso[randsamp],c=tax[randsamp], cmap='viridis', alpha=0.5)
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('MSO (a.u.)')
    ax.set_ylabel('LSO (a.u.)')

    patt= np.array(iclist)
    patt=patt/np.max(patt)
    lateralinhibition=np.ones((patt.shape[0],1))@np.mean(patt,axis=0).reshape(1,-1)
    patt=patt/(1+20*lateralinhibition)
    patt=patt/np.max(patt)


    for row in range(patt.shape[0]):    
        patt[row]=np.convolve(patt[row],np.ones(Navg)/Navg, 'same')

    #patt=patt-np.min(patt)
    patt=patt/np.max(patt)    
    dur=patt.shape[1]/fs
    tax=np.arange(patt.shape[1])/fs

    ax=fig.add_subplot(2,3,n_itd*3+2)
    ax.imshow(patt, extent=[0, dur, itd_list[-1], itd_list[0] ], cmap='inferno', vmin=0, vmax=1, aspect=dur/ITDmax)
    bestITD=itd_list[np.argmax(patt,axis=0)]
    bestITD[np.max(patt,axis=0)<0.05]=np.nan
    ax.plot(tax,bestITD, '-w')
    ax.invert_yaxis()
    ax.set_xlim([0, dur])
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 0.3, .6])
    ax.set_ylabel('ITD (ms)')
    ax.set_xlabel('Time s')

    
    stereo=itdtools.make_mixture([mono2], -1*np.array([itd])*1e-3, fs)
    iclist=[]
    for nneuron, dt_neuron in enumerate(itd_list):
    
        ic,mso,lso,pattern=IC.ICfun(1.*stereo,fs, dt_neuron,_flow=-1)

    
        iclist.append(ic**2)

    #
    patt= np.array(iclist)
    patt=patt/np.max(patt)
    lateralinhibition=np.ones((patt.shape[0],1))@np.mean(patt,axis=0).reshape(1,-1)
    patt=patt/(1+20*lateralinhibition)
    patt=patt/np.max(patt)


    for row in range(patt.shape[0]):    
        patt[row]=np.convolve(patt[row],np.ones(Navg)/Navg, 'same')

    #patt=patt-np.min(patt)
    patt=patt/np.max(patt)    
    dur=patt.shape[1]/fs
    tax=np.arange(patt.shape[1])/fs

    ax=fig.add_subplot(2,3,n_itd*3+3)
    ax.imshow(patt, extent=[0, dur, itd_list[-1], itd_list[0] ], cmap='inferno', vmin=0, vmax=1, aspect=dur/ITDmax)
    bestITD=itd_list[np.argmax(patt,axis=0)]
    bestITD[np.max(patt,axis=0)<0.05]=np.nan
    ax.plot(tax,bestITD, '-w')
    ax.invert_yaxis()
    ax.set_xlim([0, dur])
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 0.3, .6])
    ax.set_ylabel('ITD (ms)')
    ax.set_xlabel('Time s')


fig.subplots_adjust(wspace=0.5)
#fig.savefig('fig11.pdf')
plt.show()


if save_flag==True:
    with open('pkl/f11.pkl','wb') as fd:
        pickle.dump(savestruct,fd)
