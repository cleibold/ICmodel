import numpy as np
import matplotlib.pyplot as plt
import pickle

from importlib import reload
from phasewarp import phasewarp
from scipy.signal import sosfiltfilt, butter
from matplotlib import colormaps
import IC
reload(IC)

fs=96000
sos=butter(1,500/fs*2,output='sos')

winlen=4001
id_nyquist=int(winlen/2)
farr=np.arange(1,id_nyquist+1)*fs/winlen/1000
ncut=31*2
  
ITDmax=.7
itd_list=np.arange(0,ITDmax,.01)

Ncells=len(itd_list)
bins0=np.arange(len(itd_list)+1)

pshifts=2**np.arange(3,11)
save_flag=True
if save_flag==True:
    patterns={}
    iclist={}
    for pshift in pshifts:
        stereo=np.array(phasewarp(fs,1,pshift))
        stereo=sosfiltfilt(sos,stereo)

        patterns[pshift]=[]
        iclist[pshift]=[]
        for nneuron, dt_neuron in enumerate(itd_list):
                
            ic,_,_,pattern=IC.ICfun(1.*stereo,fs, dt_neuron,_flow=-1)  
    
            patterns[pshift].append(pattern)
            iclist[pshift].append(ic**2)

        patterns[pshift]=np.array(patterns[pshift])
        iclist[pshift]=np.array(iclist[pshift])

    #with open('fwarp.pkl','wb') as fd:
    #    pickle.dump(patterns,fd)
else:
    with open('fwarp.pkl','rb') as fd:
        patterns=pickle.load(fd)


Navg=2001
colphase=colormaps['rainbow']((np.arange(9))/8)
fig=plt.figure()
ax=fig.add_subplot(2,3,1)

patt=iclist[8]
patt=patt/np.max(patt)
lateralinhibition=np.ones((patt.shape[0],1))@np.mean(patt,axis=0).reshape(1,-1)
patt=patt/(1+20*lateralinhibition)
patt=patt/np.max(patt)

for row in range(patt.shape[0]):    
    patt[row]=np.convolve(patt[row],np.ones(Navg)/Navg, 'same')

patt=patt/np.max(patt)    
dur=patt.shape[1]/fs

ax.imshow(patt, extent=[0, dur, itd_list[-1], itd_list[0] ], cmap='inferno', vmin=0, vmax=1, aspect=dur/ITDmax)
ax.invert_yaxis()
ax.set_xlim([0, dur])
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.3, .6])
ax.set_ylabel('ITD (ms)')
ax.set_xlabel('Time s')
ax.set_title('Phase warp 8 Hz', fontsize=10)


ax=fig.add_subplot(2,3,2)
patt=np.mean(np.array(patterns[8]),axis=1)
patt=patt/np.max(patt)
lateralinhibition=np.ones((patt.shape[0],1))@np.mean(patt,axis=0).reshape(1,-1)
patt=patt/(1+20*lateralinhibition)
patt=patt/np.max(patt)


ax.imshow(patt[:,:ncut].T, extent=[itd_list[0],itd_list[-1] , farr[ncut], farr[0] ], cmap='inferno', vmin=0, vmax=1, aspect=ITDmax/(farr[ncut]-farr[0]))
ax.invert_yaxis()
ax.set_xlim([0, ITDmax])
ax.set_xticks([0, 0.3, .6])
#ax.set_ylim([0,1.5])
ax.set_yticks([0.5,1.0,1.5])
ax.set_xlabel('ITD (ms)')
ax.set_ylabel('BF (kHz)')

ax=fig.add_subplot(2,3,3)
for m,pkey in enumerate(patterns.keys()):
    tsig=iclist[pkey]
    tax=np.arange(tsig.shape[1])/fs
    phasemat=np.ones((tsig.shape[0],1))@np.exp(2*np.pi*1j*float(pkey)*tax).reshape(1,-1)
    
    vs=np.abs(np.sum(tsig*phasemat, axis=1))/np.sum(tsig,axis=1)
    print(pkey,np.mean(vs))
    ax.plot(itd_list,vs,'-',color=colphase[1+m])

ax.set_xticks([0,.3,.6])
ax.set_xlabel('ITD (ms)')
ax.set_ylabel('VS')
ax.set_box_aspect(1)

fig.subplots_adjust(wspace=.5)
#fig.savefig('figwarp.pdf')

plt.show()
