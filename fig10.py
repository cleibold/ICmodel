import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
import itdtools
import scipy.signal as scisig
from importlib import reload
from optpars import renorm
import pickle
#
import IC
reload(IC)
#
from scipy.io import wavfile
fs, mono1 = wavfile.read('sounds/female1.wav')
fs, mono2 = wavfile.read('sounds/male2.wav')
fs, mono3 = wavfile.read('sounds/male1.wav')
mono3=mono3/2
nlen=947401
mono1=1.*mono1[0:nlen]
mono2=1.*mono2[0:nlen]
mono3=1.*mono3[0:nlen]

mono2=mono2/np.sqrt(np.nanmean(mono2**2))*np.sqrt(np.nanmean(mono1**2))
mono3=mono3/np.sqrt(np.nanmean(mono3**2))*np.sqrt(np.nanmean(mono1**2))

#
# calculate again save_flag=True
save_flag=True

#
#Target ITDs
ITDmax=.7
itd_list=np.arange(0,ITDmax,.01)
Ncells=len(itd_list)
#

nlong=int(fs*.02)#20 ms intervals
#

tax=np.arange(len(mono1))/fs
#
r1=[]
r2=[]
rj1=[]
rj2=[]

# scenes
stereo2=itdtools.make_mixture([mono1, mono2], -1*np.array([.2,.4])*1e-3, fs)
stereo3=itdtools.make_mixture([mono1, mono2, mono3], -1*np.array([.2,.4,-.2])*1e-3, fs)
#



if save_flag==True:

    sv_struct={}
    # 2 & 3 speaker scence
    for nspeaker, stereo in enumerate([stereo2, stereo3]):

        iclist=[]
        jefflist=[]
        for nneuron, dt_neuron in enumerate(itd_list):
        
            ic,_,_,_=IC.ICfun(1.*stereo2,fs, dt_neuron)
            jeff=IC.Jeffress(1.*stereo2, int(dt_neuron*fs*1e-3),fs)
  
            iclist.append(ic)
            jefflist.append(jeff)
    
        Ric=np.array(iclist)
        Rjeff=np.array(jefflist)
        r_ic = Ric/np.sqrt(np.nanmean(Ric**2))
        r_jeff = Rjeff/np.sqrt(np.nanmean(Rjeff**2))
        
        pks_ic=[];pks_jeff=[]
        nlongsnips=int(Ric.shape[1]/nlong)
        r1=np.zeros((Ncells,nlongsnips))
        rj1=np.zeros((Ncells,nlongsnips))
        r2=np.zeros((Ncells,nlongsnips))
        rj2=np.zeros((Ncells,nlongsnips))

        for tstep in range(nlongsnips):
            inp_ic=np.mean(Ric[:,tstep*nlong:(tstep+1)*nlong]**2,axis=1)
            inp_j=np.mean(Rjeff[:,tstep*nlong:(tstep+1)*nlong]**2,axis=1)
            pks_ic.append(np.argmax(inp_ic))
            pks_jeff.append(np.argmax(inp_j))


            msnip1=mono1[tstep*nlong:(tstep+1)*nlong]
            msnip2=mono2[tstep*nlong:(tstep+1)*nlong]
    
            for n in range(Ncells):
                rsnip=r_ic[n,tstep*nlong:(tstep+1)*nlong]
                jsnip=r_jeff[n,tstep*nlong:(tstep+1)*nlong]
                r1[n,tstep]=IC.rpearson(msnip1,rsnip,fs)
                r2[n,tstep]=IC.rpearson(msnip2,rsnip,fs)
                rj1[n,tstep]=IC.rpearson(msnip1,jsnip,fs)
                rj2[n,tstep]=IC.rpearson(msnip2,jsnip,fs)

            print(tstep, nlongsnips, np.mean(r1[:,tstep]), np.mean(r2[:,tstep]))
    
        # 2 & 3 speaker scene
        sv_struct[str(nspeaker+2)]={'pks_ic':pks_ic, 'pks_jeff':pks_jeff,
                                  'r1':r1, 'r2':r2,'rj1':rj1, 'rj2':rj2,
                                  'r_ic':r_ic, 'r_jeff':r_jeff}

    #save simresults
    with open('pkl/ff10.pkl','wb') as fd:
        pickle.dump(sv_struct,fd)

else:
    with open('pkl/ff10.pkl','rb') as fd:
        sv_struct=pickle.load(fd)
        


####
#plotting
####

fig=plt.figure()   
col={'':['tab:blue','tab:orange','black'], 'j':['tab:cyan','orange','tab:gray'] }
wd={'':.04,'j':.02}
lstr={'':'IC','j':'Delay line'}

for n,scene in enumerate(['2', '3']):

    ax1=fig.add_subplot(3,2,n+1)
    ax2=fig.add_subplot(3,2,n+3)
    ax3=fig.add_subplot(3,2,n+5)
    
    for model in ['','j']:
        r1=sv_struct[scene]['r'+model+'1']
        r2=sv_struct[scene]['r'+model+'2']
        if model == '':
            ml='ic'
        else:
            ml='jeff'
            
        pks=np.array(sv_struct[scene]['pks_'+ml])/r1.shape[0]*.7
                
        taxl=np.arange(r2.shape[1])/fs*nlong
        if model=='':
            ax=ax1
            ax.set_title(scene + ' speakers', fontsize=10)
        else:
            ax=ax2
            ax.set_xlabel('Time (s)')
            
        ax.plot(taxl,np.mean(r1,axis=0),'.',  color=col[model][0])
        ax.plot(taxl,np.mean(r2,axis=0),'.',  color=col[model][1])
        ax3.hist(pks,np.arange(0,.7,.025), color=col[model][2],
                 width=wd[model],
                 label=lstr[model])
        ax3.set_ylabel('#')
        ax.set_ylim([0., 1])
        ax.set_ylabel('r$_{Pearson}$')
        
    ax3.set_xlabel('Target ITD (ms)')
    if n==0:
        ax3.legend(frameon=False,bbox_to_anchor=[0.56, 1.05],fontsize=7)
    
fig.subplots_adjust(wspace=.4, hspace=.7)
plt.show(block=0) 
