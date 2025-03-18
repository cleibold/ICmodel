import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import CloughTocher2DInterpolator as interpolate
from importlib import reload

limmin=-np.pi
limmax=np.pi

farr=np.arange(200,1550,50)
import pickle

ITDmax_=np.array([0.7, 0.3, 0.15])
a_fun, pc_fun = {}, {}

left=0.15
bottom=0.7
width=0.17
height=0.2
dw=0.13
dy=0.07

fig=plt.figure()
nr=0
for ITDmax in ITDmax_:
    idstr=str(int(ITDmax*10))
    with open('interpol_a_'+idstr+'_.pkl', 'rb') as f_d:
        a_fun[idstr]=pickle.load(f_d)
        
    with open('interpol_pc_'+idstr+'_.pkl', 'rb') as f_d:
        pc_fun[idstr]=pickle.load(f_d)
    

    

    #

    itd_range=np.arange(0,ITDmax,.025)
    irange=itd_range[-1]-itd_range[0]
    #
    pilimit=np.append(np.array(np.nan),(np.pi/itd_range[1:])/(2*np.pi)*1e3)
    #
    pilimit[pilimit>farr[-1]]=np.nan
    pilimit[pilimit<farr[0]]=np.nan

    amat=np.zeros((len(farr),len(itd_range)))
    PCmat=np.zeros((len(farr),len(itd_range)))

    for nf, f in enumerate(farr):
        print(f)
        Tperiod=1./f*1e3#ms
    
        for nl,dt in enumerate(itd_range):

            amat[nf,nl]=a_fun[idstr](f*1e-3,dt)
            pc=pc_fun[idstr](f*1e-3)        
            PCmat[nf,nl]=np.mod(pc+np.pi,2*np.pi)-np.pi




    ax=fig.add_axes([left, bottom, width, height])

    axh1=ax.imshow(np.flip(amat,axis=0), extent=[itd_range[0],itd_range[-1] , farr[-1], farr[0] ], cmap='PiYG', vmin=-1, vmax=1, aspect=irange/(farr[-1]-farr[0]),origin='lower')
    ax.invert_yaxis()
    
    ax.plot(itd_range,pilimit,'-k')
    if nr==2:
        ax.set_xlabel('ITD (ms)')
    ax.set_ylabel('BF (Hz)')
    if nr==0:
        ax.set_title('$a_{opt}$', fontsize=10)
    fig.colorbar(axh1, ax=ax, location='right', pad=.05, shrink=0.7)


    
    left=left+width+dw
    pmean=np.round(100*np.mean(PCmat[:,0]/2/np.pi))/100
    ax=fig.add_axes([left, bottom+.01, width-.02, height-.02])
    ax.plot(farr/1000,PCmat[:,0]/2/np.pi, '-', color='blue')
    ax.text(2,0, 'Max ITD = '+str(ITDmax)+' ms',rotation=90,fontsize=8)
    ax.text(.15,.85, 'mean $\psi_c$ = '+str(pmean),fontsize=8)
    ax.set_ylim([0,1])

    
    if nr==2:
        ax.set_xlabel('BF (kHz)')
    if nr==0:
        ax.set_title('$\psi_{c}$ (cyc)', fontsize=10)
        
    left=left-(width+dw)
    bottom=bottom-height-dy
    nr += 1


#plt.savefig('fig3.pdf')

plt.show(block=1)
