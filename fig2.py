import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator as interpolate
from scipy.interpolate import interp1d as interpolate1
from importlib import reload
#
from optpars import projection


psim=.125*2*np.pi
psil=0*2*np.pi
psiC=np.array([0.7,0.7,0.3,0.3])*2*np.pi

itds=np.arange(0,2.51,0.05)
aarr=np.arange(-1,1.1,.25)
colors_a = plt.cm.viridis(np.linspace(0,1,len(aarr)))
freqarr=[400,800,400,800]
fig=plt.figure(figsize=(8,5))
width=0.2
dw=0.1
height=0.2
dh=0.15
bottom=0.7
left=0.125

for nf,f in enumerate(freqarr):
    ax=fig.add_axes([left, bottom, width, height])

    for na, a in enumerate(aarr):
        tc=[projection(np.array([a]),psim,psil,psiC[nf],2*np.pi*f*dt*1e-3) for dt in itds]
        lastax=ax.plot(itds,tc/np.max(tc), color=colors_a[na])

        
    ax.set_xlabel('ITD (ms)')
    if nf==0:
        ax.set_ylabel('$\psi_c=0.7 (cyc)$\n'+'Rel. rate')
    if nf==2:
        ax.set_ylabel('$\psi_c=0.3 (cyc)$\n'+'Rel. rate')
    if nf<2:
        ax.set_title('BF = ' + str(f) + 'Hz', fontsize=10)
    ax.set_ylim([0, 1.1])
                  
    if (nf==1)+(nf==3):
        bottom=bottom-height-dh
        left=left-dw-width
    else:
        left=left+dw+width


ax0=fig.add_axes([left, bottom+height-.025, width, .015])
ax0.imshow(aarr.reshape(1,-1),  extent=[aarr[0],aarr[-1],0,1 ], cmap='viridis', aspect=width/2)
ax0.invert_yaxis()
ax0.set_xlabel('a')
ax0.set_yticks([])


bottom=bottom+2*(dh+height)
left=left+2*(dw+width)+dw/2
ax=fig.add_axes([left, bottom, width, height])
#itds=np.arange(0,0.66,0.025)
freqarr=np.arange(200,1550,50)
aarr=np.arange(-5.,5.05,.05)
bestITD=np.zeros((len(freqarr),len(aarr)))
for na, a in enumerate(aarr):
    for nf,f in enumerate(freqarr):
        itds_cyc=itds[itds<1000./f]
        tc=[projection(np.array([a]),psim,psil,0.3*2*np.pi,2*np.pi*f*dt*1e-3) for dt in itds_cyc]
        idbest=np.argmax(tc)
        bestITD[nf,na]=itds[idbest]*f/1000

ax.imshow(bestITD, extent=[aarr[0],aarr[-1] , freqarr[-1], freqarr[0] ], cmap='coolwarm', aspect=(aarr[-1]-aarr[0])/(freqarr[-1]-freqarr[0]))
ax.invert_yaxis()
ax.set_xlabel('a')
ax.set_ylabel('BF (Hz)')
ax.set_title('$\psi_c$ = 0.3 cyc', fontsize=10)

bottom=bottom-height-dh
ax=fig.add_axes([left, bottom-dh*1.5, width, height+dh*1.5])
PCarr=np.arange(0,1,.025)*2*np.pi
bestIPD=np.zeros((len(freqarr),len(PCarr)))
for npc, PC in enumerate(PCarr):
    for nf,f in enumerate(freqarr):
        itds_cyc=itds[itds<1000./f]
        tc=[projection(np.array([0.5]),psim,psil,PC,2*np.pi*f*dt*1e-3) for dt in itds_cyc]
        idbest=np.argmax(tc)
        bestIPD[nf,npc]=itds[idbest]*f/1000

axh=ax.imshow(bestIPD, extent=[PCarr[0],PCarr[-1] , freqarr[-1], freqarr[0] ], cmap='coolwarm', aspect=(PCarr[-1]-PCarr[0])/(freqarr[-1]-freqarr[0]))
ax.invert_yaxis()
ax.set_xticks([0,np.pi,np.pi*2],['0','$\pi$','$2\pi$'])
ax.set_xlabel('$\psi_c$')
ax.set_ylabel('BF (Hz)')
ax.set_title('a = 0.5', fontsize=10)

fig.colorbar(axh, ax=ax, location='bottom', pad=dh*2.5, label='Best IPD (cyc)')
#plt.savefig('fig2.pdf')

plt.show(block=0)
