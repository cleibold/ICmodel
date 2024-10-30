import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as pltcol
#
from optpars import explicit, psim, psil


#
from scipy.io import wavfile


itd_list=np.arange(0,.61,.1)
colors_itd = plt.cm.coolwarm(np.linspace(0,1,len(itd_list)))

fig=plt.figure(figsize=(5.5,6.))
axs={}
for n in range(9):
    axs[n+1]=plt.subplot(4,3,n+1)

colr=[pltcol.get_cmap('summer')(f) for f in [0,100, 200]]
nax=0
f_flag=0
for _f in [0.2, 0.4, 0.8]:
    f_flag += 1

    psi_flag=0
    for psiCOM in np.array([0.3,0.7,0.7])*2*np.pi:
        psi_flag +=1
        nax=f_flag+(psi_flag-1)*3
        
        alpha={}
        for ni, itd in enumerate(itd_list):
            soc=explicit(psim,psil,psiCOM,itd*2*np.pi*_f)
            a=soc[-1]
            print(psiCOM, a)
            alpha[ni]=np.arctan(a)
            if nax<7:
                axs[nax].plot(soc[0],soc[1], '.', color=colors_itd[ni], alpha=0.15)
            else:
                axs[nax].plot(soc[0],soc[2], '.', color=colors_itd[ni], alpha=0.15)

        for n0,itd0 in enumerate([.1,.3,.5]):    
            iopt=np.argmin(np.abs(itd0-itd_list))
            alphaopt=alpha[iopt]
            if (nax<4)+(nax>6):
                axs[nax].plot([0,2*np.cos(alphaopt)], [0,2*np.sin(alphaopt)], '-', label='a$_{opt}$('+str(itd0)+'ms)', color=colr[n0])

        axs[nax].set_xlim([-3,3])
        axs[nax].set_ylim([-2.5,2.5])
        axs[nax].set_aspect('equal')
        if nax<4:
            axs[nax].set_title('BF = ' + str(1000*_f)+' Hz', fontsize=10)
        axs[nax].set_yticks([])
        axs[nax].set_xticks([])



axs[8].legend(bbox_to_anchor=(1.0, -.2), frameon=False)
axs[7].set_xlabel('MSO (a.u.)')
axs[1].set_ylabel('contra LSO (a.u.)')
axs[1].text(-5.5,-2,'$\psi_c=0.3 cyc$', rotation=90)
axs[4].set_ylabel('contra LSO (a.u.)')
axs[4].text(-5.5,-2,'$\psi_c=0.7 cyc$', rotation=90)
axs[7].set_ylabel('ipsi LSO (a.u.)')
axs[7].text(-5.5,-2,'$\psi_c=0.7 cyc$', rotation=90)

plt.subplots_adjust(wspace=.4, hspace=.3)

ax0=fig.add_axes([.1, .22, .25, .015])
ax0.imshow(itd_list.reshape(1,-1),  extent=[itd_list[0],itd_list[-1],0,1 ], cmap='coolwarm', aspect=.05)
ax0.invert_yaxis()
ax0.set_xlabel('ITD (ms)')
ax0.set_yticks([])


#plt.savefig('fig4.pdf')
plt.show(block=0)
