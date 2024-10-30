import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle

from scipy.interpolate import CloughTocher2DInterpolator as interpolate
from scipy.interpolate import interp1d as interpolate1
#

##### physiological range
itdmax=0.7
################

maxa=10
itd_range=np.arange(0,itdmax+.025,.025)
#
psil=0*2*np.pi
psim=np.arange(0.125,.135,.025)*2*np.pi
psiCOM=np.arange(0.,1.01,.0125)*2*np.pi
#
psim=np.mod(psim+np.pi,2*np.pi)-np.pi
psiCOM=np.mod(psiCOM+np.pi,2*np.pi)-np.pi
#
savflag=False


def projection(a,psim,psil,psiCOM,dphi):

    gammaR = 1-a*np.exp(-1j*(psil+psiCOM))
    gammaL = np.exp(1j*dphi)*(np.exp(-1j*psim) + a*np.exp(-1j*psiCOM))
    
    idneg=np.where(a<0)[0]

    gammaR[idneg] = 1 + a[idneg]
    gammaL[idneg] = np.exp(1j*dphi)*(np.exp(-1j*psim) - a[idneg] * np.exp(-1j*psil))

    gammaR2 = np.abs(gammaR)**2
    gammaL2 = np.abs(gammaL)**2

    phase  = np.angle(gammaR)-np.angle(gammaL)
    
    return (gammaR2 + gammaL2 + 2.*np.sqrt(gammaR2*gammaL2)*np.cos(phase))/(1+a**2)

def aopt(psim,psil,psiCOM,dphi):

    avals=np.arange(-maxa,maxa+.02,.02)
    pia=projection(avals,psim,psil,psiCOM,dphi)

    #plt.figure()
    #plt.plot(avals,pia)
    #plt.show(block=0)
    
    return avals[np.argmax(pia)]
    

def explicit(psim,psil,psiCOM,dphi,pflag=False):

    phi=np.arange(0,1.01,.01)*2*np.pi
    
    MSO=np.cos(phi) + np.cos(phi+dphi-psim)
    LSO=np.cos(phi+dphi-psiCOM) - np.cos(phi-psil-psiCOM)
    iLSO=np.cos(phi) - np.cos(phi+dphi-psil)

    
    tmp=np.linalg.eig(np.cov(np.array([MSO,LSO])))
    ma=tmp[1][:,np.argmax(tmp[0])]
    ma=ma/ma[0]

    tmp=np.linalg.eig(np.cov(np.array([MSO,iLSO])))
    mai=tmp[1][:,np.argmax(tmp[0])]
    mai=mai/mai[0]

    a=aopt(psim,psil,psiCOM,dphi)
    

    if pflag==True:
        plt.plot(MSO,LSO,'.k')
        plt.plot(MSO,iLSO,'.b')
        plt.plot([0, ma[0]],[0, ma[1]], '-g')
        plt.plot([0, mai[0]],[0, mai[1]], '-c')

        alpha=np.arctan(a)
        plt.plot([0, np.cos(alpha)],[0, np.sin(alpha)], '-r')
        plt.xlim([-2,2])
        plt.ylim([-2,2])
        #plt.set_aspect('equal')
    

    #print(ma[1],mai[1],a)
    
    return MSO,LSO,iLSO, ma, mai, a



def renorm(target,source):
    
    scale=np.sqrt(np.nanmean(source**2)/np.nanmean(target**2))
    
    return target*scale


lambda_neg=.5
def optpsipars(freq):

    phi_range=itd_range*freq*2*np.pi

    loss=10000*np.ones((len(psiCOM),len(psim)))
    for n1,_pc in enumerate(psiCOM):
        for n2,_pm in enumerate(psim):
            aoptarr=np.array([aopt(_pm,psil,_pc,dphi) for dphi in phi_range])
            loss[n1,n2] = np.mean(-1*np.log(np.abs(aoptarr)) + np.abs(aoptarr)) + lambda_neg*np.mean(aoptarr<0)
            #loss[n1,n2] = np.mean( np.abs(aoptarr)) + lambda_neg*np.mean(aoptarr<0)

    
    #plt.figure()
    #plt.imshow(loss)
    #plt.show(block=0)
                       
    n1opt, n2opt=np.unravel_index(np.argmin(loss, axis=None), loss.shape)
    aoptarr=np.array([aopt(psim[n2opt],psil,psiCOM[n1opt],dphi) for dphi in phi_range])

    #plt.figure()
    #plt.plot(itd_range,aoptarr)
    #plt.show(block=0)
     
    
    return psim[n2opt], psiCOM[n1opt], aoptarr, phi_range




if __name__ == "__main__":

    farr=np.arange(.1,2,.05)
    Nfreq=len(farr)
    #
    if savflag==True:
        f0=.011
        fe=49.
        Nfreq=100
        farr=f0*((fe/f0)**(np.arange(Nfreq)/(Nfreq-1)))
    #

    psim_opt=np.zeros(len(farr))
    psic_opt=np.zeros(len(farr))
    a_opt=np.zeros((len(farr),len(itd_range)))
    points=[]

    for nf,_f in enumerate(farr):

        psim_opt[nf], psic_opt[nf], a_opt[nf,:], _ = optpsipars(_f)
        [points.append([_f, _dt]) for _dt in itd_range]

        print(_f, nf/len(farr), a_opt[nf,:])

    fig=plt.figure()
    ax=fig.add_subplot(2,2,1)
    ax.plot(farr,psic_opt/2/np.pi)
    ax.set_ylabel('$\psi_{COM}$ (cyc)')
    ax.set_xscale('log')


    ax=fig.add_subplot(2,2,3)
    irange=itdmax
    limmin=-1#np.min(avals)
    limmax=1#np.max(avals)
    ax.imshow(a_opt.T, extent=[itd_range[0],itd_range[-1] , farr[-1], farr[0] ], vmin=limmin, vmax=limmax, cmap='PiYG', aspect=irange/(farr[-1]-farr[0]))
    ax.invert_yaxis()
    plt.show(block=0)


    amat=copy.copy(a_opt)

    holes=np.array(np.where(np.isnan(amat))).T
    for hole in holes:
        ni,nf=hole
        nim=np.max([0,ni-1])
        nip=np.min([len(itd_range)-1,ni+1])
        nfm=np.max([0,nf-1])
        nfp=np.min([len(farr)-1,nf+1])
        cross=np.array([amat[nip,nf],amat[nim,nf],amat[ni,nfm],amat[ni,nfp]])
    
        amat[ni,nf]=np.nanmean(cross)
        print(amat[ni,nf], hole)

    avalues=[]
    for nf in range(len(farr)):
        for ni in range(len(itd_range)):
            avalues.append(amat[nf,ni])


    if savflag==True:
        a_fun=interpolate(np.array(points), np.array(avalues))
        pm_fun=interpolate1(farr, psim_opt)
        pc_fun=interpolate1(farr, psic_opt)

        with open('interpol_a_'+str(int(10*itdmax))+'_.pkl', 'wb') as f_d:
            pickle.dump(a_fun, f_d)
        with open('interpol_pm_'+str(int(10*itdmax))+'_.pkl', 'wb') as f_d:
            pickle.dump(pm_fun, f_d)
        with open('interpol_pc_'+str(int(10*itdmax))+'_.pkl', 'wb') as f_d:
            pickle.dump(pc_fun, f_d)

