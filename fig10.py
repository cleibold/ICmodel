import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
import itdtools
import scipy.signal as scisig
from importlib import reload
from optpars import renorm
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


#
#
ITDmax=.7
itd_list=np.arange(0,ITDmax,.01)

Ncells=len(itd_list)
bins0=np.arange(len(itd_list)+1)
#
nintegrate=int(fs*.005)
#

tax=np.arange(len(mono1))/fs
t0=2.44
t1=2.47
#

r1=[]
r2=[]
rj1=[]
rj2=[]
stereo=itdtools.make_mixture([mono1, mono2], -1*np.array([.2,.4])*1e-3, fs)
iclist=[]
jefflist=[]
for nneuron, dt_neuron in enumerate(itd_list):
        
    ic,_,_,_=IC.ICfun(1.*stereo,fs, dt_neuron)
    jeff=IC.Jeffress(1.*stereo, int(dt_neuron*fs*1e-3),fs)
  
    iclist.append(ic)
    jefflist.append(jeff)
    
Ric=np.array(iclist)
Rjeff=np.array(jefflist)
r_ic = Ric/np.sqrt(np.nanmean(Ric**2))
r_jeff = Rjeff/np.sqrt(np.nanmean(Rjeff**2))
        
pks_ic=[];pks_jeff=[]
for tstep in range(int(Ric.shape[1]/nintegrate)):
    inp_ic=np.mean(Ric[:,tstep*nintegrate:(tstep+1)*nintegrate]**2,axis=1)
    inp_j=np.mean(Rjeff[:,tstep*nintegrate:(tstep+1)*nintegrate]**2,axis=1)
    
    pks_ic.append(np.argmax(inp_ic))
    pks_jeff.append(np.argmax(inp_j))

hg,bins=np.histogram(pks_ic, bins0)
peaks_ic,ret=scisig.find_peaks(hg/np.sum(hg),width=[1,3])
id_ic=np.flip(np.argsort(ret['prominences']))
hg,bins=np.histogram(pks_jeff, bins0)
peaks_jeff,ret=scisig.find_peaks(hg/np.sum(hg),prominence=.025,width=[1,3])
id_jf=np.flip(np.argsort(ret['prominences']))
ip1=peaks_ic[id_ic[0]]
ip2=peaks_ic[id_ic[1]]
ipj1=peaks_jeff[id_jf[0]]
ipj2=peaks_jeff[id_jf[1]]

r1=[];r2=[];rj1=[];rj2=[]; 
for n in range(Ncells):      
   r1.append(IC.rpearson(mono1,r_ic[n,:],fs))
   r2.append(IC.rpearson(mono2,r_ic[n,:],fs))
   rj1.append(IC.rpearson(mono1,r_jeff[n,:],fs))
   rj2.append(IC.rpearson(mono2,r_jeff[n,:],fs))

   
im1=np.argmax(r1)
im2=np.argmax(r2)
imj1=np.argmax(rj1)
imj2=np.argmax(rj2)


ic1,_,_,_=IC.ICfun(1.*stereo,fs, itd_list[im1])
ic2,_,_,_=IC.ICfun(1.*stereo,fs, itd_list[im2])
jeff1=IC.Jeffress(1.*stereo, int(itd_list[imj1]*fs*1e-3),fs)
jeff2=IC.Jeffress(1.*stereo, int(itd_list[imj2]*fs*1e-3),fs)
rpjeff1=IC.rpearson(mono1,jeff1,fs)
rpjeff2=IC.rpearson(mono2,jeff2,fs)
rp1ic=IC.rpearson(mono1,ic1,fs)
rp2ic=IC.rpearson(mono2,ic2,fs)

icp1,_,_,_=IC.ICfun(1.*stereo,fs, itd_list[ip1])
icp2,_,_,_=IC.ICfun(1.*stereo,fs, itd_list[ip2])
jeffp1=IC.Jeffress(1.*stereo, int(itd_list[ipj1]*fs*1e-3),fs)
jeffp2=IC.Jeffress(1.*stereo, int(itd_list[ipj2]*fs*1e-3),fs)
rpjeffp1=IC.rpearson(mono1,jeffp1,fs)
rpjeffp2=IC.rpearson(mono2,jeffp2,fs)
rpp1ic=IC.rpearson(mono1,icp1,fs)
rpp2ic=IC.rpearson(mono2,icp2,fs)


fig=plt.figure()   
ax=fig.add_subplot(3,4,1)
ax.plot(tax, renorm(jeff1[0:len(mono1)]*1e-4,mono1*1e-4), 'b.', alpha=0.15, markersize=2)
ax.plot(tax, mono1*1e-4, color='k')
ax.plot(tax, renorm(ic1[0:len(mono1)]*1e-4,mono1*1e-4), color='C0', alpha=0.5)
ax.set_xlim([t0,t1])
ax.set_xticklabels([])
ax.set_ylim([-1,1])
ax.text(t0+.001,.785, 'r = ' + str(np.round(rpjeff1*100)/100)+', ', color='b')
ax.text(t0+.02,.785, str(np.round(rp1ic*100)/100), color='C0')
ax.set_ylabel('2 speakers')
ax.text(2.46,1.4,'Female Speaker', color='C0')
ax.text(2.445,1.1,'Best corr.', color='C0')
    
ax=fig.add_subplot(3,4,2)
ax.plot(tax, renorm(jeffp1[0:len(mono1)]*1e-4,mono1*1e-4), 'b.', alpha=0.15, markersize=2)
ax.plot(tax, mono1*1e-4, color='k')
ax.plot(tax, renorm(icp1[0:len(mono1)]*1e-4,mono1*1e-4), color='C0', alpha=0.5)
ax.set_xlim([t0,t1])
ax.set_xticklabels([])
ax.set_ylim([-1,1])
ax.text(t0+.001,.785, 'r = ' + str(np.round(rpjeffp1*100)/100)+', ', color='b')
ax.text(t0+.02,.785, str(np.round(rpp1ic*100)/100), color='C0')
ax.text(2.445,1.1,'Best rate', color='g')

ax=fig.add_subplot(3,4,3)
ax.plot(tax, renorm(jeff2[0:len(mono1)]*1e-4,mono2*1e-4), 'r.', alpha=0.15, markersize=2)
ax.plot(tax, mono2*1e-4, color='k')
ax.plot(tax, renorm(ic2[0:len(mono1)]*1e-4,mono2*1e-4), color='C1', alpha=0.5)
ax.set_xlim([t0,t1])
ax.set_xticklabels([])
ax.set_ylim([-1,1])
ax.text(t0+.001,.785, 'r = ' + str(np.round(rpjeff2*100)/100)+', ', color='r')
ax.text(t0+.02,.785, str(np.round(rp2ic*100)/100), color='C1')
ax.text(2.46,1.4,'Male Speaker', color='C1')
ax.text(2.445,1.1,'Best corr.', color='C1')

ax=fig.add_subplot(3,4,4)
ax.plot(tax, renorm(jeffp2[0:len(mono1)]*1e-4,mono2*1e-4), 'r.', alpha=0.15, markersize=2)
ax.plot(tax, mono2*1e-4, color='k')
ax.plot(tax, renorm(icp2[0:len(mono1)]*1e-4,mono2*1e-4), color='C1', alpha=0.5)
ax.set_xlim([t0,t1])
ax.set_xticklabels([])
ax.set_ylim([-1,1])
ax.text(t0+.001,.785, 'r = ' + str(np.round(rpjeffp2*100)/100)+', ', color='r')
ax.text(t0+.02,.785, str(np.round(rpp2ic*100)/100), color='C1')
ax.text(2.445,1.1,'Best rate', color='g')



#wavfile.write('ic_female1.wav', fs, ic1.astype(np.int16))
#wavfile.write('jeff_female1.wav', fs, jeff1.astype(np.int16))
#wavfile.write('ic_male2.wav', fs, ic2.astype(np.int16))
#wavfile.write('jeff_male2.wav', fs, jeff2.astype(np.int16))
#wavfile.write('mixture.wav', fs, stereo.transpose().astype(np.int16))

stereo=itdtools.make_mixture([mono1, mono2, mono3], -1*np.array([.2,.4,-.2])*1e-3, fs)
iclist=[]
jefflist=[]
for nneuron, dt_neuron in enumerate(itd_list):
        
    ic,_,_,_=IC.ICfun(1.*stereo,fs, dt_neuron)
    jeff=IC.Jeffress(1.*stereo, int(dt_neuron*fs*1e-3),fs)
   
    iclist.append(ic)
    jefflist.append(jeff)
    
Ric=np.array(iclist)
Rjeff=np.array(jefflist)
r_ic = Ric/np.sqrt(np.mean(Ric**2))
r_jeff = Rjeff/np.sqrt(np.mean(Rjeff**2))
pks_ic=[];pks_jeff=[]
for tstep in range(int(Ric.shape[1]/nintegrate)):
    inp_ic=np.mean(Ric[:,tstep*nintegrate:(tstep+1)*nintegrate]**2,axis=1)
    inp_j=np.mean(Rjeff[:,tstep*nintegrate:(tstep+1)*nintegrate]**2,axis=1)
    
    pks_ic.append(np.argmax(inp_ic))
    pks_jeff.append(np.argmax(inp_j))


hg,bins=np.histogram(pks_ic, bins0)
peaks_ic,ret=scisig.find_peaks(hg/np.sum(hg),width=[1,3])
id_ic=np.flip(np.argsort(ret['prominences']))
hg,bins=np.histogram(pks_jeff, bins0)
peaks_jeff,ret=scisig.find_peaks(hg/np.sum(hg),prominence=.025,width=[1,3])
id_jf=np.flip(np.argsort(ret['prominences']))
ip1=peaks_ic[id_ic[0]]
ip2=peaks_ic[id_ic[1]]
ipj1=peaks_jeff[id_jf[0]]
ipj2=peaks_jeff[id_jf[1]]



ip1=peaks_ic[0]
ip2=peaks_ic[1]
ipj1=peaks_jeff[0]
ipj2=peaks_jeff[1]

r1=[];r2=[];rj1=[];rj2=[]; 
for n in range(Ncells):      
   r1.append(IC.rpearson(mono1,r_ic[n,:],fs))
   r2.append(IC.rpearson(mono2,r_ic[n,:],fs))
   rj1.append(IC.rpearson(mono1,r_jeff[n,:],fs))
   rj2.append(IC.rpearson(mono2,r_jeff[n,:],fs))


im1=np.argmax(r1)
im2=np.argmax(r2)
imj1=np.argmax(rj1)
imj2=np.argmax(rj2)
ic1,_,_,_=IC.ICfun(1.*stereo,fs, itd_list[im1])
ic2,_,_,_=IC.ICfun(1.*stereo,fs, itd_list[im2])
jeff1=IC.Jeffress(1.*stereo, int(itd_list[imj1]*fs*1e-3),fs)
jeff2=IC.Jeffress(1.*stereo, int(itd_list[imj2]*fs*1e-3),fs)

rpjeff1=IC.rpearson(mono1,jeff1,fs)
rpjeff2=IC.rpearson(mono2,jeff2,fs)
rp1ic=IC.rpearson(mono1,ic1,fs)
rp2ic=IC.rpearson(mono2,ic2,fs)

icp1,_,_,_=IC.ICfun(1.*stereo,fs, itd_list[ip1])
icp2,_,_,_=IC.ICfun(1.*stereo,fs, itd_list[ip2])
jeffp1=IC.Jeffress(1.*stereo, int(itd_list[ipj1]*fs*1e-3),fs)
jeffp2=IC.Jeffress(1.*stereo, int(itd_list[ipj2]*fs*1e-3),fs)
rpjeffp1=IC.rpearson(mono1,jeffp1,fs)
rpjeffp2=IC.rpearson(mono2,jeffp2,fs)
rpp1ic=IC.rpearson(mono1,icp1,fs)
rpp2ic=IC.rpearson(mono2,icp2,fs)


ax=fig.add_subplot(3,4,5)
ax.plot(tax, renorm(jeff1[0:len(mono1)]*1e-4,mono1*1e-4), 'b.', alpha=0.15, markersize=2)
ax.plot(tax, mono1*1e-4, color='k')
ax.plot(tax, renorm(ic1[0:len(mono1)]*1e-4,mono1*1e-4), color='C0', alpha=0.5)
ax.set_xlim([t0,t1])
ax.set_ylim([-1,1])
ax.text(t0+.001,.785, 'r = ' + str(np.round(rpjeff1*100)/100)+', ', color='b')
ax.text(t0+.02,.785, str(np.round(rp1ic*100)/100), color='C0')
ax.set_xlabel('Time (s)')
ax.set_ylabel('3 speakers')

ax.text(2.416,-.2, 'Sound pressure (a.u.)', rotation=90)

ax=fig.add_subplot(3,4,6)
ax.plot(tax, renorm(jeffp1[0:len(mono1)]*1e-4,mono1*1e-4), 'b.', alpha=0.15, markersize=2)
ax.plot(tax, mono1*1e-4, color='k')
ax.plot(tax, renorm(icp1[0:len(mono1)]*1e-4,mono1*1e-4), color='C0', alpha=0.5)
ax.set_xlim([t0,t1])
ax.set_ylim([-1,1])
ax.text(t0+.001,.785, 'r = ' + str(np.round(rpjeffp1*100)/100)+', ', color='b')
ax.text(t0+.02,.785, str(np.round(rpp1ic*100)/100), color='C0')

ax=fig.add_subplot(3,4,7)
ax.plot(tax, renorm(jeff2[0:len(mono1)]*1e-4,mono2*1e-4), 'r.', alpha=0.15, markersize=2)
ax.plot(tax, mono2*1e-4, color='k')
ax.plot(tax, renorm(ic2[0:len(mono1)]*1e-4,mono2*1e-4), color='C1', alpha=0.5)
ax.set_xlim([t0,t1])
ax.set_ylim([-1,1])
ax.text(t0+.001,.785, 'r = ' + str(np.round(rpjeff2*100)/100)+', ', color='r')
ax.text(t0+.02,.785, str(np.round(rp2ic*100)/100), color='C1')

ax=fig.add_subplot(3,4,8)
ax.plot(tax, renorm(jeffp2[0:len(mono1)]*1e-4,mono2*1e-4), 'r.', alpha=0.15, markersize=2)
ax.plot(tax, mono2*1e-4, color='k')
ax.plot(tax, renorm(icp2[0:len(mono1)]*1e-4,mono2*1e-4), color='C1', alpha=0.5)
ax.set_xlim([t0,t1])
ax.set_ylim([-1,1])
ax.text(t0+.001,.785, 'r = ' + str(np.round(rpjeffp2*100)/100)+', ', color='r')
ax.text(t0+.02,.785, str(np.round(rpp2ic*100)/100), color='C1')


fig.subplots_adjust(wspace=.4, hspace=.2)
#fig.savefig('TeX/fig10.pdf')

plt.show(block=0)

#wavfile.write('ic3_female1.wav', fs, ic1.astype(np.int16))
#wavfile.write('jeff3_female1.wav', fs, jeff1.astype(np.int16))
#wavfile.write('ic3_male2.wav', fs, ic2.astype(np.int16))
#wavfile.write('jeff3_male2.wav', fs, jeff2.astype(np.int16))
#wavfile.write('mixture3.wav', fs, stereo.transpose().astype(np.int16))
