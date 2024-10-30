import numpy as np

def phasewarp(fs,T,pshift):

    N=int(T*fs/2)
    ishift=int(pshift*T)

    #print(N,ishift)
    
    fleft=np.exp(1j*np.random.rand(N)*2*np.pi)
    fright=np.zeros_like(fleft)

    fright[ishift:]=fleft[:-ishift]
    fright[:ishift]=fleft[-ishift:]

    left=np.real(np.fft.ifft(np.append(fleft,np.zeros_like(fleft))))*2
    right=np.real(np.fft.ifft(np.append(fright,np.zeros_like(fleft))))*2

    return left,right

