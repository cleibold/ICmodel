import numpy as np
#

def assign_itd(mono, itd, fs):

    itd_bins=int(np.round(np.abs(itd)*fs))

    length=mono.shape[0]

    stereo=np.zeros((2,length+itd_bins))
    
    sig1=np.append(mono, np.zeros(itd_bins))#leading signal
    sig2=np.append(np.zeros(itd_bins),mono)#delayed signal

    if itd>0:
        stereo=np.array([sig1,sig2])#R Leading
    else:
        tmp=np.array([sig2,sig1])#L Leading
        stereo = np.append(tmp[:,itd_bins:],np.zeros((2,itd_bins)),axis=1)

    return stereo
    


def make_mixture(mono_list, itd_list, fs):


    stereo_list=[]
    len_list=[]
    for ns,mono in enumerate(mono_list):
        stereo=assign_itd(mono,itd_list[ns],fs)
        stereo_list.append(stereo)
        len_list.append(stereo.shape[1])

    length=max(len_list)
    stereo=np.zeros((2,length))

    for ns, src in enumerate(stereo_list):
        stereo[:,0:len_list[ns]] += src

        
    return stereo
