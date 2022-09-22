###!/usr/bin/env python
### coding: utf-8

### In[ ]:

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.signal import stft
from scipy.signal import butter
from scipy import signal
import random
from matplotlib import figure
import matplotlib 
#matplotlib.use('agg')
matplotlib.use('TkAgg')
import time
import obspy
from librosa import stft as stft_2
from PIL import Image
from matplotlib import cm


kjk=0
filter_but = butter(N=5,Wn=[0.5,15],btype='bandpass',fs=100,output='sos')
all_init = time.time()
time_to_process=[]
for path in os.listdir('Datos alejandro/señales'):
    evento = pd.read_pickle(f'Datos alejandro/señales/{path}')
    t_init = time.time()
    if evento is not None:
        evento_completo = evento[0].data
        evento_completo = signal.sosfilt(filter_but, evento_completo)
        f, t, Sxx = spectrogram(evento_completo,fs=100,window='hamming',nperseg=100,noverlap=80,nfft=1024,mode='complex')
        # f, t, Sxx = stft(evento_completo,fs=100,window='hamming',nperseg=100,noverlap=80,nfft=1024)
        # Sxx = stft_2(evento_completo,window='hamming', win_length=200,n_fft=1024, hop_length=20,dtype=np.complex128)
        Sxx = Sxx[0:300,:]
        product = Sxx*np.conj(Sxx)
        temp = np.log10(100*product.real+1e-5)
        mini = temp.min()
        maxi = temp.max()
        temp = (temp - mini) / (maxi - mini)
        k = temp.min() + 0.5*(temp.max()-temp.min())
        temp[temp <= k] = k
        temp = (temp - temp.min()) / (temp.max() - temp.min())
        w=224
        h=224
        im = Image.fromarray(np.uint8(temp*255))
        im=im.resize((224,224))
        pix = np.array(im)
        pix3d = getattr(cm, 'jet')(pix, bytes=True)[:,:,:3]
        np.save(f'code_test/pix_test{kjk}.npy',pix3d)
        # fig = plt.figure(frameon=False)
        # fig.set_size_inches(w,h)
        # plt.pcolormesh(temp, shading='gouraud',cmap="jet")
        # plt.axis('off')
        # plt.savefig(f'code_test/original{kjk}.png',dpi=1)
        # plt.close(fig)
        # plt.close('all')
        # plt.cla() 
        # plt.clf()
        kjk+=1
    t_end = time.time()
    time_to_process.append(t_end - t_init)

print(f'total time = {t_end-all_init}','\n',
np.min(np.array(time_to_process)),'\n',
np.mean(np.array(time_to_process)),'\n',
np.max(np.array(time_to_process)))

# spectrogram = 0.29 s
# guardar txt = 4.69 s
# guardar image = 14 segundos
# lectura datos + librosa + savetxt = 0.599 s (min =0.10, max =3.18) (22 datos)
# lectura datos + stft + savetxt = 0.583 s (min =0.090, max =3.17) (22 datos)
# lectura datos + spectrogram + savetxt = 0.590 s (min =0.10, max =3.173) (22 datos)
# lectura datos + spectrogram + plt.savefig = 2.094 s (min =0.59, max =9.79) (22 datos)

# modo antiguo (con pcolormesh)
# total time = 80.24442791938782
#  1.195021629333496
#  3.6334417950023306
#  16.193886280059814

# modo nuevo:
# total time = 3.26
#  minimo = 0.031
#  media  = 0.126
#  máximo = 0.585

### %%
