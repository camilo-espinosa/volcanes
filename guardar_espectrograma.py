###!/usr/bin/env python
### coding: utf-8

### In[ ]:

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.signal import butter
from scipy import signal
import random

import time
import obspy
from PIL import Image
from matplotlib import cm



filter_but = butter(N=5,Wn=[0.5,15],btype='bandpass',fs=100,output='sos')

for path in os.listdir('Datos alejandro/señales'):
    evento = pd.read_pickle(f'Datos alejandro/señales/{path}')

    if evento is not None:
        evento_completo = evento[0].data
        evento_completo = signal.sosfilt(filter_but, evento_completo)
        f, t, Sxx = spectrogram(evento_completo,fs=100,window='hamming',nperseg=100,noverlap=80,nfft=1024,mode='complex')
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

        # Transformar a PIL
        im = Image.fromarray(np.uint8(temp*255))
        # cambiar tamaño a 224x224
        im=im.resize((224,224))
        # De vuelta a numpy
        pix = np.array(im)
        np.save(f'code_test/1D_{path}.npy',pix)
        # Aplicar colormap para obtener RGB
        pix3d = getattr(cm, 'jet')(pix, bytes=True)[:,:,:3]
        # Guardar archivo con formato .npy
        np.save(f'code_test/{path}.npy',pix3d)

### %%
