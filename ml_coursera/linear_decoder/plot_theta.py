import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from tqdm import tqdm

def plot(images):
    print 'plotting images.shape:', images.shape                      
    plt.figure(figsize=(12,12))
    images = images - np.mean(images.flatten())
    channel_size = 64
    B = images[0:channel_size,:]
    C = images[channel_size:channel_size*2,:]
    D = images[2*channel_size:channel_size*3,:]
    print "B,C,D:", B.shape, C.shape, D.shape
    B = B /(np.ones((B.shape[0],1))*np.max(abs(B.flatten()))).astype(float)
    C = C /(np.ones((C.shape[0],1))*np.max(abs(C.flatten()))).astype(float)
    D = D /(np.ones((D.shape[0],1))*np.max(abs(D.flatten()))).astype(float)
    for i in tqdm(range(400)):
        plt.subplot(20,20,i+1)
        ind = i #np.random.choice(range(images.shape[1]),1)
        img = np.zeros((8,8,3))
        img[:,:,2] = np.reshape(B[:,ind],(8,8))
        img[:,:,1] = np.reshape(C[:,ind],(8,8))
        img[:,:,0] = np.reshape(D[:,ind],(8,8))
        plt.imshow(img)#,norm=LogNorm())
        plt.xticks([])
        plt.yticks([])
    plt.show()

w1 = np.loadtxt('trained_W1.txt')
plot(w1[:,1:].T)

