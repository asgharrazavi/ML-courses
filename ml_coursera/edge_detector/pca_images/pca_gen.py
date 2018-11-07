import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

#---------------- loading data -----------------------
data = io.loadmat('pca_exercise/IMAGES_RAW.mat')
print data.keys()
images = data['IMAGESr']
print "images.shape:", images.shape
#-----------------------------------------------------

#----------------------------------display images-----------
def display_images(patches):
    plt.figure(figsize=(15,15))
    gs1 = gridspec.GridSpec(2,5)
    gs1.update(wspace=0.001, hspace=0.001) # set the spacing between axes. 
    for i in range(10):
        img = patches[:,:,i]
        ax1 = plt.subplot(gs1[i])
        ax1.imshow(img,cmap='gray')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    plt.show()
#display_images(images) 
#----------------------------------------------------------------------------

#----------------------------------display_network---------------------------
def display_network(patches,num=100):
    plt.figure(figsize=(15,15))
    gs1 = gridspec.GridSpec(10,10)
    gs1.update(wspace=0.001, hspace=0.001) # set the spacing between axes. 
    for i in range(num):
	ind1 = np.random.choice(range(patches.shape[2]),1)
    	ind2 = np.random.choice(range(patches.shape[0]-12),1)
        img = patches[ind2:ind2+12,ind2:ind2+12,ind1][:,:,0]
        ax1 = plt.subplot(gs1[i])
        ax1.imshow(img,cmap='gray')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_ylim([0,img.shape[0]])
    plt.show()
#display_network(images,num=100) 
#----------------------------------------------------------------------------

#-----------------------display patches------------------------
def display_patches(patches,name,num=100):
    plt.figure(figsize=(15,15))
    gs1 = gridspec.GridSpec(10,10)
    gs1.update(wspace=0.001, hspace=0.001) # set the spacing between axes. 
    for i in range(num):
   	ind = np.random.choice(range(patches.shape[1]),1)
        ax1 = plt.subplot(gs1[i])
        img = np.reshape(patches[:,ind],(int(np.sqrt(patches.shape[0])),int(np.sqrt(patches.shape[0]))))
        ax1.imshow(img,cmap='gray')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_ylim([0,img.shape[0]])
    plt.savefig(name)
    plt.show()
#----------------------------------------------------------------

#---------sampleIMAGES---------------------------------------------------------------------------------------------------------
def sampleIMAGES(images):
    patchsize = 12  # we'll use 12x12 patches 
    numpatches = 10000
    patches = np.zeros((patchsize*patchsize, numpatches))
    for i in range(numpatches):
        img_id = np.random.choice(range(10),1)
        image = images[:,:,img_id]
        ind_x = np.random.choice(range(image.shape[0]-patchsize),1)
        ind_y = np.random.choice(range(image.shape[0]-patchsize),1)
        img = image[ind_x:ind_x+patchsize,ind_y:ind_y+patchsize].flatten()
        patches[:,i] = img
    return patches

def get_patches():
    patches2 = sampleIMAGES(images)
    print "patches2.shape:", patches2.shape					#(144, 10000)
#    patches = np.array([patches2[:,i] - np.mean(patches2[:,i]) for i in range(patches2.shape[1])]).T
    patches = patches2
    print "patches.shape:", patches.shape					#(144, 10000)
    display_patches(patches,'sample.png',num=100)
    np.savetxt('patches.txt',patches)
    return patches
#try: patches = np.loadtxt('patches.txt')
#except: print "getting patches...\n"; patches = get_patches()
patches = get_patches()
#--------------------------------------------------------------------------------------------------------------------------------

#---------PCA----------------------------------------------------------------------------------
def pca(patches):
    cov = np.dot(patches,patches.T) / float(patches.shape[1])
#    u, s, v = np.linalg.svd(cov)
    s, u = np.linalg.eigh(cov)
    print "eigenvalues.shape, eigenvectors.shape:", s.shape, u.shape
#    plt.imshow(u,cmap='Blues')
#    plt.ylim([0,u.shape[0]])
#    plt.show()
    return cov, u, s
cov, vecs, vals = pca(patches)
contr = vals / float(np.sum(vals))
c99, k = 0, 0
while c99 < 0.99:
    c99 = np.sum(contr[0:k])
    k += 1
k = k - 1
#plt.plot(vals,'.-')
#plt.show()
plt.imshow(cov)
plt.show()
print "number of pca components for 99 percent contribution to total fluctuation:", k
#plt.plot(vals)
#plt.show()
#-----------------------------------------------------------------------------------------------

#-----------xHat---------------------
def plot_pca_white(vecs,k,patches,num=100):
    uu = vecs.copy()
    for i in range(k+1,vecs.shape[0]): print i; uu[:,k] = 0
#    xrot = np.dot(uu.T , patches)
    xrot = np.dot(vecs.T , patches)
    print "xrot.shape:", xrot.shape			#(144, 10000)
    display_patches(patches,'rotated_all_vecs.png',num=100)

#plot_pca_white(vecs,k,patches)


