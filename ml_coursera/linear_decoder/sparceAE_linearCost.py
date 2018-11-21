import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import scipy.optimize as op
import copy
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from numba import autojit
import time
from tqdm import tqdm

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''These are data for sparce autoencoder'''''''''''''''''''''''''''''''''''''''''''''''''''''
#-----------loading data----------------
#data = io.loadmat('IMAGES.mat')
#print data.keys()
#images = data['IMAGES']
#print 'images.shape:', images.shape			# (512, 512, 10)
def plot():
    plt.figure(figsize=(20,7))
    for i in range(images.shape[2]):
        plt.subplot(2,5,i+1)
        plt.imshow(images[:,:,i],norm=LogNorm())
    plt.show()
#plot()
#--------------------------------------
#---------sampleIMAGES---------------------------------------------------------------------------------------------------------
def sampleIMAGES():
    patchsize = 8  # we'll use 8x8 patches 
    numpatches = 20000
    patches = np.zeros((patchsize*patchsize, numpatches))
#    img_id = np.random.choice(range(10),1)
#    image = images[:,:,img_id]
#    print "selected image id:", img_id
    for i in range(numpatches):
        img_id = np.random.choice(range(10),1)
        image = images[:,:,img_id]
        ind_x = np.random.choice(range(image.shape[0]-patchsize),1).astype(int)
        ind_y = np.random.choice(range(image.shape[0]-patchsize),1).astype(int)
#       print ind_x, ind_y
        img = image[ind_x[0]:ind_x[0]+patchsize,ind_y[0]:ind_y[0]+patchsize].flatten()
        patches[:,i] = img
    return patches
def get_patches_s():
    patches = sampleIMAGES()
    patches2 = np.array([patches[:,i] - np.mean(patches[:,i]) for i in range(patches.shape[1]) ]).T
    pstd3 = 3 * np.std(patches2.flatten())
    ind_min = (patches2 < -pstd3)
    ind_max = (patches2 >  pstd3)
    patches2[ind_min] = -pstd3
    patches2[ind_max] = pstd3
    patches2 = patches2 / float(pstd3)
    print "min(patches2), max(patches2):", np.min(patches2), np.max(patches2)
    # Rescale from [-1,1] to [0.1,0.9]
    patches = (patches2 + 1) * 0.4 + 0.1;
    print "min, max:", np.min(patches), np.max(patches)
    print "patches.shape:", patches.shape				# (64, 20000) 
    return patches
#patches = get_patches_s() 
#I was supposed to Squash data to [0.1, 0.9] since we use sigmoid as the activation function in the output layer by 
#"Truncate to +/-3 standard deviations and scale to -1 to 1" and "Rescale from [-1,1] to [0.1,0.9]"
##I did it!
#------------------------------------------------------------------------------------------------------------------------------
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''These are data for Linear decoder'''''''''''''''''''''''''''''''''''''''''
#-----------loading data----------------
#data = io.loadmat('stlSampledPatches.mat')
#print data.keys()
#images = data['patches']#[:,::10000]
#print 'images.shape:', images.shape			# (192, 100000)
def plot(images):
    print 'plotting images.shape:', images.shape			# (192, 100000)
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
    for i in range(100):
        plt.subplot(10,10,i+1)
	ind = i #np.random.choice(range(images.shape[1]),1)
  	img = np.zeros((8,8,3))
	img[:,:,2] = np.reshape(B[:,ind],(8,8))
	img[:,:,1] = np.reshape(C[:,ind],(8,8))
	img[:,:,0] = np.reshape(D[:,ind],(8,8))
        plt.imshow(img)#,norm=LogNorm())
  	plt.xticks([])
  	plt.yticks([])
    plt.show()
#plot()
#images = np.reshape(images,(12,16,images.shape[1]))
#--------------------------------------
#---------sampleIMAGES---------------------------------------------------------------------------------------------------------
def get_pacthes():
    global ZCAWhite
    # this is the way that it should be (if I coded it right)...................
    patches = images
#    epsilon = 0.1
#    patches1 = np.array([patches[i] - np.mean(patches[i]) for i in range(patches.shape[0]) ])
#    patches = patches1
#    sigma = np.dot(patches,patches.T) / float(images.shape[1])
#    u, s, v = np.linalg.svd(sigma)
#    print "shapes: u,s,v:", u.shape, s.shape, v.shape
#    xrot = np.dot(u , patches)
#    xrot2 = np.array([xrot[i] / float(np.sqrt(s[i]+epsilon)) for i in range(len(xrot))])
#    patches = np.dot(u.T,xrot2)
#    ZCAWhite = xrot2
#    print "min, max:", np.min(patches), np.max(patches)
#    print "pathes.shape:", patches.shape
##    np.savetxt('zca_patches.txt',patches)
    #............................................................................

    # this also didn't work.................
    patches2 = np.array([patches[:,i] - np.mean(patches[:,i]) for i in range(patches.shape[1]) ]).T
    pstd3 = 3 * np.std(patches2.flatten())
    ind_min = (patches2 < -pstd3)
    ind_max = (patches2 >  pstd3)
    patches2[ind_min] = -pstd3
    patches2[ind_max] = pstd3
    patches2 = patches2 / float(pstd3)
    print "min(patches2), max(patches2):", np.min(patches2), np.max(patches2)
    # Rescale from [-1,1] to [0.1,0.9]
    patches = (patches2 + 1) * 0.4 + 0.1;
    print "min, max:", np.min(patches), np.max(patches)
    #......................................
    print "patches.shape:", patches.shape                         

    return patches
#patches = get_pacthes()					# (192, 100000)
#patches = patches[64*1:64*2,0:20000]

#------------------------------------------------------------------------------------------------------------------------------
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#'''''''''''''''''''''''''''''''''''''''''''''''''''''load data processed by matlab'''''''''''''''''
patches = io.loadmat('data_from_matlab/patches.mat')
print patches.keys()
ZCAWhite = io.loadmat('data_from_matlab/ZCAWhite.mat')
print ZCAWhite.keys()
patches = patches['patches']
ZCAWhite = ZCAWhite['ZCAWhite']
print "patches.shape:", patches.shape
print "ZCAWhite.shape:", ZCAWhite.shape
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#----------------------------------display_network---------------------------
def display_network(patches,num=200):
    inds = np.random.choice(range(patches.shape[1]),num)
    sel_patches = patches[:,inds]
    plt.figure(figsize=(15,15))
    gs1 = gridspec.GridSpec(20,10)
    gs1.update(wspace=0.001, hspace=0.001) # set the spacing between axes. 
    for i in range(num):
        ax1 = plt.subplot(gs1[i])
        img = np.reshape(sel_patches[:,i],(int(np.sqrt(sel_patches[:,i].shape[0])),int(np.sqrt(sel_patches[:,i].shape[0]))))
#        ax1.imshow(img,norm=LogNorm(),cmap='gray')
        ax1.imshow(img,norm=LogNorm())
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
  #     ax1.set_aspect('equal')
    plt.show()
#display_network(patches,num=200) 
#----------------------------------------------------------------------------

#-----------initializeParameters--------------------------------
def initializeParameters(hidden_layer_size, input_layer_size):
    #Initialize parameters randomly based on layer sizes.
    r  = np.sqrt(6) / np.sqrt(hidden_layer_size+input_layer_size+1);   # we'll choose weights uniformly from the interval [-r, r]
    W1 = np.random.random((hidden_layer_size, input_layer_size + 1)) * 2 * r - r;
    W2 = np.random.random((input_layer_size, hidden_layer_size + 1)) * 2 * r - r;
    b1 = np.zeros((hidden_layer_size, 1))
    b2 = np.zeros((input_layer_size, 1))
    theta = np.concatenate((W1.flatten(),b1.flatten(),W2.flatten(),b2.flatten()))
    return theta
#---------------------------------------------------------------

#============================================================sparseAutoencoderCost===================================================================================
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))
def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))
def randInitializeWeights(L_out, L_in):
    epsilon_init = 0.12                 # This number comes from: ( sqrt(6) / (sqrt{(# of layers before)+(# of layers after)}) )
    return np.random.random((L_out, 1 + L_in)) * 2 * epsilon_init - epsilon_init
def debug_randInitializeWeights(L_out, L_in):
    return np.reshape(np.sin(range(L_out * (1 + L_in))),(L_out, 1 + L_in))


############ somehow this function givs wrong results ######################################################################
def _h_theta(W1, W2, b1, b2, X):                 #E.g. Theta1: (25,401), Theta2: (10,26), X: (5000,401)
    a1 = X.T
    B1 = np.array([b1[:,0] for i in range(X.shape[0])]).T; z2 = np.dot(W1,a1) + B1; a2 = sigmoid(z2) 
    a2 = np.concatenate((a2,np.ones((1,a2.shape[1]))))
    B2 = np.array([b2[:,0] for i in range(X.shape[0])]).T; z3 = np.dot(W2,a2) + B2; a3 = sigmoid(z3)
#    print "x.shape, a1.shape, z2.shape, a2.shape, z3.shape, a3.shape, B1.shape, B2.shape:"
#    print  X.shape, a1.shape, z2.shape, a2.shape, a3.shape, a3.shape, B1.shape, B2.shape    #(100, 65) (65, 100) (25, 100) (26, 100) (64, 100) (64, 100) (25, 100) (64, 100)
    return z2, z3, a1, a2, a3
#############################################################################################################################

#---------we don't need this function in this neural network----------
def _fix_y(y,num_labels):
    yy = np.zeros((len(y),num_labels))
    for i in range(len(y)): yy[i] = 1
    return yy.T
#--------------------------------------------------------------------

@autojit
def _add_regCost(W1,W2,y,lambdaa):
    m = float(len(y))
    reg_j = 0
    reg_j += np.sum(W1[:,1:]**2)  # [:,1:]:we don't regularize the terms that correspond to the bias 
    reg_j += np.sum(W2[:,1:]**2)  # [:,1:]:we don't regularize the terms that correspond to the bias 
    reg_j = (lambdaa/(2.0)) * reg_j
    return reg_j

@autojit
def _reshape_theta(theta, input_layer_size, hidden_layer_size, num_labels):
#    t0 = time.time()
#    print "lens(theta, input_layer_size, hidden_layer_size, num_labels):", len(theta), input_layer_size, hidden_layer_size, num_labels
    #Theta1: (25,64+1), b1: (25,1); Theta2: (64,26), b2: (64,1), X: (5000,64+1)
    theta1 = np.reshape(theta[0:hidden_layer_size*(input_layer_size + 1)],(hidden_layer_size,input_layer_size + 1))
    B1 = np.reshape(theta[hidden_layer_size*(input_layer_size + 1):(hidden_layer_size*(input_layer_size + 1))+hidden_layer_size],(hidden_layer_size, 1))
    theta2 = np.reshape(theta[ len(theta1.flatten())+len(B1.flatten()):  \
			       len(theta1.flatten())+len(B1.flatten()) + (input_layer_size*(hidden_layer_size+1))],(input_layer_size, hidden_layer_size + 1))
    B2 = np.reshape(theta[len(theta1.flatten())+len(B1.flatten()) + (input_layer_size*(hidden_layer_size+1)):] ,(input_layer_size, 1))
#    t1 = time.time()
#    print "\t reshape_theta run time: %1.2f" %(t1-t0)		#This step is fast 
    return theta1, B1, theta2, B2

def sparseAutoencoderCost(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa, sparsityParam, beta, reshape_theta=True, debug=False):
#    t0 = time.time()
    if reshape_theta : W1, b1, W2, b2 = _reshape_theta(theta, input_layer_size, hidden_layer_size, num_labels)
#    nn_params = [W1, b1, W2, b2]
    X = np.c_[np.ones(X.shape[0]),X]
    m = len(y)			#this is correct
    z2, z3, a1, a2, a3 = _h_theta2(W1, b1, W2, b2, X)
    a3_yy = np.array([np.sum((a3[i] - y[i])**2) for i in range(y.shape[0])])
    J = (1.0/m) * np.sum(0.5*a3_yy)
    reg_J = _add_regCost(W1,W2,y,lambdaa)
    ro = float(sparsityParam)
    ro_h = np.mean(a2,axis=0)  
    ind1, ind2 = (ro_h == 0) , (ro_h == 1) ; ro_h[ind1] = 1e-15 ; ro_h[ind2] = 1-1e-15
    kl_J = beta * np.sum( ( ro * np.log(ro/ro_h) ) + ( (1 - ro) * np.log((1-ro)/(1-ro_h)) ) )
    cost =  J + reg_J + kl_J

    g_prime_z2 = sigmoidGradient(z2)
    g_prime_z3 = sigmoidGradient(z3)
    ro_H = np.mean(a2,axis=0)[1:] 
    ro_h = np.array([ro_H for i in range(a2.shape[0])])
    ind1, ind2 = (ro_h == 0) , (ro_h == 1)
    ro_h[ind1] = 1e-15
    ro_h[ind2] = 1-1e-15
    kl_term = beta * ( (float(1-ro)/(1-ro_h)) - (ro/ro_h) )
#    delta3 = (a3 - y) * g_prime_z3			# This is for sparce autoencoder
    delta3 = (a3 - y) 					# This is for Linear decoder
    delta2 = (np.dot(W2[:,1:].T,delta3.T).T + kl_term) * g_prime_z2
    DW1 = np.dot(delta2.T, a1.T)
    DW2 = np.dot(delta3.T, a2)
    DW1 = (1.0/m) * DW1
    DW2 = (1.0/m) * DW2
    DW1[:,1:] = DW1[:,1:] + (float(lambdaa)) * W1[:,1:]              				# reqularizing all output nodes and 1:all input nodes
    DW2[:,1:] = DW2[:,1:] + (float(lambdaa)) * W2[:,1:]
    DB1 = (1.0/m) * delta2
    DB2 = (1.0/m) * delta3
    DB1 = np.sum(DB1,axis=0)
    DB2 = np.sum(DB2,axis=0)
    D = []
    D.extend(DW1.flatten()); D.extend(DB1.flatten())
    D.extend(DW2.flatten()); D.extend(DB2.flatten())
#    t1 = time.time()
#    print "sparce autoencoder run time: %1.2f" %(t1-t0) 
    if debug: return cost, DW1, DB1, DW2, DB2
    else: return cost, np.array(D)

def _h_theta2(W1, b1, W2, b2, X):                 #Theta1: (25,401), Theta2: (10,26), X: (5000,401)
#    t0 = time.time()
#    t00 = time.time()
    a1 = X.T
    B1 = np.array([b1[:,0] for i in range(X.shape[0])]) ; B2 = np.array([b2[:,0] for i in range(X.shape[0])])
#    t10 = time.time()
#    t01 = time.time()
    z2 = np.dot(W1,a1).T + B1; a2 = sigmoid(z2) ; a2 = np.c_[np.ones(a2.shape[0]),a2]
    z3 = np.dot(W2,a2.T).T + B2
#    a3 = sigmoid(z3)				# This is for sparce autoencoder
    a3 = z3					# This is for Linear decoder
#    t11 = time.time()
#    t1 = time.time()
#    print "\t total _h_theta2 run time: %1.2f" %(t1-t0)			#this 2 (dot products in 5 times slower)
#    print "\t\t part 1 time: %1.2f \tpart 2 time: %1.2f" %(t10-t00,t11-t01)
    return z2, z3, a1, a2, a3

def computeNumericalGradient(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa ,sparsityParam, beta, reshape_theta=True, debug=True):
    unrolled_theta = np.array(theta)
    numgrad = np.zeros(len(unrolled_theta))
    perturb = np.zeros(len(unrolled_theta))
    print "numgrad.shape, perturb.shape:", numgrad.shape, perturb.shape
    e = 1e-4;
    for p in range(len(unrolled_theta)):
        if p%500 == 0: print p, "percent completed:", p*100/len(unrolled_theta)
        perturb[p] = e
# 	print "len(unrolled_theta - perturb):", len(unrolled_theta - perturb)
        loss1 = sparseAutoencoderCost(unrolled_theta - perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa, sparsityParam, beta, reshape_theta=True, debug=True)[0]
        loss2 = sparseAutoencoderCost(unrolled_theta + perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa, sparsityParam, beta, reshape_theta=True, debug=True)[0]
        numgrad[p] = (loss2 - loss1) / (2.0*e)
        perturb[p] = 0
    return numgrad
#====================================================================================================================================================================

#-----------------Debugging Cost function and Gradient---------------------------------------------
debug = False
if debug:
    debugHiddenSize = 10 #5
    debugvisibleSize =  192 #8
#    patches = np.random.random((8,10)).T
    patches = patches.T
    num_labels = patches.shape[1]
    sparsityParam = 0.035
    lambdaa = 3e-3
    beta = 5  
#    theta = initializeParameters(debugHiddenSize, debugvisibleSize);
    initial_Theta1 = debug_randInitializeWeights(debugHiddenSize, debugvisibleSize)
    initial_Theta2 = debug_randInitializeWeights(num_labels, debugHiddenSize)
    initial_b1 = debug_randInitializeWeights(debugHiddenSize, 0)
    initial_b2 = debug_randInitializeWeights(num_labels, 0)
    print " initial_Theta1.shape, initial_b1.shape, initial_Theta2.shape, initial_b2.shape", initial_Theta1.shape, initial_b1.shape, initial_Theta2.shape, initial_b2.shape
#    print " initial_Theta1, initial_b1, initial_Theta2, initial_b2", initial_Theta1, initial_b1, initial_Theta2, initial_b2
    initial_nn_params = [initial_Theta1, initial_b1, initial_Theta2, initial_b2]
    theta = []
    for i in range(len(initial_nn_params)): theta.extend(initial_nn_params[i].flatten())
    print "debug X and y shapes:", patches.shape, patches.shape

    cost, DW1, DB1, DW2, DB2 = sparseAutoencoderCost(theta, debugvisibleSize, debugHiddenSize, num_labels, patches, patches, lambdaa ,sparsityParam, beta, reshape_theta=True, debug=True)
    np.savetxt('W1_debug.txt',DW1)
    np.savetxt('B1_debug.txt',DB1)
    np.savetxt('W2_debug.txt',DW2)
    np.savetxt('B2_debug.txt',DB2)
#----check gradient against numerical gradient--
    numgrad = computeNumericalGradient(theta, debugvisibleSize, debugHiddenSize, num_labels, patches, patches, lambdaa ,sparsityParam, beta, reshape_theta=True, debug=True)
    np.savetxt('num_grad_unrolled_debug.txt',numgrad)
#    print grad[0][0:10], numgrad[0:10]
def plot_num_gradient(grad,numgrad):
    unrolled_grad = []
    unrolled_grad.extend(grad[0].flatten())
#    unrolled_grad.extend(np.mean(grad[1],axis=1).flatten())
    unrolled_grad.extend(grad[1].flatten())
    unrolled_grad.extend(grad[2].flatten())
#    unrolled_grad.extend(np.mean(grad[3],axis=1).flatten())
    unrolled_grad.extend(grad[3].flatten())
    diff = np.linalg.norm(numgrad-unrolled_grad) / np.linalg.norm(numgrad+unrolled_grad).astype(float)
    print "diff:", diff
    plt.plot(unrolled_grad - numgrad,'r.')
    plt.xlabel('Tetha');plt.ylabel('grad - numgrad')
    plt.show()
if debug: plot_num_gradient([DW1, DB1, DW2, DB2],numgrad) ; quit()
#--------------------------------------------------------------------------------------------------


#===============Main training NN===================================================================================================================================
imageChannels = 3 #1 #3
patchDim   = 8
numPatches = 100000# 10000 #100000
visibleSize = patchDim * patchDim * imageChannels
outputSize  = visibleSize
hiddenSize  = 400 # 25 #400
sparsityParam = 0.035 # 0.01 #0.035
lambdaa = 0.003 # 0.0001 #3e-3
beta = 5.0 #3 #5
MaxIter = 400
X = patches.T
def train_nn():
    unrolled_theta = initializeParameters(hiddenSize, visibleSize)
    out = op.fmin_l_bfgs_b(sparseAutoencoderCost,unrolled_theta,fprime=None,args=(visibleSize, hiddenSize, 'num_labels', X, X, lambdaa, sparsityParam, beta),maxfun=MaxIter, disp=1)
#    out = op.fmin_cg(sparseAutoencoderCost,unrolled_theta,nnBackPro,args=(input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa, sparsityParam, beta),full_output=1,maxiter=MaxIter, disp=1)
    W1, b1, W2, b2 = _reshape_theta(out[0], visibleSize, hiddenSize, 'num_labels')
    np.savetxt('trained_W1.txt',W1)
    np.savetxt('trained_b1.txt',b1)
    np.savetxt('trained_W2.txt',W2)
    np.savetxt('trained_b2.txt',b2)
#train_nn()
#===================================================================================================================================================================

#--------------Display Theta1------------
def plot_Theta1(Theta1):
    print "Theta1.shape:", Theta1.shape
    n_units = Theta1.shape[0]
    n_plts = 1
    while n_plts * n_plts < n_units:
        n_plts += 1
    for i in tqdm(range(n_units)):
        plt.subplot(n_plts,n_plts,i+1)
        img = np.reshape(Theta1[i,1:],(8,8))
        img = img - np.mean(img)
#        print "i,||x||^2:", i, np.sum(img**2)
        img  = img / float(np.sqrt(np.sum(img**2)))
        plt.imshow(np.rot90(img),cmap='gray')
#        plt.imshow(np.rot90(img),cmap='gray',norm=LogNorm())
        plt.xlim([0,8])
        plt.ylim([0,8])
        plt.xticks([])
        plt.yticks([])
    plt.show()
#-----------------------------------


#--------plot W1------------
def plot_w1(images):
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
        img[:,:,0] = np.reshape(B[:,ind],(8,8))
        img[:,:,1] = np.reshape(C[:,ind],(8,8))
        img[:,:,2] = np.reshape(D[:,ind],(8,8))
        plt.imshow(img)#,norm=LogNorm())
        plt.xticks([])
        plt.yticks([])
    plt.show()

Theta1 = np.loadtxt('trained_W1.txt')
print "W1.shape:", Theta1.shape
plot_w1(np.dot(Theta1[:,1:],ZCAWhite).T)
#plot_Theta1(Theta1)
