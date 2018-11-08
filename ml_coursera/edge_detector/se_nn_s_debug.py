import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import scipy.optimize as op
import copy
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

#-----------loading data----------------
data = io.loadmat('IMAGES.mat')
print data.keys()
images = data['IMAGES']
print 'images.shape:', images.shape
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
    numpatches = 10000
    patches = np.zeros((patchsize*patchsize, numpatches))
#    img_id = np.random.choice(range(10),1)
#    image = images[:,:,img_id]
#    print "selected image id:", img_id
    for i in range(numpatches):
        img_id = np.random.choice(range(10),1)
        image = images[:,:,img_id]
        ind_x = np.random.choice(range(image.shape[0]-patchsize),1).astype(int)
        ind_y = np.random.choice(range(image.shape[0]-patchsize),1).astype(int)
        img = image[ind_x[0]:ind_x[0]+patchsize,ind_y[0]:ind_y[0]+patchsize].flatten()
        patches[:,i] = img
    return patches
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

#I was supposed to Squash data to [0.1, 0.9] since we use sigmoid as the activation function in the output layer by 
#"Truncate to +/-3 standard deviations and scale to -1 to 1" and "Rescale from [-1,1] to [0.1,0.9]"
##I did it!
#------------------------------------------------------------------------------------------------------------------------------

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
        ax1.imshow(img,norm=LogNorm(),cmap='gray')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
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
#    print "W1, W2, b1, b2, W1.shape, W2.shape, b1.shape, b2.shape:", W1, W2, b1, b2, W1.shape, W2.shape, b1.shape, b2.shape
    print "W1.shape, W2.shape, b1.shape, b2.shape:",  W1.shape, W2.shape, b1.shape, b2.shape
    # Convert weights and bias gradients to the vector form.
    # This step will "unroll" (flatten and concatenate together) all 
    # your parameters into a vector, which can then be used with minFunc. 
    #theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];
#    theta = np.concatenate((W1.flatten(),W2.flatten(),b1.flatten(),b2.flatten()))
    theta = np.concatenate((W1.flatten(),b1.flatten(),W2.flatten(),b2.flatten()))
    return theta
#theta = initializeParameters(hidden_layer_size, input_layer_size)
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

def _add_regCost(W1,W2,y,lambdaa):
    m = float(len(y))
    reg_j = 0
    reg_j += np.sum(W1[:,1:]**2)  # [:,1:]:we don't regularize the terms that correspond to the bias 
    reg_j += np.sum(W2[:,1:]**2)  # [:,1:]:we don't regularize the terms that correspond to the bias 
    reg_j = (lambdaa/(2.0)) * reg_j
    return reg_j

def _reshape_theta(theta, input_layer_size, hidden_layer_size, num_labels):
#    print "lens(theta, input_layer_size, hidden_layer_size, num_labels):", len(theta), input_layer_size, hidden_layer_size, num_labels
    #Theta1: (25,64+1), b1: (25,1); Theta2: (64,26), b2: (64,1), X: (5000,64+1)
#    print "theta.shape before reshaping:", theta.shape
    theta1 = np.reshape(theta[0:hidden_layer_size*(input_layer_size + 1)],(hidden_layer_size,input_layer_size + 1))
    print "theta1.shape:", theta1.shape
    B1 = np.reshape(theta[hidden_layer_size*(input_layer_size + 1):(hidden_layer_size*(input_layer_size + 1))+hidden_layer_size],(hidden_layer_size, 1))
    print "B1.shape:", B1.shape
    theta2 = np.reshape(theta[ len(theta1.flatten())+len(B1.flatten()):  \
			       len(theta1.flatten())+len(B1.flatten()) + (num_labels*(hidden_layer_size+1))],(num_labels, hidden_layer_size + 1))
    print "theta2.shape:", theta2.shape
    B2 = np.reshape(theta[len(theta1.flatten())+len(B1.flatten()) + (num_labels*(hidden_layer_size+1)):] ,(num_labels, 1))
    print "B2.shape:", B2.shape
    return theta1, B1, theta2, B2

def _add_KL_term(nn_params, sparsityParam, beta, X):
    W1, b1, W2, b2 = nn_params[0], nn_params[1], nn_params[2], nn_params[3]
    ro = float(sparsityParam)
    z2, z3, a1, a2, a3 = _h_theta2(W1, b1, W2, b2, X)
    ro_h = np.mean(a2,axis=0)  
    ind1, ind2 = (ro_h == 0) , (ro_h == 1) ; ro_h[ind1] = 1e-15 ; ro_h[ind2] = 1-1e-15
    kl_term = beta * np.sum( ( ro * np.log(ro/ro_h) ) + ( (1 - ro) * np.log((1-ro)/(1-ro_h)) ) )
    return kl_term

def sparseAutoencoderCost(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa, sparsityParam, beta, reshape_theta=True):
    if reshape_theta : W1, b1, W2, b2 = _reshape_theta(theta, input_layer_size, hidden_layer_size, num_labels)
    nn_params = [W1, b1, W2, b2]
    X = np.c_[np.ones(X.shape[0]),X]
    m = len(y)
    z2, z3, a1, a2, a3 = _h_theta2(W1, b1, W2, b2, X)
#    indd = np.random.choice(range(m),2)
#    print "indd, 0.5*np.sum((y[indd[0]-y[indd[1]])**2):",indd, 0.5*np.sum((y[indd[0]]-y[indd[1]])**2)
    a3_yy = np.array([np.sum((a3[i] - y[i])**2) for i in range(y.shape[0])])
#    print "min(a3), max(a3), min(y), max(y), min(X), max(X):"
#    print  np.min(a3), np.max(a3), np.min(y), np.max(y), np.min(X), np.max(X)		#0.0186312364048 0.980283400057 0.1 0.9 0.1 1.0
    J = (1.0/m) * np.sum(0.5*a3_yy)
    reg_J = _add_regCost(W1,W2,y,lambdaa)
    kl_J = _add_KL_term(nn_params, sparsityParam, beta, X)
    cost =  J + reg_J + kl_J
    return cost

def _h_theta2(W1, b1, W2, b2, X):                 #Theta1: (25,401), Theta2: (10,26), X: (5000,401)
    a1 = X.T
    B1 = np.array([b1[:,0] for i in range(X.shape[0])]) ; B2 = np.array([b2[:,0] for i in range(X.shape[0])])
    z2 = np.dot(W1,a1).T + B1; a2 = sigmoid(z2) ; a2 = np.c_[np.ones(a2.shape[0]),a2]
    z3 = np.dot(W2,a2.T).T + B2; a3 = sigmoid(z3)
    return z2, z3, a1, a2, a3

def nnBackPro(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa ,sparsityParam, beta, reshape_theta=True, debug=False):
    if reshape_theta : W1, b1, W2, b2 = _reshape_theta(theta, input_layer_size, hidden_layer_size, num_labels)
    nn_params = [W1,b1,W2,b2]
#    print "W1.shape, b1.shape, W2.shape, b2.shape:", W1.shape, b1.shape, W2.shape, b2.shape    		#(25, 65) (25, 1) (64, 26) (64, 1)
    X = np.c_[np.ones(X.shape[0]),X]
    m = len(y)
    ro = float(sparsityParam)
    z2, z3, a1, a2, a3 = _h_theta2(W1, b1, W2, b2, X)
    g_prime_z2 = sigmoidGradient(z2)
    g_prime_z3 = sigmoidGradient(z3)
    ro_H = np.mean(a2,axis=0)[1:] 
    ro_h = np.array([ro_H for i in range(a2.shape[0])])
    ind1, ind2 = (ro_h == 0) , (ro_h == 1)
    ro_h[ind1] = 1e-15
    ro_h[ind2] = 1-1e-15
    kl_term = beta * ( (float(1-ro)/(1-ro_h)) - (ro/ro_h) )
    delta3 = (a3 - y) * g_prime_z3
    print "(a3 - y).shape:", (a3 - y).shape
    print "g_prime_z2.shape ,g_prime_z3.shape, delta3.shape, kl_term.shape:", g_prime_z2.shape ,g_prime_z3.shape , delta3.shape, kl_term.shape
    delta2 = (np.dot(W2[:,1:].T,delta3.T).T + kl_term) * g_prime_z2
    DW1 = np.dot(delta2.T, a1.T)
    DW2 = np.dot(delta3.T, a2)

    DW1 = (1.0/m) * DW1
    DW2 = (1.0/m) * DW2
    DW1[:,1:] = DW1[:,1:] + (float(lambdaa)) * W1[:,1:]              				# reqularizing all output nodes and 1:all input nodes
    DW2[:,1:] = DW2[:,1:] + (float(lambdaa)) * W2[:,1:]
    DB1 = (1.0/m) * delta2
    DB2 = (1.0/m) * delta3
#    print "DW1.shape, DB1.shape, DW2.shape, DB2.shape: ", DW1.shape, DB1.shape, DW2.shape, DB2.shape							#(25, 65) (50, 26)
    DB1 = np.sum(DB1,axis=0)
    DB2 = np.sum(DB2,axis=0)
#    print "DW1.shape, DB1.shape, DW2.shape, DB2.shape: ", DW1.shape, DB1.shape, DW2.shape, DB2.shape							#(25, 65) (50, 26)
    D = []
    D.extend(DW1.flatten()); D.extend(DB1.flatten())
    D.extend(DW2.flatten()); D.extend(DB2.flatten())
    if debug: return DW1, DB1, DW2, DB2
    else: return np.array(D)

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
        loss1 = sparseAutoencoderCost(unrolled_theta - perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa, sparsityParam, beta, reshape_theta=True)
        loss2 = sparseAutoencoderCost(unrolled_theta + perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa, sparsityParam, beta, reshape_theta=True)
        numgrad[p] = (loss2 - loss1) / (2.0*e)
        perturb[p] = 0
    return numgrad
#====================================================================================================================================================================

input_layer_size = 8*8   	# number of input units 
hidden_layer_size = 25     	# number of hidden units 
sparsityParam = 0.01    	# desired average activation of the hidden units.
lambdaa = 0.0001     		# weight decay parameter       
beta = 3.0            		# weight of sparsity penalty term       

#lambdaa = 0.0
#beta = 0.0

#patches = np.loadtxt('patches2.txt',delimiter=',')
#patches = patches[:,::10]				#patches:(64,10000)
X = patches.T
y = patches.T
print "min(X), max(X):", np.min(X), np.max(X)
num_labels = X.shape[1]
num_data = X.shape[0]
MaxIter = 400

#-----------------Debugging Cost function and Gradient---------------------------------------------
debug = True
if debug:
    initial_Theta1 = debug_randInitializeWeights(hidden_layer_size, input_layer_size)
    initial_Theta2 = debug_randInitializeWeights(num_labels, hidden_layer_size)
    initial_b1 = debug_randInitializeWeights(hidden_layer_size, 0)
    initial_b2 = debug_randInitializeWeights(num_labels, 0)
    print " initial_Theta1.shape, initial_b1.shape, initial_Theta2.shape, initial_b2.shape", initial_Theta1.shape, initial_b1.shape, initial_Theta2.shape, initial_b2.shape
#    print " initial_Theta1, initial_b1, initial_Theta2, initial_b2", initial_Theta1, initial_b1, initial_Theta2, initial_b2
    initial_nn_params = [initial_Theta1, initial_b1, initial_Theta2, initial_b2]
    theta = []
    for i in range(len(initial_nn_params)): theta.extend(initial_nn_params[i].flatten())
#    X = debug_randInitializeWeights(num_data,input_layer_size - 1)
#    y  = 1 + np.mod(range(num_data), num_labels)
#    y = np.reshape(y,(y.shape[0],1))
#    X = debug_randInitializeWeights(len(y),input_layer_size - 1)
#    y  = 1 + np.mod(range(len(y)), num_labels)
#    y = np.reshape(y,(y.shape[0],1))
#    y = X
    print "debug X and y shapes:", X.shape, y.shape

    grad = nnBackPro(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa ,sparsityParam, beta, reshape_theta=True, debug=True)
    np.savetxt('W1_debug.txt',grad[0])
    np.savetxt('B1_debug.txt',grad[1])
    np.savetxt('W2_debug.txt',grad[2])
    np.savetxt('B2_debug.txt',grad[3])
#----check gradient against numerical gradient--
    numgrad = computeNumericalGradient(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa ,sparsityParam, beta, reshape_theta=True, debug=True)
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
    plt.plot(unrolled_grad - numgrad,'r.')
    plt.xlabel('Tetha');plt.ylabel('grad - numgrad')
    plt.show()
if debug: plot_num_gradient(grad,numgrad); quit()
#--------------------------------------------------------------------------------------------------


#===============Main training NN===================================================================================================================================
def train_nn():
#    initial_Theta1 = randInitializeWeights(hidden_layer_size, input_layer_size)
#    initial_Theta2 = randInitializeWeights(num_labels, hidden_layer_size)
#    initial_b1 = debug_randInitializeWeights(hidden_layer_size, 0)
#    initial_b2 = debug_randInitializeWeights(num_labels, 0)
#    nn_params = [initial_Theta1, initial_b1, initial_Theta2, initial_b2]
#    unrolled_theta2 = []
#    for i in range(len(nn_params)):
#        unrolled_theta2.extend(nn_params[i].flatten())
    unrolled_theta = initializeParameters(hidden_layer_size, input_layer_size)
#    print "X.shape, y.shape, unrolled_theta.size, unrolled_theta2.size:", X.shape, y.shape, len(unrolled_theta), len(unrolled_theta2)
    out = op.fmin_l_bfgs_b(sparseAutoencoderCost,unrolled_theta,nnBackPro,args=(input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa, sparsityParam, beta),maxfun=MaxIter, disp=1)
#    out = op.fmin_cg(sparseAutoencoderCost,unrolled_theta,nnBackPro,args=(input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa, sparsityParam, beta),full_output=1,maxiter=MaxIter, disp=1)
    Theta1, b1, Theta2, b2 = _reshape_theta(out[0], input_layer_size, hidden_layer_size, num_labels)
    np.savetxt('trained_theta1_10000.txt',Theta1)
    np.savetxt('trained_theta2_10000.txt',Theta2)
train_nn()
quit()
#===================================================================================================================================================================

#--------------Display Theta1------------
def plot_Theta1(Theta1):
    print "Theta1.shape:", Theta1.shape
    n_units = Theta1.shape[0]
    n_plts = 1
    while n_plts * n_plts < n_units:
        n_plts += 1
    for i in range(n_units):
        plt.subplot(n_plts,n_plts,i+1)
        img = np.reshape(Theta1[i,1:],(8,8))
	img = img - np.mean(img)
	print "i,||x||^2:", i, np.sum(img**2)
	img  = img / float(np.sqrt(np.sum(img**2)))
        plt.imshow(np.rot90(img),cmap='gray')
#        plt.imshow(np.rot90(img),cmap='gray',norm=LogNorm())
        plt.xlim([0,8])
        plt.ylim([0,8])
        plt.xticks([])
        plt.yticks([])
    plt.show()
#-----------------------------------

#--------------Display Theta2------------
def plot_Theta2(Theta2):
    print "Theta2.shape:", Theta2.shape
    n_units = Theta2.shape[0]
    n_plts = 1
    while n_plts * n_plts < n_units:
        n_plts += 1
    for i in range(n_units):
        plt.subplot(n_plts,n_plts,i+1)
        img = np.reshape(Theta2[i,1:],(5,5))
	img = img - np.mean(img)
	print "i,||x||^2:", i, np.sum(img**2)
	img  = img / float(np.sqrt(np.sum(img**2)))
#        plt.imshow(np.rot90(img),cmap='gray',norm=LogNorm())
        plt.imshow(np.rot90(img),cmap='gray')
#        plt.xlim([0,5])
#        plt.ylim([0,5])
        plt.xticks([])
        plt.yticks([])
    plt.savefig('theta2.png')
    plt.show()
#-----------------------------------


Theta1 = np.loadtxt('trained_theta1_10000.txt')
plot_Theta1(Theta1)
#Theta2 = np.loadtxt('trained_theta2_l1.0.txt')
#plot_Theta2(Theta2)
