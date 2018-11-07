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
    for i in range(numpatches):
    	img_id = np.random.choice(range(10),1)
    	image = images[:,:,img_id] 
	ind_x = np.random.choice(range(image.shape[0]-patchsize),1)   
	ind_y = np.random.choice(range(image.shape[0]-patchsize),1) 
        img = image[ind_x:ind_x+patchsize,ind_y:ind_y+patchsize].flatten()
	patches[:,i] = img  
    return patches
patches = sampleIMAGES()
np.savetxt('patches.txt',patches)
patches2 = np.array([patches[:,i] - np.mean(patches[:,i]) for i in range(patches.shape[1]) ]).T
pstd3 = 3 * np.std(patches2.flatten())
print "min max. mini-av max-av, std3:", np.min(patches), np.max(patches), np.min(patches2), np.max(patches2), pstd3
patches = (patches2 + abs(np.min(patches2))) 
patches = patches / np.max(patches)
print "min, max:", np.min(patches), np.max(patches)

#I was supposed to Squash data to [0.1, 0.9] since we use sigmoid as the activation function in the output layer by 
#"Truncate to +/-3 standard deviations and scale to -1 to 1" and "Rescale from [-1,1] to [0.1,0.9]"
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
  #  	ax1.set_aspect('equal')
    plt.show()
#display_network(patches,num=200) 
#----------------------------------------------------------------------------

#-----------initializeParameters--------------------------------
def initializeParameters(hidden_layer_size, input_layer_size):
    #Initialize parameters randomly based on layer sizes.
    r  = np.sqrt(6) / np.sqrt(hidden_layer_size+input_layer_size+1);   # we'll choose weights uniformly from the interval [-r, r]
    W1 = np.random.random((hidden_layer_size, input_layer_size)) * 2 * r - r;
    W2 = np.random.random((input_layer_size, hidden_layer_size)) * 2 * r - r;

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

def _h_theta(nn_params, X):                 #E.g. Theta1: (25,401), Theta2: (10,26), X: (5000,401)
    for i in range(len(nn_params)):
        if i == 0:  zz = np.dot(nn_params[i],X.T).T ; aa = sigmoid(zz)
        else: aa = np.c_[np.ones(aa.shape[0]),aa]; zz = np.dot(nn_params[i],aa.T).T ; aa = sigmoid(zz)
    a3 = aa                            
    return a3

def _add_regCost(nn_params,y,lambdaa):
    m = float(len(y))
    reg_j = 0
    for i in range(len(nn_params)):
        reg_j += np.sum(nn_params[i][:,1:]**2)  # [:,1:]:we don't regularize the terms that correspond to the bias 
    reg_j = (lambdaa/(2.0 * m)) * reg_j
    return reg_j

def _add_KL_term(nn_params, sparsityParam, beta, X):
    ro = float(sparsityParam)
    W1 = nn_params[0]
    z2 = np.dot(W1,X.T).T
    a2 = sigmoid(z2)
    ro_h = np.mean(a2,axis=0)  
    ind1, ind2 = (ro_h == 0) , (ro_h == 1)
    ro_h[ind1] = 1e-10
    ro_h[ind2] = 1-1e-10
#    print "a2.shape, ro_h.shape, theta1.shape:", a2.shape, ro_h.shape, nn_params[0].shape			# (100, 25) (25,) (25, 65)
    kl_term = beta * np.sum( ( ro * np.log(ro/ro_h) ) + ( (1 - ro) * np.log((1-ro)/(1-ro_h)) ) )
    return kl_term

def _reshape_theta(theta, input_layer_size, hidden_layer_size, num_labels):
    #Theta1: (25,401), Theta2: (10,26), X: (5000,401)
    theta1 = np.reshape(theta[0:hidden_layer_size*(input_layer_size + 1)],(hidden_layer_size,input_layer_size + 1))
    theta2 = np.reshape(theta[hidden_layer_size*(input_layer_size + 1):] ,(num_labels, hidden_layer_size + 1))
    return [theta1, theta2]
def _fix_y(y,num_labels):
    yy = np.zeros((len(y),num_labels))
    for i in range(len(y)):
        yy[i] = 1
    if 0 :print y[2080], yy[2080]
    return yy

def sparseAutoencoderCost(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa, sparsityParam, beta, reshape_theta=True):
    num_labels = X.shape[0] 
    if reshape_theta : nn_params = _reshape_theta(theta, input_layer_size, hidden_layer_size, num_labels)
    cost = 0
    X = np.c_[np.ones(X.shape[0]),X]
    y = range(X.shape[0])
    yy = _fix_y(y,num_labels)
    m = len(y)
    h_theta = _h_theta(nn_params, X)
    J = 0
    num_labels = X.shape[0]
    for i in range(num_labels):
	if [0] in h_theta[i]: print "h_theta[i] has zeros at::", i
	if [1] in h_theta[i]: print "h_theta[i] has ones at::", i
        J += (-1.0/m) * np.sum(yy[i]*np.log(h_theta[i]) + (1-yy[i])*np.log(1-h_theta[i]))
    reg_J = _add_regCost(nn_params,yy,lambdaa)
    kl_J = _add_KL_term(nn_params, sparsityParam, beta, X)
    cost =  J + reg_J + kl_J
    return cost

def nnBackPro(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa ,sparsityParam, beta, reshape_theta=True, debug=False):
    if reshape_theta : nn_params = _reshape_theta(theta, input_layer_size, hidden_layer_size, num_labels)
    X = np.c_[np.ones(X.shape[0]),X]
    m = len(y)
    yy = _fix_y(y,num_labels)
    D1, D2 = 0, 0
    ro = float(sparsityParam)
    def __get_ro_h(W1,X):
    	z2 = np.dot(W1,X.T).T
    	a2 = sigmoid(z2)
    	ro_h = np.mean(a2,axis=0) 
	ro_H = np.array([ro_h for i in range(a2.shape[0])])
	return ro_H.T
    ro_h = __get_ro_h(nn_params[0],X)
    a1 = X
    z2 = np.dot(nn_params[0],a1.T)
    a2 = sigmoid(z2)
    a2 = np.concatenate((np.ones((1,a2.shape[1])),a2))
    z3 = np.dot(nn_params[1],a2)
    a3 = sigmoid(z3)
    ind1, ind2 = (ro_h == 0) , (ro_h == 1)
    ro_h[ind1] = 1e-10
    ro_h[ind2] = 1-1e-10
    kl_term = beta * ( (float(1-ro)/(1-ro_h)) - (ro/ro_h) )
    delta3 = a3 - yy
    g_prime_z2 = sigmoidGradient(z2)
    delta2_term1 =  (np.dot(nn_params[1][:,1:].T,delta3) + kl_term)
    delta2 = delta2_term1 * g_prime_z2	
    D1 = np.dot(delta2, a1)
    D2 = np.dot(delta3, a2.T)
    D1 = (1.0/m) * D1
    D2 = (1.0/m) * D2
    D1[:,1:] = D1[:,1:] + (float(lambdaa)/m) * nn_params[0][:,1:]              				# reqularizing all output nodes and 1:all input nodes
    D2[:,1:] = D2[:,1:] + (float(lambdaa)/m) * nn_params[1][:,1:]
    D = []
    D.extend(D1.flatten())
    D.extend(D2.flatten())
    if debug: return D1, D2
    else: return np.array(D)

def computeNumericalGradient(costFunc, nn_params):
    unrolled_theta = []
    for i in range(len(nn_params)):
        unrolled_theta.extend(nn_params[i].flatten())
    numgrad = np.zeros(len(unrolled_theta))
    perturb = np.zeros(len(unrolled_theta))
    print "numgrad.shape, perturb.shape:", numgrad.shape, perturb.shape
    e = 1e-4;
    for p in range(len(unrolled_theta)):
        if p%1000 == 0: print p, "percent completed:", p*100/len(unrolled_theta)
        perturb[p] = e
        loss1 = costFunc(unrolled_theta - perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa, sparsityParam, beta, reshape_theta=True)
        loss2 = costFunc(unrolled_theta + perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa, sparsityParam, beta, reshape_theta=True)
        numgrad[p] = (loss2 - loss1) / (2.0*e)
        perturb[p] = 0
    return numgrad
#====================================================================================================================================================================

input_layer_size = 8*8   	# number of input units 
hidden_layer_size = 25     	# number of hidden units 
sparsityParam = 0.01    	# desired average activation of the hidden units.
lambdaa = 0.0001     		# weight decay parameter       
beta = 3.0            		# weight of sparsity penalty term       

patches = np.loadtxt('patches2.txt',delimiter=',')
#patches = patches[:,::2]				#patches:(64,10000)
X = patches.T
y = patches.T
num_labels = X.shape[0]
num_data = X.shape[0]
MaxIter = 40

#-----------------Debugging Cost function and Gradient---------------------------------------------
debug = False
if debug:
    initial_Theta1 = debug_randInitializeWeights(hidden_layer_size, input_layer_size)
    initial_Theta2 = debug_randInitializeWeights(num_labels, hidden_layer_size)
    print " initial_Theta1.shape, initial_Theta2.shape:", initial_Theta1.shape, initial_Theta2.shape
    initial_nn_params = [initial_Theta1, initial_Theta2]
    theta = []
    for i in range(len(initial_nn_params)): theta.extend(initial_nn_params[i].flatten())
    X = debug_randInitializeWeights(num_data,input_layer_size - 1)
    y  = 1 + np.mod(range(num_data), num_labels)
    y = np.reshape(y,(y.shape[0],1))
    print "debug X and y shapes:", X.shape, y.shape
    grad = nnBackPro(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa ,sparsityParam, beta, reshape_theta=True, debug=True)
    np.savetxt('grad1_debug.txt',grad[0])
    np.savetxt('grad2_debug.txt',grad[1])
#----check gradient against numerical gradient--
    numgrad = computeNumericalGradient(sparseAutoencoderCost,initial_nn_params)
    np.savetxt('num_grad_unrolled_debug.txt',numgrad)
#    print grad[0][0:10], numgrad[0:10]
def plot_num_gradient(grad,numgrad):
    unrolled_grad = []
    for i in range(len(grad)):
	unrolled_grad.extend(grad[i].flatten())
    plt.plot(unrolled_grad - numgrad,'r.')
    plt.xlabel('Tetha');plt.ylabel('grad - numgrad')
    plt.show()
if debug: plot_num_gradient(grad,numgrad)
#--------------------------------------------------------------------------------------------------


#===============Main training NN===================================================================================================================================
def train_nn():
    initial_Theta1 = randInitializeWeights(hidden_layer_size, input_layer_size)
    initial_Theta2 = randInitializeWeights(num_labels, hidden_layer_size)
    nn_params = [initial_Theta1, initial_Theta2]
    unrolled_theta = []
    for i in range(len(nn_params)):
        unrolled_theta.extend(nn_params[i].flatten())
    print "X.shape, y.shape:", X.shape, y.shape
    out = op.fmin_l_bfgs_b(sparseAutoencoderCost,unrolled_theta,nnBackPro,args=(input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa, sparsityParam, beta),maxfun=MaxIter, disp=1)
    Theta1, Theta2 = _reshape_theta(out[0], input_layer_size, hidden_layer_size, num_labels)
    np.savetxt('trained_theta1_10000.txt',Theta1)
    np.savetxt('trained_theta2_10000.txt',Theta2)
#train_nn()
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
        plt.imshow(np.rot90(img),cmap='gray',norm=LogNorm())
        plt.xlim([0,5])
        plt.ylim([0,5])
        plt.xticks([])
        plt.yticks([])
    plt.show()
#-----------------------------------


Theta1 = np.loadtxt('trained_theta1_10000.txt')
plot_Theta1(Theta1)
#Theta2 = np.loadtxt('trained_theta2_l1.0.txt')
#plot_Theta2(Theta2)
