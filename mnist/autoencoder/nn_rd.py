import os,sys
if len(sys.argv) < 4:
    print """\nThe script %s needs following inputs:
\t1. Size of hidden layer 1
\t2. Size of hidden layer 2
\t3. Size of hidden layer 3
    """ %sys.argv[0]
    quit()
import numpy as np
import scipy.optimize as op
import h5py
import time


hiddensize1 = int(sys.argv[1])
hiddensize2 = int(sys.argv[2])
hiddensize3 = int(sys.argv[3])

#--------load data------------------
images = np.array(h5py.File('train_images.h5').get('arr_0')).T / 255.
labels = np.array(h5py.File('train_labels.h5').get('arr_0'))
print "mnist images.shape, labels.shape:", images.shape, labels.shape
#-----------------------------------

#============= Stacked Autoencoder ==========================
#-----------initializeParameters--------------------------------
def initializeParameters(hidden_layer_size, input_layer_size):
    r  = np.sqrt(6) / np.sqrt(hidden_layer_size+input_layer_size+1);   # we'll choose weights uniformly from the interval [-r, r]
    W1 = np.random.random((hidden_layer_size, input_layer_size + 1)) * 2 * r - r;
    W2 = np.random.random((input_layer_size, hidden_layer_size + 1)) * 2 * r - r;
    b1 = np.zeros((hidden_layer_size, 1))
    b2 = np.zeros((input_layer_size, 1))
    print "W1.shape, W2.shape, b1.shape, b2.shape:",  W1.shape, W2.shape, b1.shape, b2.shape
    theta = np.concatenate((W1.flatten(),b1.flatten(),W2.flatten(),b2.flatten()))
    return theta
#---------------------------------------------------------------

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))
def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))
def randInitializeWeights(L_out, L_in):
    epsilon_init = 0.12                 # This number comes from: ( sqrt(6) / (sqrt{(# of layers before)+(# of layers after)}) )
    return np.random.random((L_out, 1 + L_in)) * 2 * epsilon_init - epsilon_init
def debug_randInitializeWeights(L_out, L_in):
    return np.reshape(np.sin(range(L_out * (1 + L_in))),(L_out, 1 + L_in))

def _add_regCost(W1,W2,y,lambdaa):
    m = float(len(y))
    reg_j = 0
    reg_j += np.sum(W1[:,1:]**2)  # [:,1:]:we don't regularize the terms that correspond to the bias 
    reg_j += np.sum(W2[:,1:]**2)  # [:,1:]:we don't regularize the terms that correspond to the bias 
    reg_j = (lambdaa/(2.0)) * reg_j
    return reg_j

def _add_KL_term(a2, sparsityParam, beta, X):
    ro = float(sparsityParam)
    ro_h = np.mean(a2,axis=0)  
    ind1, ind2 = (ro_h == 0) , (ro_h == 1) ; ro_h[ind1] = 1e-15 ; ro_h[ind2] = 1-1e-15
    kl_term = beta * np.sum( ( ro * np.log(ro/ro_h) ) + ( (1 - ro) * np.log((1-ro)/(1-ro_h)) ) )
    return kl_term

def _reshape_theta(theta, input_layer_size, hidden_layer_size):
    theta1 = np.reshape(theta[0:hidden_layer_size*(input_layer_size + 1)],(hidden_layer_size,input_layer_size + 1))
    B1 = np.reshape(theta[hidden_layer_size*(input_layer_size + 1):(hidden_layer_size*(input_layer_size + 1))+hidden_layer_size],(hidden_layer_size, 1))
    theta2 = np.reshape(theta[ len(theta1.flatten())+len(B1.flatten()):  \
			       len(theta1.flatten())+len(B1.flatten()) + (input_layer_size*(hidden_layer_size+1))],(input_layer_size, hidden_layer_size + 1))
    B2 = np.reshape(theta[len(theta1.flatten())+len(B1.flatten()) + (input_layer_size*(hidden_layer_size+1)):] ,(input_layer_size, 1))
    return theta1, B1, theta2, B2

def _h_theta2(W1, b1, W2, b2, X):                 #Theta1: (25,401), Theta2: (10,26), X: (5000,401)
    a1 = X.T
    B1 = np.array([b1[:,0] for i in range(X.shape[0])]) ; B2 = np.array([b2[:,0] for i in range(X.shape[0])])
    z2 = np.dot(W1,a1).T + B1; a2 = sigmoid(z2) ; a2 = np.c_[np.ones(a2.shape[0]),a2]
    z3 = np.dot(W2,a2.T).T + B2; a3 = sigmoid(z3)
    return z2, z3, a1, a2, a3

def sparseAutoencoderCost(theta, input_layer_size, hidden_layer_size, X, y, lambdaa, sparsityParam, beta, reshape_theta=True, debug=False):
    if reshape_theta : W1, b1, W2, b2 = _reshape_theta(theta, input_layer_size, hidden_layer_size)
    X = np.c_[np.ones(X.shape[0]),X]
    m = len(y)
    z2, z3, a1, a2, a3 = _h_theta2(W1, b1, W2, b2, X)
    a3_yy = np.array([np.sum((a3[i] - y[i])**2) for i in range(y.shape[0])])
    J = (1.0/m) * np.sum(0.5*a3_yy)
    reg_J = _add_regCost(W1,W2,y,lambdaa)
    kl_J = _add_KL_term(a2, sparsityParam, beta, X)
    cost =  J + reg_J + kl_J
  
    #backpro
    ro = float(sparsityParam)
    g_prime_z2 = a2[:,1:] * ( 1. - a2[:,1:])
    g_prime_z3 = a3 * ( 1. - a3 )
    ro_H = np.mean(a2,axis=0)[1:] 
    ro_h = np.array([ro_H for i in range(a2.shape[0])])
    ind1, ind2 = (ro_h == 0) , (ro_h == 1)
    ro_h[ind1] = 1e-15
    ro_h[ind2] = 1-1e-15
    kl_term = beta * ( (float(1-ro)/(1-ro_h)) - (ro/ro_h) )
    delta3 = (a3 - y) * g_prime_z3
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
    if debug: return cost, DW1, DB1, DW2, DB2
    else: return cost, np.array(D)
#============================================================


#===================== train layer 1 =====================
def train_nn1():
    #------parameters-----
    input_layer_size = images.shape[0]
    hidden_layer_size = hiddensize1
    X = images.T
    y = X
    MaxIter = 100 # 400
    sparsityParam = 0.1
    lambdaa = 1e-6 # 3e-3
    beta = 0.01
    #---------------------
    unrolled_theta = initializeParameters(hidden_layer_size, input_layer_size)
    print "\nstarting to train on X.shape: %s and y.shape: %s with input_layer_size: %d and hidden_layer_size:%d" %(X.shape, y.shape, input_layer_size, hidden_layer_size)
    print "lambda is:%f, beta is:%f, and sparsityParam is :%f" %(lambdaa, beta, sparsityParam)
    out = op.fmin_l_bfgs_b(sparseAutoencoderCost,unrolled_theta,fprime=None,args=(input_layer_size, hidden_layer_size, X, y, lambdaa, sparsityParam, beta),maxfun=MaxIter, disp=1)
    Theta1, b1, Theta2, b2 = _reshape_theta(out[0], input_layer_size, hidden_layer_size)
    np.savetxt('trained_W1_l1.txt',Theta1)
    np.savetxt('trained_b1_l1.txt',b1)
    np.savetxt('trained_W2_l1.txt',Theta2)
    np.savetxt('trained_b2_l1.txt',b2)
    return Theta1, b1, Theta2, b2
Theta1, b1, Theta2, b2 = train_nn1()
#=======================================================

#===================== train layer 2 =====================
def feedForwardAutoencoder(W1, b1, X):  # W1: (hiddensize,inputsize+1)  b1: (hiddensize?,1)
    a1 = X#.T
    B1 = np.array([b1[:,0] for i in range(X.shape[1])])
    z2 = np.dot(W1[:,1:],a1).T
    z2 = z2 + B1
    a2 = sigmoid(z2)
    return a2

def train_nn2():
    W1 = np.loadtxt('trained_W1_l1.txt')
    B1 = np.loadtxt('trained_b1_l1.txt')
    b1 = np.zeros((len(B1),1))
    b1[:,0] = B1
    #------parameters-----
    input_layer_size = W1.shape[0]
    hidden_layer_size = hiddensize2
    MaxIter = 100 # 400
    sparsityParam = 0.1
    lambdaa = 1e-6 # 3e-3
    beta = 0.01
    #---------------------
    #----feedforward------------
    x = feedForwardAutoencoder(W1, b1, images)
    X = x
    y = X
    #---------------------------
    unrolled_theta = initializeParameters(hidden_layer_size, input_layer_size)
    print "\nstarting to train on X.shape: %s and y.shape: %s with input_layer_size: %d and hidden_layer_size:%d" %(X.shape, y.shape, input_layer_size, hidden_layer_size)
    print "lambda is:%f, beta is:%f, and sparsityParam is :%f" %(lambdaa, beta, sparsityParam)
    out = op.fmin_l_bfgs_b(sparseAutoencoderCost,unrolled_theta,fprime=None,args=(input_layer_size, hidden_layer_size, X, y, lambdaa, sparsityParam, beta),maxfun=MaxIter, disp=1)
    Theta1, b1, Theta2, b2 = _reshape_theta(out[0], input_layer_size, hidden_layer_size)
    np.savetxt('trained_W1_l2.txt',Theta1)
    np.savetxt('trained_b1_l2.txt',b1)
    np.savetxt('trained_W2_l2.txt',Theta2)
    np.savetxt('trained_b2_l2.txt',b2)
    return Theta1, b1, Theta2, b2

Theta1, b1, Theta2, b2 = train_nn2()
#=======================================================

#===================== train layer 3 =====================
def train_nn3():
    W1 = np.loadtxt('trained_W1_l1.txt')
    B1 = np.loadtxt('trained_b1_l1.txt')
    b1 = np.zeros((len(B1),1))
    b1[:,0] = B1
    W2 = np.loadtxt('trained_W1_l2.txt')
    B2 = np.loadtxt('trained_b1_l2.txt')
    b2 = np.zeros((len(B2),1))
    b2[:,0] = B2
    #------parameters-----
    input_layer_size = W2.shape[0]
    hidden_layer_size = hiddensize3
    MaxIter = 100 # 400
    sparsityParam = 0.1
    lambdaa = 1e-6 # 3e-3
    beta = 0.01
    #---------------------
    #----feedforward------------
    x1 = feedForwardAutoencoder(W1, b1, images).T
    x2 = feedForwardAutoencoder(W2, b2, x1)
    X = x2
    y = X
    #---------------------------
    unrolled_theta = initializeParameters(hidden_layer_size, input_layer_size)
    print "\nstarting to train on X.shape: %s and y.shape: %s with input_layer_size: %d and hidden_layer_size:%d" %(X.shape, y.shape, input_layer_size, hidden_layer_size)
    print "lambda is:%f, beta is:%f, and sparsityParam is :%f" %(lambdaa, beta, sparsityParam)
    out = op.fmin_l_bfgs_b(sparseAutoencoderCost,unrolled_theta,fprime=None,args=(input_layer_size, hidden_layer_size, X, y, lambdaa, sparsityParam, beta),maxfun=MaxIter, disp=1)
    Theta1, b1, Theta2, b2 = _reshape_theta(out[0], input_layer_size, hidden_layer_size)
    np.savetxt('trained_W1_l3.txt',Theta1)
    np.savetxt('trained_b1_l3.txt',b1)
    np.savetxt('trained_W2_l3.txt',Theta2)
    np.savetxt('trained_b2_l3.txt',b2)
    return Theta1, b1, Theta2, b2

Theta1, b1, Theta2, b2 = train_nn3()
#=========================================================

#===================== train layer 4 =====================
def train_nn4():
    W1 = np.loadtxt('trained_W1_l1.txt')
    B1 = np.loadtxt('trained_b1_l1.txt')
    b1 = np.zeros((len(B1),1))
    b1[:,0] = B1
    W2 = np.loadtxt('trained_W1_l2.txt')
    B2 = np.loadtxt('trained_b1_l2.txt')
    b2 = np.zeros((len(B2),1))
    b2[:,0] = B2
    W3 = np.loadtxt('trained_W1_l3.txt')
    B3 = np.loadtxt('trained_b1_l3.txt')
    b3 = np.zeros((len(B3),1))
    b3[:,0] = B3
    #------parameters-----
    input_layer_size = W3.shape[0]
    hidden_layer_size = 2
    MaxIter = 100 # 400
    sparsityParam = 0.1
    lambdaa = 1e-6 # 3e-3
    beta = 0.01
    #---------------------
    #----feedforward------------
    x1 = feedForwardAutoencoder(W1, b1, images).T
    x2 = feedForwardAutoencoder(W2, b2, x1).T
    x3 = feedForwardAutoencoder(W3, b3, x2)
    X = x3
    y = X
    #---------------------------
    unrolled_theta = initializeParameters(hidden_layer_size, input_layer_size)
    print "\nstarting to train on X.shape: %s and y.shape: %s with input_layer_size: %d and hidden_layer_size:%d" %(X.shape, y.shape, input_layer_size, hidden_layer_size)
    print "lambda is:%f, beta is:%f, and sparsityParam is :%f" %(lambdaa, beta, sparsityParam)
    out = op.fmin_l_bfgs_b(sparseAutoencoderCost,unrolled_theta,fprime=None,args=(input_layer_size, hidden_layer_size, X, y, lambdaa, sparsityParam, beta),maxfun=MaxIter, disp=1)
    Theta1, b1, Theta2, b2 = _reshape_theta(out[0], input_layer_size, hidden_layer_size)
    np.savetxt('trained_W1_l4.txt',Theta1)
    np.savetxt('trained_b1_l4.txt',b1)
    np.savetxt('trained_W2_l4.txt',Theta2)
    np.savetxt('trained_b2_l4.txt',b2)
    return Theta1, b1, Theta2, b2

Theta1, b1, Theta2, b2 = train_nn4()
#=========================================================

