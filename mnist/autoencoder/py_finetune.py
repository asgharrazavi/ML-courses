import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import scipy.optimize as op
import copy
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import h5py
import mdtraj.io as ioo

#--------load data------------------
images = np.array(h5py.File('train_images.h5').get('arr_0')).T / 255.
labels = np.array(h5py.File('train_labels.h5').get('arr_0'))
print "mnist images.shape, labels.shape:", images.shape, labels.shape
#-----------------------------------

#-------------------load stacked theta-----------------------
def load_thetas():
    W1_l1 = np.loadtxt('trained_W1_l1.txt')
    b1_l1 = np.loadtxt('trained_b1_l1.txt')
    W2_l1 = np.loadtxt('trained_W2_l1.txt')
    b2_l1 = np.loadtxt('trained_b2_l1.txt')
    W1_l2 = np.loadtxt('trained_W1_l2.txt')
    b1_l2 = np.loadtxt('trained_b1_l2.txt')
    W2_l2 = np.loadtxt('trained_W2_l2.txt')
    b2_l2 = np.loadtxt('trained_b2_l2.txt')
    W1_l3 = np.loadtxt('trained_W1_l3.txt')
    b1_l3 = np.loadtxt('trained_b1_l3.txt')
    W2_l3 = np.loadtxt('trained_W2_l3.txt')
    b2_l3 = np.loadtxt('trained_b2_l3.txt')
    W1_l4 = np.loadtxt('trained_W1_l4.txt')
    b1_l4 = np.loadtxt('trained_b1_l4.txt')
    W2_l4 = np.loadtxt('trained_W2_l4.txt')
    b2_l4 = np.loadtxt('trained_b2_l4.txt')
    print "W1_l1, b1_l1, W2_l1, b2_l1:", W1_l1.shape, b1_l1.shape, W2_l1.shape, b2_l1.shape
    print "W1_l2, b1_l2, W2_l2, b2_l2:", W1_l2.shape, b1_l2.shape, W2_l2.shape, b2_l2.shape
    print "W1_l3, b1_l3, W2_l3, b2_l3:", W1_l3.shape, b1_l3.shape, W2_l3.shape, b2_l3.shape
    print "W1_l4, b1_l4, W2_l4, b2_l4:", W1_l4.shape, b1_l4.shape, W2_l4.shape, b2_l4.shape
    return W1_l1[:,1:], b1_l1, W1_l2[:,1:], b1_l2, W1_l3[:,1:], b1_l3, W1_l4[:,1:], b1_l4
#------------------------------------------------------------

#======================fine tuning===========================
#============================================================sparseAutoencoderCost===================================================================================
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))
def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def _reshape_theta(theta,w1_shape,b1_shape, w2_shape,b2_shape, w3_shape,b3_shape, w4_shape,b4_shape):
    w11, b11, w22, b22, w33, b33, w44, b44 = w1_shape[0]*w1_shape[1], b1_shape[0], w2_shape[0]*w2_shape[1], b2_shape[0], w3_shape[0]*w3_shape[1], b3_shape[0], w4_shape[0]*w4_shape[1], b4_shape[0] 
    w1 = np.reshape(theta[0:w11],w1_shape)
    b1 = np.reshape(theta[w11 : w11+b11],(b1_shape[0],1))
    w2 = np.reshape(theta[w11+b11 : w11+b11+w22],w2_shape)
    b2 = np.reshape(theta[w11+b11+w22 : w11+b11+w22+b22],(b2_shape[0],1))
    w3 = np.reshape(theta[w11+b11+w22+b22 : w11+b11+w22+b22+w33],w3_shape)
    b3 = np.reshape(theta[w11+b11+w22+b22+w33 : w11+b11+w22+b22+w33+b33],(b3_shape[0],1))
    w4 = np.reshape(theta[w11+b11+w22+b22+w33+b33 : w11+b11+w22+b22+w33+b33+w44],w4_shape)
    b4 = np.reshape(theta[w11+b11+w22+b22+w33+b33+w44 : w11+b11+w22+b22+w33+b33+w44+b44],(b4_shape[0],1))
    return w1, b1, w2, b2, w3, b3, w4, b4

def reshape_theta(theta,shapes):
    w1_shape,b1_shape, w2_shape,b2_shape, w3_shape,b3_shape, w4_shape,b4_shape,w5_shape,b5_shape, w6_shape,b6_shape, w7_shape,b7_shape, w8_shape,b8_shape = shapes
    theta1_len = 0
    for i in range(len(shapes)/2): theta1_len += shapes[i][0]*shapes[i][1]
    W1, b1, W2, b2, W3, b3, W4, b4 = _reshape_theta(theta[0:theta1_len],w1_shape,b1_shape, w2_shape,b2_shape, w3_shape,b3_shape, w4_shape,b4_shape)
    W5, b5, W6, b6, W7, b7, W8, b8 = _reshape_theta(theta[theta1_len:],w5_shape,b5_shape, w6_shape,b6_shape, w7_shape,b7_shape, w8_shape,b8_shape)
    return W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8, b8

def _h_theta2(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8, b8 ,X):                 #Theta1: (25,401), Theta2: (10,26), X: (5000,401)
    a1 = X
    B1 = np.array([b1[:,0] for i in range(X.shape[0])]) ; B2 = np.array([b2[:,0] for i in range(X.shape[0])])
    B3 = np.array([b3[:,0] for i in range(X.shape[0])]) ; B4 = np.array([b4[:,0] for i in range(X.shape[0])])
    B5 = np.array([b5[:,0] for i in range(X.shape[0])]) ; B6 = np.array([b6[:,0] for i in range(X.shape[0])])
    B7 = np.array([b7[:,0] for i in range(X.shape[0])]) ; B8 = np.array([b8[:,0] for i in range(X.shape[0])])
#    print "B1, B2, B3, B4, B5, B6, B7, B8:", B1.shape, B2.shape, B3.shape, B4.shape, B5.shape, B6.shape, B7.shape, B8.shape
    z2 = np.dot(W1,a1.T).T + B1; a2 = sigmoid(z2) 
    z3 = np.dot(W2,a2.T).T + B2; a3 = sigmoid(z3) 
    z4 = np.dot(W3,a3.T).T + B3; a4 = sigmoid(z4) 
    z5 = np.dot(W4,a4.T).T + B4; a5 = sigmoid(z5) 
#    print "shapes: z2, z3, z4, z5, a1, a2, a3, a4, a5 ", z2.shape, z3.shape, z4.shape, z5.shape, a1.shape, a2.shape, a3.shape, a4.shape, a5.shape
    z6 = np.dot(W5,a5.T).T + B5; a6 = sigmoid(z6) 
    z7 = np.dot(W6,a6.T).T + B6; a7 = sigmoid(z7) 
    z8 = np.dot(W7,a7.T).T + B7; a8 = sigmoid(z8) 
    z9 = np.dot(W8,a8.T).T + B8; a9 = sigmoid(z9) 
    return z2, z3, z4, z5, z6, z7, z8, z9, a1, a2, a3, a4, a5, a6, a7, a8, a9

def _add_regCost(W8, W7 ,y,lambdaa):
    m = float(len(y))
    reg_j = 0
    reg_j += np.sum(W8**2)  
#    reg_j += np.sum(W7**2)  
    reg_j = (lambdaa/(2.0)) * reg_j
    return reg_j

def sparseAutoencoderCost(theta, shapes, x, lambdaa ,sparsityParam, beta, do_reshape_theta=True, debug=False):
    if do_reshape_theta : 
	W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8, b8 = reshape_theta(theta, shapes)
    m = len(x)
    X = x 
    z2, z3, z4, z5, z6, z7, z8, z9, a1, a2, a3, a4, a5, a6, a7, a8, a9 = _h_theta2(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8, b8 ,X)
  
    #cost
    x_y = np.array([np.sum((a9[i] - x[i])**2) for i in range(a9.shape[0])])
    cost = (1.0/m) * np.sum(0.5 * x_y)
    cost += _add_regCost(W8,W7, x,lambdaa) 
#    print cost

    #gradiant
#    delta9 = -1 * (a9 - x) * a9 * (1 - a9)
    delta9 = 1 * (a9 - x) * a9 * (1 - a9)
    delta8 = (np.dot(W8[:,:].T,delta9.T).T ) * a8[:,:] * (1 - a8[:,:])
    delta7 = (np.dot(W7[:,:].T,delta8.T).T ) * a7[:,:] * (1 - a7[:,:])
    delta6 = (np.dot(W6[:,:].T,delta7.T).T ) * a6[:,:] * (1 - a6[:,:])
    delta5 = (np.dot(W5[:,:].T,delta6.T).T ) * a5[:,:] * (1 - a5[:,:])
    delta4 = (np.dot(W4[:,:].T,delta5.T).T ) * a4[:,:] * (1 - a4[:,:])
    delta3 = (np.dot(W3[:,:].T,delta4.T).T ) * a3[:,:] * (1 - a3[:,:])
    delta2 = (np.dot(W2[:,:].T,delta3.T).T ) * a2[:,:] * (1 - a2[:,:])
    delta1 = (np.dot(W1[:,:].T,delta2.T).T ) * a1[:,:] * (1 - a1[:,:])

    DW1 = (1.0/m) * np.dot(delta2.T, a1) #+ lambdaa * W1
    DW2 = (1.0/m) * np.dot(delta3.T, a2) #+ lambdaa * W2
    DW3 = (1.0/m) * np.dot(delta4.T, a3) #+ lambdaa * W3
    DW4 = (1.0/m) * np.dot(delta5.T, a4) #+ lambdaa * W4
    DW5 = (1.0/m) * np.dot(delta6.T, a5) #+ lambdaa * W5
    DW6 = (1.0/m) * np.dot(delta7.T, a6) #+ lambdaa * W6
    DW7 = (1.0/m) * np.dot(delta8.T, a7) #+ lambdaa * W7
    DW8 = (1.0/m) * np.dot(delta9.T, a8) + lambdaa * W8

    DB1 = (1.0/m) * delta2
    DB2 = (1.0/m) * delta3
    DB3 = (1.0/m) * delta4
    DB4 = (1.0/m) * delta5
    DB5 = (1.0/m) * delta6
    DB6 = (1.0/m) * delta7
    DB7 = (1.0/m) * delta8
    DB8 = (1.0/m) * delta9
    DB1 = np.sum(DB1,axis=0)
    DB2 = np.sum(DB2,axis=0)
    DB3 = np.sum(DB3,axis=0)
    DB4 = np.sum(DB4,axis=0)
    DB5 = np.sum(DB5,axis=0)
    DB6 = np.sum(DB6,axis=0)
    DB7 = np.sum(DB7,axis=0)
    DB8 = np.sum(DB8,axis=0)

    D = []
    D.extend(DW1.flatten()); D.extend(DB1.flatten())
    D.extend(DW2.flatten()); D.extend(DB2.flatten())
    D.extend(DW3.flatten()); D.extend(DB3.flatten())
    D.extend(DW4.flatten()); D.extend(DB4.flatten())
    D.extend(DW5.flatten()); D.extend(DB5.flatten())
    D.extend(DW6.flatten()); D.extend(DB6.flatten())
    D.extend(DW7.flatten()); D.extend(DB7.flatten())
    D.extend(DW8.flatten()); D.extend(DB8.flatten())
    if debug: return cost, DW1, DB1, DW2, DB2, DW3, DB3, DW4, DB4, DW5, DB5, DW6, DB6, DW7, DB7, DW8, DB8
    else: return cost, np.array(D)

def computeNumericalGradient(theta, shapes, X, lambdaa ,sparsityParam, beta, do_reshape_theta=True, debug=True):
    unrolled_theta = np.array(theta)
    numgrad = np.zeros(len(unrolled_theta))
    perturb = np.zeros(len(unrolled_theta))
    print "numgrad.shape, perturb.shape:", numgrad.shape, perturb.shape
    e = 1e-4;
    for p in range(len(unrolled_theta)):
        if p%500 == 0: print p, "percent completed:", p*100/len(unrolled_theta)
        perturb[p] = e
# 	print "len(unrolled_theta - perturb):", len(unrolled_theta - perturb)
        loss1 = sparseAutoencoderCost(theta - perturb, shapes, X, lambdaa ,sparsityParam, beta, do_reshape_theta=True, debug=False)[0]
        loss2 = sparseAutoencoderCost(theta + perturb, shapes, X, lambdaa ,sparsityParam, beta, do_reshape_theta=True, debug=False)[0]
        numgrad[p] = (loss2 - loss1) / (2.0*e)
        perturb[p] = 0
    return numgrad
#====================================================================================================================================================================

#-----------------Debugging Cost function and Gradient---------------------------------------------
debug = False
if debug:
    inputSize = 3;
    hiddenSize1 = 35;
    hiddenSize2 = 15;
    hiddenSize3 = 5;
    hiddenSize4 = 2;
    lambdaa = 0.0;
    X = np.random.random((inputSize, 10)).T
#    y = np.array([ 0, 1, 0, 1, 0, 1, 1, 0, 1, 0 ])
    w1 = np.random.random((hiddenSize1,inputSize +0))
    w2 = np.random.random((hiddenSize2,hiddenSize1+0))
    w3 = np.random.random((hiddenSize3,hiddenSize2+0))
    w4 = np.random.random((hiddenSize4,hiddenSize3+0))
#    b1 = np.random.random((hiddenSize1,1))
#    b2 = np.random.random((hiddenSize2,1))
#    b3 = np.random.random((hiddenSize3,1))
#    b4 = np.random.random((hiddenSize4,1))
    b1 = np.zeros((hiddenSize1,1))
    b2 = np.zeros((hiddenSize2,1))
    b3 = np.zeros((hiddenSize3,1))
    b4 = np.zeros((hiddenSize4,1))

    w5, w6, w7, w8 = w4.T, w3.T, w2.T, w1.T
    b5, b6, b7, b8 = b3, b2, b1, np.ones((inputSize,1))
    initial_nn_params = [w1,b1, w2,b2, w3,b3, w4,b4,   w5,b5, w6,b6, w7,b7, w8,b8]
    theta = []
    for i in range(len(initial_nn_params)): theta.extend(initial_nn_params[i].flatten())
#    print "debug n_samples, inputsize, hidensize, n_labels:", X.shape[0], inputSize, hiddenSize, 2
    print "debug X shape:", X.shape
    print "debug sizes, input, L1, L2, L3, L4:", inputSize, hiddenSize1, hiddenSize2, hiddenSize3, hiddenSize4

    lambdaa, sparsityParam, beta = 1, 0.1, 0.1 #1, 0.1, 0.1
    
    shapes = [w1.shape, b1.shape, w2.shape, b2.shape, w3.shape,b3.shape, w4.shape,b4.shape,\
	      w5.shape, b5.shape, w6.shape, b6.shape, w7.shape,b7.shape, w8.shape,b8.shape,]

    print "debug shapes: w1, b1, w2, b2, w3, b3, w4, b4:", w1.shape, b1.shape, w2.shape, b2.shape, w3.shape, b3.shape, w4.shape, b4.shape
    print "debug shapes: w5, b5, w6, b6, w7, b7, w8, b8:", w5.shape, b5.shape, w6.shape, b6.shape, w7.shape, b7.shape, w8.shape, b8.shape
    grad = sparseAutoencoderCost(theta, shapes, X, lambdaa ,sparsityParam, beta, do_reshape_theta=True, debug=False)
#    np.savetxt('W1_debug.txt',grad[1])
#    np.savetxt('B1_debug.txt',grad[2])
#    np.savetxt('W2_debug.txt',grad[3])
#    np.savetxt('B2_debug.txt',grad[4])
#    np.savetxt('theta_soft_debug.txt',grad[5])
#----check gradient against numerical gradient--
    numgrad = computeNumericalGradient(theta, shapes, X, lambdaa ,sparsityParam, beta, do_reshape_theta=True, debug=False)
#    np.savetxt('num_grad_unrolled_debug.txt',numgrad)
def plot_num_gradient(grad,numgrad):
    unrolled_grad = grad[1]
#    unrolled_grad.extend(grad[1].flatten())
#    unrolled_grad.extend(grad[2].flatten())
#    unrolled_grad.extend(grad[3].flatten())
#    unrolled_grad.extend(grad[4].flatten())
#    unrolled_grad.extend(grad[5].flatten())
    plt.plot(unrolled_grad - numgrad,'r.')
    print "total diff:", np.sum(abs(unrolled_grad - numgrad))
    print "average diff:", np.mean(abs(unrolled_grad - numgrad))
    plt.xlabel('Tetha');plt.ylabel('grad - numgrad')
#    plt.savefig('grad_check.png')
    plt.show()
if debug: plot_num_gradient(grad,numgrad) ; quit()
#--------------------------------------------------------------------------------------------------


#-general parameters-
inputSize = 28 * 28
numClasses = 10
hiddenSizeL1 = 200
hiddenSizeL2 = 200
sparsityParam = 0.1
beta = 0.1
X = images[:,::10].T
#X = images.T
y = labels
#--------------------

def train_stackedAE():
    w1, b1, w2, b2, w3, b3, w4, b4 = load_thetas()
    b1, b2, b3, b4 = np.reshape(b1,(len(b1),1)), np.reshape(b2,(len(b2),1)), np.reshape(b3,(len(b3),1)), np.reshape(b4,(len(b4),1))
    print "w1, b1, w2, b2, w3, b3, w4, b4 :", w1.shape, b1.shape, w2.shape, b2.shape, w3.shape, b3.shape, w4.shape, b4.shape
   
    w5, w6, w7, w8 = w4.T, w3.T, w2.T, w1.T
    b5, b6, b7, b8 = b3, b2, b1, np.ones((inputSize,1))
    initial_nn_params = [w1,b1, w2,b2, w3,b3, w4,b4,   w5,b5, w6,b6, w7,b7, w8,b8]

    shapes = [w1.shape, b1.shape, w2.shape, b2.shape, w3.shape,b3.shape, w4.shape,b4.shape,\
              w5.shape, b5.shape, w6.shape, b6.shape, w7.shape,b7.shape, w8.shape,b8.shape]

    theta = []

    sum1, sum2 = 0, 0
    for i in range(8): sum1 += len(initial_nn_params[i].flatten()) 
    for i in range(8,16): sum2 += len(initial_nn_params[i].flatten()) 
    print "len(theta1), len(theta2):", sum1, sum2
  
    for i in range(len(initial_nn_params)): theta.extend(initial_nn_params[i].flatten())
   
    n_epochs = 20
    sample_size = 1024 #512
    MaxIter = 1000
    lambdaa = 1e-6
    for i in range(n_epochs):
        inds = np.random.choice(range(images.shape[1]),sample_size,replace=False)
        X = images[:,inds].T
        print "==" * 30 , " epoch: %d " %i , "==" * 30
        out = op.fmin_l_bfgs_b(sparseAutoencoderCost,theta,fprime=None,args=(shapes, X, lambdaa ,sparsityParam, beta),maxfun=MaxIter, disp=50)
        theta = out[0]

    inds = np.random.choice(range(images.shape[1]),10000,replace=False)
    X = images[:,inds].T
    MaxIter = 4000
    out = op.fmin_l_bfgs_b(sparseAutoencoderCost,theta,fprime=None,args=(shapes, X, lambdaa ,sparsityParam, beta),maxfun=MaxIter, disp=20)
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8, b8 = reshape_theta(out[0], shapes)

    po_fix = 'epoch%s_size%s' %(n_epochs, sample_size)
    np.savetxt('trained_W1_ft_%s.txt' %po_fix,W1)
    np.savetxt('trained_b1_ft_%s.txt' %po_fix,b1)
    np.savetxt('trained_W2_ft_%s.txt' %po_fix,W2)
    np.savetxt('trained_b2_ft_%s.txt' %po_fix,b2)
    np.savetxt('trained_W3_ft_%s.txt' %po_fix,W3)
    np.savetxt('trained_b3_ft_%s.txt' %po_fix,b3)
    np.savetxt('trained_W4_ft_%s.txt' %po_fix,W4)
    np.savetxt('trained_b4_ft_%s.txt' %po_fix,b4)

train_stackedAE()
#stackedAEcost(theta,inputSize, hiddenSizeL1, hiddenSizeL2, numClasses, X, y, lambdaa, sparsityParam, beta)
#stackedAEgrad(theta,inputSize, hiddenSizeL1, hiddenSizeL2, numClasses, X, y, lambdaa, sparsityParam, beta)
#=============================================================

