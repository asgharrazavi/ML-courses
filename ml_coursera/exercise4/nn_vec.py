import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import scipy.optimize as op
import copy
from matplotlib.colors import LogNorm

#-------------------------------#
#   Vectorized Neural Netwrok   #
#-------------------------------#


#-----------loading data----------------
data = io.loadmat('ex4/ex4data1.mat')
print data.keys()
y = data['y']					#shape : (5000,1)
x = data['X']					#shape : (5000, 400)
#X = np.c_[np.ones(x.shape[0]),x]
print "y.shape, x.shape:", y.shape, x.shape
#--------------------------------------

#--------------load neural network parameters----------
data = io.loadmat('ex4/ex4weights.mat')
Theta1 = data['Theta1']			 	#shape: (25,401)
Theta2 = data['Theta2']				#shape: (10,26)
print "Theta1.shape, Theta2.shape:", Theta1.shape, Theta2.shape
#------------------------------------------------------

#-----------plot some of the data------
def plot(x,y):
    m = len(y)
    rand_x = np.random.choice(m,100)
    xx = x[rand_x,:]
    for i in range(100):
        plt.subplot(10,10,i+1)
        img = np.reshape(xx[i],(20,20))
        plt.imshow(np.rot90(img))
        plt.xlim([0,20])
        plt.ylim([0,20])
        plt.xticks([])
        plt.yticks([])
    plt.show()
#plot(x,y)
#-------------------------------------

#----------------Compute neural network cost function------------------------
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def randInitializeWeights(L_out, L_in):
    epsilon_init = 0.12			# This number comes from: ( sqrt(6) / (sqrt{(# of layers before)+(# of layers after)}) )
    return np.random.random((L_out, 1 + L_in)) * 2 * epsilon_init - epsilon_init

def debug_randInitializeWeights(L_out, L_in):
    return np.reshape(np.sin(range(L_out * (1 + L_in))),(L_out, 1 + L_in))

def _h_theta(nn_params, X):                 #Theta1: (25,401), Theta2: (10,26), X: (5000,401)
    for i in range(len(nn_params)):
	if i == 0:  zz = np.dot(nn_params[i],X.T).T ; aa = sigmoid(zz)
	else: aa = np.c_[np.ones(aa.shape[0]),aa]; zz = np.dot(nn_params[i],aa.T).T ; aa = sigmoid(zz)
    a3 = aa                            #a3.shape: (5000,10)
    return a3

def _h_theta2(nn_params, X):                 #Theta1: (25,401), Theta2: (10,26), X: (5000,401)
    a1 = X.T
    z2 = np.dot(nn_params[0],a1).T ; a2 = sigmoid(z2) ; a2 = np.c_[np.ones(a2.shape[0]),a2]
    z3 = np.dot(nn_params[1],a2.T).T ; a3 = sigmoid(z3)
    return z2, z3, a1, a2, a3

def _fix_y(y,num_labels):
    yy = np.zeros((len(y),num_labels))
    for i in range(len(y)):
 	yy[i][y[i]-1] = 1
    if 0 :print y[2080], yy[2080]
    return yy

def _add_regCost(nn_params,y,lambdaa):
    m = float(len(y))
    reg_j = 0
    for i in range(len(nn_params)):
	reg_j += np.sum(nn_params[i][:,1:]**2)	# [:,1:]:we don't regularize the terms that correspond to the bias 
    reg_j = (lambdaa/(2.0 * m)) * reg_j
    return reg_j

def nnBackPro(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa ,reshape_theta=True):
    if reshape_theta: nn_params = _reshape_theta(nn_params,input_layer_size, hidden_layer_size, num_labels)
    X = np.c_[np.ones(X.shape[0]),X]
    m = len(y)
    yy = _fix_y(y,num_labels)
    z2, z3, a1, a2, a3 = _h_theta2(nn_params, X)
    delta3_v = a3 - yy
    g_prime_z2 = sigmoidGradient(z2)
    g_prime_z3 = sigmoidGradient(z3)
    delta2_v = np.dot(nn_params[1][:,1:].T,delta3_v.T).T * g_prime_z2
    D1 = np.dot(delta2_v.T, a1.T)  
    D2 = np.dot(delta3_v.T, a2)  
    D1 = (1.0/m) * D1
    D2 = (1.0/m) * D2
    D1[:,1:] = D1[:,1:] + (float(lambdaa)/m) * nn_params[0][:,1:]		# reqularizing all output nodes and 1:all input nodes
    D2[:,1:] = D2[:,1:] + (float(lambdaa)/m) * nn_params[1][:,1:]
    D = []
    D.extend(D1.flatten())
    D.extend(D2.flatten())
    return np.array(D)

def _reshape_theta(theta, input_layer_size, hidden_layer_size, num_labels):
    #Theta1: (25,401), Theta2: (10,26), X: (5000,401)
    theta1 = np.reshape(theta[0:hidden_layer_size*(input_layer_size + 1)],(hidden_layer_size,input_layer_size + 1))
    theta2 = np.reshape(theta[hidden_layer_size*(input_layer_size + 1):] ,(num_labels, hidden_layer_size + 1))
    return [theta1, theta2]

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa ,reshape_theta=True):
    X = np.c_[np.ones(X.shape[0]),X]
    if reshape_theta: nn_params = _reshape_theta(nn_params,input_layer_size, hidden_layer_size, num_labels)
    m = float(len(y))
    y = y[:,0]
    h_theta = _h_theta(nn_params, X)
    yy = _fix_y(y,num_labels)
    J = (-1.0/m) * np.sum(yy*np.log(h_theta) + (1-yy)*np.log(1-h_theta)) 
    reg_J = _add_regCost(nn_params,y,lambdaa)
    print J, reg_J
    return J + reg_J

def computeNumericalGradient(costFunc, nn_params):
    unrolled_theta = []
    for i in range(len(nn_params)):
	unrolled_theta.extend(nn_params[i].flatten())
    numgrad = np.zeros(len(unrolled_theta))			
    perturb = np.zeros(len(unrolled_theta))
    print "numgrad.shape, perturb.shape:", numgrad.shape, perturb.shape
    e = 1e-4;
    for p in range(len(unrolled_theta)):
	if p%500 == 0: print p, ";\tpercent completed:", p*100/len(unrolled_theta)
    	perturb[p] = e
    	loss1 = costFunc(unrolled_theta - perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa, reshape_theta=True)
    	loss2 = costFunc(unrolled_theta + perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa, reshape_theta=True)
    	numgrad[p] = (loss2 - loss1) / (2.0*e)
    	perturb[p] = 0
    return numgrad

#--------testing sigmoidGradient function----
#print sigmoidGradient(1)
#print sigmoidGradient(100)
#print sigmoidGradient(-100)
#print sigmoidGradient(0)      #should be 0.25
#-------------------------------------------

input_layer_size  = 400				# 20x20 Input Images of Digits
hidden_layer_size = 25   			# 25 hidden units
num_labels = 10  
lambdaa = 3.0
#-----------------Debugging Cost function and Gradient---------------------------------------------
debug = False
if debug:
    y = y[::100,0]
    initial_Theta1 = debug_randInitializeWeights(hidden_layer_size, input_layer_size)
    initial_Theta2 = debug_randInitializeWeights(num_labels, hidden_layer_size)
    print " initial_Theta1.shape, initial_Theta2.shape:", initial_Theta1.shape, initial_Theta2.shape
    initial_nn_params = [initial_Theta1, initial_Theta2]
    X = debug_randInitializeWeights(len(y),input_layer_size - 1)
    y  = 1 + np.mod(range(len(y)), num_labels)
    y = np.reshape(y,(y.shape[0],1))
    print "debug X and y shapes:", X.shape, y.shape
    grad = nnBackPro(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa ,reshape_theta=False)
    np.savetxt('grad_all_debug.txt',grad)
#----check gradient against numerical gradient--
#if 0:
    numgrad = computeNumericalGradient(nnCostFunction,initial_nn_params)
    np.savetxt('num_grad_unrolled_debug.txt',numgrad)
def plot_num_gradient(grad,numgrad):
    unrolled_grad = grad
    plt.plot(unrolled_grad - numgrad,'r.')
    plt.xlabel('Tetha');plt.ylabel('grad - numgrad')
    plt.show()
if debug: plot_num_gradient(grad,numgrad)
#-----------------------------------------------
#--------------------------------------------------------------------------------------------------

#------------Predict------------
def predict(Theta1, Theta2, X):
    X = np.c_[np.ones(X.shape[0]),X]
    nn_params = [Theta1, Theta2]
    a3 = _h_theta(nn_params, X)
    a3 = np.rint(a3)
    p = np.zeros(X.shape[0],dtype=int)
    for i in range(len(p)):
        w = np.where(a3[i]==1)[0]
        if len(w) > 0: p[i] = w[0]+1
        else: pass
    return p
#------------------------------

#--------------Display Theta1------------
def plot_Theta1(Theta1):
    print "Theta1.shape:", Theta1.shape
    n_units = Theta1.shape[0]
    n_plts = 1
    while n_plts * n_plts < n_units:
	n_plts += 1
    for i in range(n_units):
        plt.subplot(n_plts,n_plts,i+1)
        img = np.reshape(Theta1[i,1:],(20,20))
        plt.imshow(np.rot90(img),cmap='gray')
#        plt.imshow(np.rot90(img),cmap='gray',norm=LogNorm())
        plt.xlim([0,20])
        plt.ylim([0,20])
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

#===============Main training NN===================================================================================================================================
def train_nn():
    initial_Theta1 = randInitializeWeights(hidden_layer_size, input_layer_size)
    initial_Theta2 = randInitializeWeights(num_labels, hidden_layer_size)
    nn_params = [initial_Theta1, initial_Theta2]
    unrolled_theta = []
    for i in range(len(nn_params)):
    	unrolled_theta.extend(nn_params[i].flatten())
    lambdaa = 0.1
    MaxIter = 400
    print "X.shape, y.shape:", x.shape, y.shape
#    out = op.fmin_cg(nnCostFunction,unrolled_theta,nnBackPro,args=(input_layer_size, hidden_layer_size, num_labels, x, y, lambdaa),full_output=1,maxiter=MaxIter) 
    out = op.fmin_l_bfgs_b(nnCostFunction,unrolled_theta,nnBackPro,args=(input_layer_size, hidden_layer_size, num_labels, x, y, lambdaa),maxfun=MaxIter, disp=1)
    Theta1, Theta2 = _reshape_theta(out[0], input_layer_size, hidden_layer_size, num_labels)
    np.savetxt('trained_theta1_l1.0.txt',Theta1)
    np.savetxt('trained_theta2_l1.0.txt',Theta2)
    p = predict(Theta1, Theta2, x)
    print "prediction accuracy: %f percent:" %(np.mean(p == y.ravel())*100)
    #MaxIter = 50 , lambda = 1.0 ::: prediction accuracy:  94.340000 percent
    #MaxIter = 400, lambda = 1.0 ::: prediction accuracy:  98.480000 percent
    #MaxIter = 400, lambda = 0.1 ::: prediction accuracy: 100.000000 percent
#===================================================================================================================================================================

train_nn()
#Theta1 = np.loadtxt('trained_theta1_l1.0.txt')
#plot_Theta1(Theta1)
#Theta2 = np.loadtxt('trained_theta2_l1.0.txt')
#plot_Theta2(Theta2)
