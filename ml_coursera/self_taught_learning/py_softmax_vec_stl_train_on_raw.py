import numpy as np
import mdtraj.io as io
import matplotlib.pyplot as plt
import scipy.optimize as op
import copy
import h5py
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

#-----------loading data----------------
images = io.loadh('mnist_images.h5')['arr_0'].T
labels = io.loadh('mnist_labels.h5')['arr_0']
print "images.shape, labels,shape:", images.shape, labels.shape                 # (784, 60000) (60000,)
#---------------------------------------

#----------label vs. unlabel---------------
labeledSet = (labels  <= 4)
unlabeledSet = (labels > 4)

numTrain = len(labels[labeledSet])/2
print "numTrain:", numTrain
print "len(labeled), len(unlabeled):", len(labels[labeledSet]), len(labels[unlabeledSet])
print "number of True in 1st half:", len(np.where(labeledSet[0:numTrain] == True)[0])
print "number of True in 2nd half:", len(np.where(labeledSet[numTrain:] == True)[0])

unlabeledData = images[:, unlabeledSet]

train_test_img = images[:,labeledSet]
train_test_lab = labels[labeledSet]
trainData = train_test_img[:,0:numTrain]
trainLabels = train_test_lab[0:numTrain]
testData = train_test_img[:,numTrain:]
testLabels = train_test_lab[numTrain:]


#trainData   = images[:, labeledSet[0:numTrain]]
#trainLabels = labels[labeledSet[0:numTrain]]

#testData   = images[:, labeledSet[numTrain:]]
#testLabels = labels[labeledSet[numTrain:]]

# Output Some Statistics
print '# examples in unlabeled set: %d\n' %unlabeledData.shape[1]                       # 29404 
print '# examples in supervised training set: %d\n\n' %trainData.shape[1]               # 7824                          
print '# examples in supervised testing set: %d\n\n' %testData.shape[1]                 # 22772
print "train lables, test labels:", np.unique(trainLabels), np.unique(testLabels)
#------------------------------------------

#------------------------------------------
trainFeatures = io.loadh('trainFeatures.h5')['arr_0']
testFeatures = io.loadh('testFeatures.h5')['arr_0']
print "trainFeatures.shape:", trainFeatures.shape
print "testFeatures.shape:", testFeatures.shape
#----------------------------------------------------------

#-----------softmax---------------------
def _fix_y(y,num_labels):							# y : (60000,10)
    yy = np.zeros((len(y),num_labels))
    for i in range(len(y)):
	yy[i,int(y[i])] = 1
    return yy.T

def softmaxCost(theta, numClasses, inputSize, lambdaa, inputData, labels):
    theta = np.reshape(theta, (numClasses, inputSize))
    m = len(labels)
    maxx = np.max(np.dot(theta,inputData))
    numer = np.exp(np.dot(theta,inputData) - maxx)
    denum = np.sum(np.exp(np.dot(theta,inputData) - maxx),axis=0).astype(float)
    yy = _fix_y(labels,numClasses)
    cost = (-1.0/m) * np.sum(yy * np.log(numer/denum))
    cost += (lambdaa/2.0) * np.sum(theta**2)
    return cost

def grad(theta, numClasses, inputSize, lambdaa, inputData, labels):
    theta = np.reshape(theta, (numClasses, inputSize))
    m = len(labels)
    maxx = np.max(np.dot(theta,inputData))
    numer = np.exp(np.dot(theta,inputData) - maxx)
    denum = np.sum(np.exp(np.dot(theta,inputData) - maxx),axis=0).astype(float)
    yy = _fix_y(labels,numClasses)
    yy_p = yy - (numer/denum)
    thetagrad = np.dot(inputData,yy_p.T).T    
    thetagrad = (-1.0/m) * thetagrad
    thetagrad += lambdaa * theta
    return thetagrad.flatten()

def computeNumericalGradient(theta, numClasses, inputSize, lambdaa, inputData, labels):
    print "X.shape, y.shape:", inputData.shape, labels.shape
    unrolled_theta = theta[:,0]
    numgrad = np.zeros(len(unrolled_theta))
    perturb = np.zeros(len(unrolled_theta))
    print "numgrad.shape, perturb.shape:", numgrad.shape, perturb.shape
    e = 1e-4;
    for p in range(len(unrolled_theta)):
        if p%500 == 0: print p, "percent completed:", p*100/len(unrolled_theta)
        perturb[p] = e
        loss1 = softmaxCost(unrolled_theta - perturb, numClasses, inputSize, lambdaa, inputData, labels) 
        loss2 = softmaxCost(unrolled_theta + perturb, numClasses, inputSize, lambdaa, inputData, labels) 
        numgrad[p] = (loss2 - loss1) / (2.0*e)
        perturb[p] = 0
    return numgrad
#-------------------------------------------------------------------------------------------------------------------

#------parameters---------------------------
inputSize = 28 * 28; # Size of input vector (MNIST images are 28x28)
numClasses = 10;     # Number of classes (MNIST images fall into 10 classes)

lambdaa = 1e-4;      # Weight decay parameter
inputData = images
#-----------------------------------------

#-------------------Debugging------------------------------------------
Debug = False
if Debug:
#    inputSize = 8 
    images = images[:,::10000]
    labels = labels[::10000]
    #Randomly initialise theta
    theta = 0.005 * np.random.random((numClasses * inputSize, 1))
    cost = softmaxCost(theta, numClasses, inputSize, lambdaa, images, labels)
    grad = grad(theta, numClasses, inputSize, lambdaa, images, labels)
    numgrad = computeNumericalGradient(theta, numClasses, inputSize, lambdaa, images, labels)
    np.savetxt('num_grad_unrolled_debug.txt',numgrad)
    def plot_num_gradient(grad,numgrad):
        unrolled_grad = grad.flatten()
        print "grad.shape, numgrad.shape, unrolled_grad.shape:", np.array(grad).shape, np.array(numgrad).shape, np.array(unrolled_grad).shape
        plt.plot(unrolled_grad - numgrad,'r.')
        plt.xlabel('Tetha');plt.ylabel('grad - numgrad')
        plt.savefig('check.png')
        plt.show()
    plot_num_gradient(grad,numgrad)
#---------------------------------------------------------------------

#-------------------------Training--------------------------------------------------------------
#------parameters---------------------------
print "Starting training..."
inputSize = 28 * 28
numClasses = 5 #10     
lambdaa = 1e-4
inputData = trainData
labels = trainLabels
#-----------------------------------------
def train_nn():
    MaxIter = 100
    unrolled_theta = 0.005 * np.random.random((numClasses * inputSize, 1))
    out = op.fmin_l_bfgs_b(softmaxCost,unrolled_theta,grad,args=(numClasses, inputSize, lambdaa, inputData, labels),maxfun=MaxIter, disp=1)
    theta = np.reshape(out[0], (numClasses, inputSize))
    np.savetxt('trained_theta_raw.txt',theta)
    return theta
#train_nn()
#-----------------------------------------------------------------------------------------------

#--------Predict-------------
def softmaxPredict(theta, data):
    print theta.shape, data.shape
    pred = np.dot(theta,data)
#    pred = np.rint(pred)
    pred = [np.argmax(pred[:,i]) for i in range(data.shape[1])]
    return pred
#theta = train_nn()
try: theta = np.loadtxt('trained_theta_raw.txt')
except: theta = train_nn()
p = softmaxPredict(theta, trainData)
print "train prediction accuracy for raw Data: %f percent" %(np.mean(p == trainLabels.ravel())*100)
p = softmaxPredict(theta, testData)
print "test prediction accuracy for raw Data: %f percent" %(np.mean(p == testLabels.ravel())*100)
