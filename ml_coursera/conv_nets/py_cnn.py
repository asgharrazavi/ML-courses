import numpy as np
import h5py
import scipy.io as io
from tqdm import tqdm
import mdtraj.io as ioo
import scipy.optimize as op

#----load theta------
W1 = np.loadtxt('../linear_decoder/trained_W1.txt')
b1 = np.loadtxt('../linear_decoder/trained_b1.txt')
#ZCAWhite = np.loadtxt('../linear_decoder/ZCAWhite.txt')
print "W1.shape, b1.shape:", W1.shape, b1.shape
#--------------------

#------load images------
#data = io.loadmat('stlSubset/stlTrainSubset.mat')
#print data.keys()
#trainImages = data['trainImages']
#numTrainImages = data['numTrainImages']
#trainLabels = data['trainLabels']

data = io.loadmat('stlSubset/stlTestSubset.mat')
trainImages = data['testImages']
numTrainImages = data['numTestImages']
trainLabels = data['testLabels']

ZCAWhite = io.loadmat('../linear_decoder/data_from_matlab/ZCAWhite.mat')
ZCAWhite = ZCAWhite['ZCAWhite']
print "ZCAWhite.shape:", ZCAWhite.shape
meanPatch = io.loadmat('../linear_decoder/data_from_matlab/mean_patches.mat')['meanPatch'][:,0]
print "meanPatch.shape:", meanPatch.shape
print "trainImages.shape, numTrainImages, trainLabels.shape:", trainImages.shape, numTrainImages, trainLabels.shape
#-----------------------

#-------convolution-------
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def feedForwardAutoencoder(W1, b1, X):
    a1 = X#.T
    try : B1 = np.array([b1[:,0] for i in range(X.shape[1])])
    except : B1 = np.array([b1 for i in range(X.shape[0])])
#    print "W1.shape, a1.shape, B1.shape:", W1.shape,a1.shape, B1.shape
    try : z2 = np.dot(W1,a1).T
    except : z2 = np.dot(W1[:,1:],a1).T
    z2 = z2 + B1
    a2 = sigmoid(z2)
#    a2 = np.c_[np.ones(a2.shape[0]),a2]
    return a2

def cnnConvolve(patchDim, numFeatures, images, W, b, ZCAWhite, meanPatch):
    numImages = images.shape[3]
    imageDim = images.shape[0]
    imageChannels = images.shape[2]
    				#   400          8           64        8     
    convolvedFeatures = np.zeros((numFeatures, numImages, imageDim - patchDim, imageDim - patchDim))
    for i in tqdm(range(numImages)):
        for j in range(imageDim - patchDim):
	    for k in range(imageDim - patchDim):
    		img = images[j : j + patchDim, k : k + patchDim, :, i].flatten()
#   		print i, j, k, img.shape ; quit()
    		img = img - meanPatch
		img = np.dot(img,ZCAWhite)
		convolvedFeatures[:,i,j,k] = sigmoid(np.dot(W[:,1:],img) + b)
    print "convolvedFeatures.shape:", convolvedFeatures.shape
#    print "convolvedFeatures:", convolvedFeatures
    return convolvedFeatures

def check_Convolve(convolvedFeatures,W1,b1,images,ZCAWhite,meanPatch):
    numImages = images.shape[3]
    imageDim = images.shape[0]
    imageChannels = images.shape[2]
    hiddenSize = W1.shape[0]
    patchDim = 8
    visibleSize = patchDim * patchDim * imageChannels

    for i in range(1000):
        featureNum = np.random.choice(range(hiddenSize),1)[0]
        imageNum = np.random.choice(range(8),1)[0]
        imageRow = np.random.choice(range(imageDim - patchDim),1)[0]
        imageCol = np.random.choice(range(imageDim - patchDim),1)[0] 
#        print "featureNum, imageNum, imageRow, imageCol:", featureNum, imageNum, imageRow, imageCol 
#	print "convImages.shape:", convImages.shape

        patch = convImages[imageRow:imageRow + patchDim , imageCol:imageCol + patchDim , :, imageNum].flatten()
        patch = patch - meanPatch
        patch = np.dot(ZCAWhite , patch)
        
        features = feedForwardAutoencoder(W1,b1,patch).T			# (400,192)
#        print "features.shape, convolvedFeatures0.shape:", features.shape, convolvedFeatures.shape	# (400, 8, 56, 56) 
#  	print convolvedFeatures[featureNum, imageNum, imageRow, imageCol]
	print "%1.20f, %1.20f" %(features[featureNum,0], convolvedFeatures[featureNum, imageNum, imageRow, imageCol])
        if abs(features[featureNum,0] - convolvedFeatures[featureNum, imageNum, imageRow, imageCol]) > 1e-9:
            print('Convolved feature does not match activation from autoencoder\n')
            print('Feature Number    : %d\n', featureNum)
            print('Image Number      : %d\n', imageNum)
            print('Image Row         : %d\n', imageRow)
            print('Image Column      : %d\n', imageCol)
            print('Convolved feature : %0.5f\n', convolvedFeatures[featureNum, imageNum, imageRow, imageCol])
            print('Sparse AE feature : %0.5f\n', features[featureNum])
	    print "Exiting..." ; quit()
    print 'Congratulations! Your convolution code passed the test.'


#patchDim = 8
#numFeatures = W1.shape[0]
#convImages = trainImages[:, :, :, 0:8]
#convolvedFeatures = cnnConvolve(patchDim, numFeatures, convImages, W1, b1, ZCAWhite, meanPatch)
#check_Convolve(convolvedFeatures, W1, b1, convImages, ZCAWhite, meanPatch)
#-------------------------------------------------------------------------------------------------------------------------------

#----------------------------- pooling -------------------------------------
def cnnPool(poolDim, convolvedFeatures):
    numFeatures = convolvedFeatures.shape[0]
    numImages = convolvedFeatures.shape[1]
    convolvedDim = convolvedFeatures.shape[2]
				# 400		8		56/19					56/19
    pooledFeatures = np.zeros((numFeatures, numImages,  convolvedDim / poolDim, 		convolvedDim / poolDim))
#    pooledFeatures = np.zeros((numFeatures, numImages, np.floor(convolvedDim / poolDim), np.floor(convolvedDim / poolDim)))
    for i in tqdm(range(numImages)):
	for j in range(convolvedDim / poolDim):
	    for k in range(convolvedDim / poolDim):
		av = np.mean(np.mean(convolvedFeatures[:,i,j*poolDim:j*poolDim+poolDim,k*poolDim:k*poolDim+poolDim],axis=1),axis=1)
		pooledFeatures[:,i,j,k] = av 
    print "pooledFeatures.shape:", pooledFeatures.shape
#    print pooledFeatures
    return pooledFeatures

def check_pool():
    testMatrix = np.reshape(range(64), (8, 8))
    expectedMatrix = [np.mean(np.mean(testMatrix[0:4, 0:4])), np.mean(np.mean(testMatrix[0:4, 4:8]))\
                     ,np.mean(np.mean(testMatrix[4:8, 0:4])), np.mean(np.mean(testMatrix[4:8, 4:8])) ]
    print "expectedMatrix", expectedMatrix
    testMatrix = np.reshape(testMatrix, (1, 1, 8, 8))
    pooledFeatures = cnnPool(4,testMatrix).flatten()
    print "pooledFeatures", pooledFeatures

#check_pool()
#poolDim = 19
#patchDim = 8
#numFeatures = W1.shape[0]
#convImages = trainImages[:, :, :, 0:8]
#convolvedFeatures = cnnConvolve(patchDim, numFeatures, convImages, W1, b1, ZCAWhite, meanPatch)
#pooledFeatures = cnnPool(poolDim, convolvedFeatures)
#----------------------------------------------------------------------------

#--------- convolve and pool on all train and test data------------
def conv_pool():
    patchDim = 8
    poolDim = 19
    convImages = trainImages
    numFeatures = W1.shape[0]
    convolvedFeatures = cnnConvolve(patchDim, numFeatures, convImages, W1, b1, ZCAWhite, meanPatch)
    pooledFeatures = cnnPool(poolDim, convolvedFeatures)
    ioo.saveh('pooledFeatures_test.h5',pooledFeatures)
#conv_pool()
#------------------------------------------------------------------

#---------------- Train softmax classifier on train data ---------------------------------------------

#================== softmax ====================================================
def fix_y(y,num_labels):                                                        # y : (60000,10)
    yy = np.zeros((len(y),num_labels))
    if np.min(y) != 0: adjust = 1
    else: adjust = 0
    for i in range(len(y)):
        yy[i,int(y[i]-adjust)] = 1
    return yy.T

def softmaxCost(theta, numClasses, inputSize, lambdaa, inputData, labels):
    theta = np.reshape(theta, (numClasses, inputSize))
    m = len(labels)
    maxx = np.max(np.dot(theta,inputData))
    numer = np.exp(np.dot(theta,inputData) - maxx)
    denum = np.sum(np.exp(np.dot(theta,inputData) - maxx),axis=0).astype(float)
    yy = fix_y(labels,numClasses)
    cost = (-1.0/m) * np.sum(yy * np.log(numer/denum))
    cost += (lambdaa/2.0) * np.sum(theta**2)
    yy_p = yy - (numer/denum)
    thetagrad = np.dot(inputData,yy_p.T).T
    thetagrad = (-1.0/m) * thetagrad
    thetagrad += lambdaa * theta
    return cost, thetagrad.flatten()
#===============================================================================

def run_sotmax():
    pooledFeatures_train = ioo.loadh('pooledFeatures_train.h5')['arr_0']
    n_samples = pooledFeatures_train.shape[1]
    pooledFeatures_train = np.transpose(pooledFeatures_train, axes = [0,2,3,1] )
    print "Pooled data.shape:", pooledFeatures_train.shape
    inputSize = len(pooledFeatures_train.flatten()) / n_samples
    softmaxX = np.reshape(pooledFeatures_train,(inputSize, n_samples))
    softmaxY = trainLabels
    print "number of labels in softmaxY:", np.unique(softmaxY)
    MaxIter = 1000 #200
    softmaxLambda = 1e-4
    numClasses = 4
    unrolled_theta = 0.005 * np.random.random((numClasses * inputSize, 1))
    print "softmaxX.shape, softmaxY.shape, inputSize, numClasses:", softmaxX.shape, softmaxY.shape, inputSize, numClasses
    out = op.fmin_l_bfgs_b(softmaxCost,unrolled_theta,fprime=None,args=(numClasses, inputSize, softmaxLambda, softmaxX, softmaxY),maxfun=MaxIter, disp=1)
    theta = np.reshape(out[0], (numClasses, inputSize))
    np.savetxt('softmax_theta.txt',theta)
#run_sotmax()
#----------------------------------------------------------------------------------------------------

#----------- Test ----------
def softmaxPredict(theta, data):
    print "theta.shape, softmaxX.shape:", theta.shape, data.shape
    pred = np.dot(theta,data)
    pred = [np.argmax(pred[:,i]) for i in range(data.shape[1])]
    return pred

def run_predict():
    data = io.loadmat('stlSubset/stlTestSubset.mat')
    testLabels = data['testLabels']
    pooledFeatures_test = ioo.loadh('pooledFeatures_test.h5')['arr_0']
    n_samples = pooledFeatures_test.shape[1]
    pooledFeatures_test = np.transpose(pooledFeatures_test, axes = [0,2,3,1] )
    print "Pooled data.shape:", pooledFeatures_test.shape
    inputSize = len(pooledFeatures_test.flatten()) / n_samples
    softmaxX = np.reshape(pooledFeatures_test,(inputSize, n_samples))
    softmaxY = testLabels - 1
    theta = np.loadtxt('softmax_theta.txt')
    p = softmaxPredict(theta, softmaxX)
    print "predicted:", p[0:20]
    print "actual:", np.array(softmaxY[0:20].flatten())
    print "prediction accuracy: %f percent" %(np.mean(p == softmaxY.ravel())*100)
run_predict()
#---------------------------


