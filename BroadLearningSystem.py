import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA 
import time
from FLBLS import LDMBLS,REMBLS


def show_accuracy(predictLabel, Label): 
    count = 0
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label_1[j] == predlabel[j]:
            count += 1
    return (round(count/len(Label),5))


def tansig(x):
    return (2/(1+np.exp(-2*x)))-1


def sigmoid(data):
    return 1.0/(1+np.exp(-data))
    

def linear(data):
    return data
    

def tanh(data):
    return (np.exp(data)-np.exp(-data))/(np.exp(data)+np.exp(-data))
    

def relu(data):
    return np.maximum(data, 0)


def pinv(A, reg):
    return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)


def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum( -a - b, 0)
    return z


def sparse_bls(A, b):
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)   
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def BLS(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, method='LDMBLS'):
    L = 0
    train_x = preprocessing.scale(train_x, axis=1)
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0],1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2*N1])
    Beta1OfEachWindow = []

    distOfMaxAndMin = []
    minOfEachWindow = []
    ymin = 0
    ymax = 1
    train_acc_all = np.zeros([1,L+1])
    test_acc = np.zeros([1,L+1])
    train_time = np.zeros([1,L+1])
    test_time = np.zeros([1,L+1])
    time_start=time.time()#计时开始
    for i in range(N2):
        random.seed(i)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1,N1)-1
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow) 
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        betaOfEachWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias,betaOfEachWindow)
#        print('Feature nodes in window: max:',np.max(outputOfEachWindow),'min:',np.min(outputOfEachWindow))
        distOfMaxAndMin.append(np.max(outputOfEachWindow,axis =0) - np.min(outputOfEachWindow,axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow,axis = 0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:, N1*i:N1*(i+1)] = outputOfEachWindow
        del outputOfEachWindow 
        del FeatureOfEachWindow 
        del weightOfEachWindow 

    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])

    if N1*N2>=N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3))-1
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T
    
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayer)
#    print('Enhance nodes: max:',np.max(tempOfOutputOfEnhanceLayer),'min:',np.min(tempOfOutputOfEnhanceLayer))

    parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)

    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    
    #生成最终输入
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer,OutputOfEnhanceLayer])

    #更新权重
    #pinvOfInput = pinv(InputOfOutputLayer,c)
    #OutputWeight = np.dot(pinvOfInput,train_y)

    if method == 'LDMBLS':
        OutputWeight = LDMBLS(InputOfOutputLayer, train_y)
    elif method == 'REMBLS':
        OutputWeight = REMBLS(InputOfOutputLayer, train_y, c)
    else:
        raise ValueError("Invalid method. Choose 'LDMBLS' or 'REMBLS'.")

    time_end=time.time()
    trainTime = time_end - time_start
    
    OutputOfTrain = np.dot(InputOfOutputLayer,OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain,train_y)
    print('Training accurate is' ,trainAcc*100,'%')
    print('Training time is ',trainTime,'s')
    train_acc_all[0][0] = trainAcc
    train_time[0][0] = trainTime
    #测试过程
    test_x = preprocessing.scale(test_x,axis = 1)
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0],1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0],N2*N1])
    time_start=time.time()

    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest,Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:,N1*i:N1*(i+1)] =(ymax-ymin)*(outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i]-ymin

    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0],1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest,weightOfEnhanceLayer)

    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)    

    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,OutputOfEnhanceLayerTest])

    OutputOfTest = np.dot(InputOfOutputLayerTest,OutputWeight)
    time_end=time.time()
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest,test_y)
    print('Testing accurate is' ,testAcc * 100,'%')
    print('Testing time is ',testTime,'s')
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime

    return test_acc,test_time,train_acc_all,train_time

