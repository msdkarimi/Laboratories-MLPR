import array
import math

import numpy
import scipy

def toCol(theArray:numpy.array):
    try:
        return theArray.reshape(theArray.size, 1)
    except Exception as e:
        print(e)

def toRow(theArray:numpy.array):
    try:
        return theArray.reshape(1,theArray.size)
    except Exception as e:
        print(e)

def mean(samples: numpy.array):
    try:
        return samples.mean(1).reshape(samples.shape[0], 1)
    except Exception as e:
        print(e)


def centerSampels (samples: numpy.array):
    try:
        mean = samples.mean(1).reshape(samples.shape[0], 1)
        return samples - mean

    except Exception as e:
        print(e)

def covarinceMartix( sampels:numpy.array )-> numpy.array:
    try:
        return   numpy.dot(centerSampels(sampels), centerSampels(sampels).T)/ sampels.shape[1]

    except Exception as e:
        print(e)


def kfold(samples: numpy.array, labels, foldNumber: int, folds: int = 3, seed = 0):
    # maskOfEachClass = dict()
    # for classLabel in range(len(set(labels))):
    #     maskOfGivenClass = numpy.where (labels == classLabel)
    #     maskOfEachClass[classLabel] = samples[:, maskOfGivenClass[0]]
    numpy.random.seed(seed)
    idx = numpy.random.permutation(samples.shape[1])

    testDataIndex = [foldNumber]
    trainDataIndx = list(set(idx) - set(testDataIndex))

    testData = samples[:, testDataIndex]
    testLabel = labels[testDataIndex]

    trainData = samples[:, trainDataIndx]
    trainLabel = labels[trainDataIndx]

    return (trainData, trainLabel), (testData, testLabel)

def PCA(samples: numpy.array, covarianceMatrix: numpy.array, m: int = 4)->tuple:

    try:
        U, s, _ = numpy.linalg.svd(covarianceMatrix)
        P = U[:, 0:m]
        mapped = numpy.dot(P.T, samples)
        return mapped

    except Exception as e:
        print(e)

def LDA(samples: numpy.array, labels: numpy.array, m: int = 2):

    try:
        sW, sB = getSW_SB(samples, labels)
        _, U = scipy.linalg.eigh(sB, sW)
        W = U[:, ::-1][:, 0:m]

        mapped = numpy.dot(W.T, samples)
        return mapped, W

    except Exception as e:
        print(e)


def getSW_SB(dataSet, labels):
    try:
        sW = 0
        for i in range(len(set(labels))):
            eachClass = dataSet[:, labels == i]
            nEachClass = eachClass.shape[1]
            swCTemp = covarinceMartix(eachClass)
            swCTemp = swCTemp * nEachClass
            sW = sW + swCTemp

        sW = sW / dataSet.shape[1]

        sB = 0
        for i in range(len(set(labels))):
            mu = mean(dataSet)
            eachClass = dataSet[:, labels == i]
            nEachClass = eachClass.shape[1]
            muC = mean(eachClass)
            sBtemp = nEachClass * numpy.dot((muC - mu), (muC - mu).T)

            sB = sB + sBtemp

        sB = sB / dataSet.shape[1]

        return sW, sB
    except Exception as e:
        print(e)



def logpdf_GAU_ND(X, mu, C):
    try:
        y = [ logpdfOneSample(X[:,i:i+1], mu, C) for i in range(X.shape[1])]
        return numpy.array(y).ravel()

    except Exception as e:
        print(e)

def logpdfOneSample(x, mu, C):
    try:
        xc = x - mu
        M = x.shape[0]
        constant = -0.5 * M * numpy.log(2*numpy.pi)
        logDetSigma = numpy.linalg.slogdet(C)[1]
        invSigma = numpy.linalg.inv(C)
        vector = numpy.dot(xc.T, numpy.dot(invSigma,xc))
        return constant - 0.5 * logDetSigma - 0.5 * vector

    except Exception as e:
        print(e)

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def model_MVG(trainData, trainLabel, testData, testLabel , prior, model = "G"):
    try:
        means = {}
        sigmas = {}
        tiedSigma = 0

        for i in range(len(set(trainLabel))):
            theClassOfLabel = trainData [:, trainLabel == i]
            mu = mean(theClassOfLabel)
            sigma = covarinceMartix(theClassOfLabel)
            if model == "N":
                I = numpy.eye(trainData.shape[0], trainData.shape[0])
                sigma = sigma *  I
            means[i] = mu
            sigmas[i] = sigma

        if model == "T":
            for i in range(len(set(trainLabel))):
                theSigma = sigmas[i]
                givenClassSize = trainData[:, trainLabel == i].shape[1]
                tiedSigma += (givenClassSize * theSigma)

            tiedSigma /= trainData.shape[1]



        logSJoint = numpy.zeros(( len(set(testLabel)) ,testData.shape[1]))
        for i in range(len(set(testLabel))):
            muML = means[i]
            if model != "T":
                sigmaML = sigmas[i]
            else:
                sigmaML = tiedSigma
            scores = logpdf_GAU_ND(testData, muML, sigmaML) + numpy.log(prior)
            logSJoint[i,:] =  scores

        SMarginal = scipy.special.logsumexp(logSJoint, 0)
        logPostorior = logSJoint - SMarginal

        # predictedLabels = logPostorior.argmax(0)
        return logPostorior, SMarginal, logSJoint

    except Exception as e:
        print(e)

def accuracyError(testLabels, predeictedLabels):
    try:
        totalNumber = testLabels.shape[0]
        trueAssignment = sum(testLabels == predeictedLabels)
        return (trueAssignment/totalNumber) * 100, ( 1-(trueAssignment/totalNumber)) * 100

    except Exception as e:
        print(e)

class MyKFold:
    def __init__(self,n_splits = 3, seed = 0):
        self.numberOfFolds = n_splits
        self.seed = seed

    def kfSplite(self, Xdataset, Ydataset):
        pairOfTrainTest = list()
        numpy.random.seed(self.seed)
        indices = numpy.random.permutation(Xdataset.shape[1])
        folds = numpy.array_split(indices, self.numberOfFolds)

        for fold in range(self.numberOfFolds):
            test_indices = folds[fold]
            train_indices = numpy.concatenate(folds[:fold] + folds[fold + 1:])

            Xtrain = Xdataset[:, train_indices]
            Ytrain = Ydataset[train_indices]

            Xtest = Xdataset[:, test_indices]
            Ytest = Ydataset[test_indices]


            pairOfTrainTest.append(((Xtrain, Ytrain),(Xtest, Ytest)))

        self.allFolds = pairOfTrainTest


def load_iris_binary(D, L):
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L


def logReg_obj_wrap_prior_weighted(trainData, trainLabel, priorTarget, lambdaa):
    features = trainData.shape[0]
    Z = trainLabel * 2.0 - 1.0

    class0_size = len(trainLabel[trainLabel == 0])
    class1_size = len(trainLabel[trainLabel == 1])

    class0_z = Z[trainLabel == 0]
    class1_z = Z[trainLabel == 1]

    def logReg(V):
        w = toCol(V[0:features])
        b = V[-1]

        class0_scores = numpy.dot(w.T, trainData[:, trainLabel == 0]) + b
        class1_scores = numpy.dot(w.T, trainData[:, trainLabel == 1]) + b

        class0_prior_Weight_loss = ((1-priorTarget)/class0_size) * numpy.logaddexp(0, -class0_z * class0_scores).sum()
        class1_prior_Weight_loss = ((priorTarget)/class1_size) * numpy.logaddexp(0, -class1_z * class1_scores).sum()
        regulizer = 0.5 * lambdaa * numpy.linalg.norm(w)**2

        return regulizer + class1_prior_Weight_loss + class0_prior_Weight_loss
    return logReg


def train_logReg_prior_weighted(trainData, trainLabel, testData, priorTarget = 0.5, lambdaa=0.001):
    logReg_obj = logReg_obj_wrap_prior_weighted(trainData, trainLabel, priorTarget, lambdaa)
    x0 = numpy.zeros(trainData.shape[0]+1)
    V, _, _ = scipy.optimize.fmin_l_bfgs_b(logReg_obj, x0=x0, approx_grad=True)
    w, b = toCol(V[0:trainData.shape[0]]), V[-1]
    scores = numpy.dot(w.T, testData) + b
    predictedLabels = (scores.ravel() > 0) * 1
    return scores, predictedLabels

def logReg_obj_wrap(trainData, trainLabel, lambdaa):
    features = trainData.shape[0]
    Z = trainLabel * 2.0 - 1.0
    def logReg(V):
        w = toCol(V[0:features])
        b = V[-1]
        scores = numpy.dot(w.T, trainData) + b
        loss_sample = numpy.logaddexp(0, -Z * scores)
        loss = loss_sample.mean() + 0.5 * lambdaa * numpy.linalg.norm(w)**2
        return loss
    return logReg

def train_logReg(trainData, trainLabel, lambdaa):
    logReg_obj = logReg_obj_wrap(trainData, trainLabel, lambdaa)
    x0 = numpy.zeros(trainData.shape[0]+1)
    V, _, _ = scipy.optimize.fmin_l_bfgs_b(logReg_obj, x0=x0, approx_grad=True)
    return toCol(V[0:trainData.shape[0]]), V[-1]


def confusion_matrix(ground_truth_labels, predicted_labels):
    cm = numpy.zeros((len(set(ground_truth_labels)), len(set(ground_truth_labels))))

    mask_GT = ground_truth_labels == 0
    predicted = predicted_labels[mask_GT]
    ground_truth = ground_truth_labels[mask_GT]
    tN = sum(predicted == ground_truth)
    fP = sum (ground_truth_labels == 0) - tN

    mask_GT = ground_truth_labels == 1
    predicted = predicted_labels[mask_GT]
    ground_truth = ground_truth_labels[mask_GT]
    tP = sum(predicted == ground_truth)
    fN = sum(ground_truth_labels == 1) - tP

    cm[0][0] = tN
    cm[0][1] = fP
    cm[1][0] = fN
    cm[1][1] = tP
    tpR = tP/(fN + tP)
    fpR = fP/(fP+tN)
    fnR = 1 - tpR
    tnR = 1 - fpR
    return cm, tpR, fpR, fnR, tnR

def empirical_bayes_risk(fnR, fpR, fnC, fpC, targetPrior):
    return (targetPrior * fnC * fnR) + ( (1-targetPrior)* fpC * fpR)

def normal_dcf(fnR, fpR, fnC, fpC, targetPrior = 0.5):
    return empirical_bayes_risk(fnR, fpR, fnC, fpC, targetPrior)/ min(targetPrior*fnC, (1-targetPrior)*fpC)