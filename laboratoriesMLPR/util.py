import array

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


def kfold(samples: numpy.array, labels, foldNumber: int, folds: int = 4, seed = 0):
    numpy.random.seed(seed)
    sampleSize = samples.shape[1]
    idx = numpy.random.permutation(sampleSize)

    eachBin = sampleSize // folds
    startIndex = foldNumber * eachBin
    endIndex = startIndex + eachBin

    if foldNumber == (folds-1):
        endIndex = sampleSize

    idxTest = idx[startIndex:endIndex]
    idxTrain = numpy.random.permutation(list(set(idx) - set(idxTest)))

    DTR = samples[:, idxTrain]
    DTE = samples[:, idxTest]
    LTR = labels[idxTrain]
    LTE = labels[idxTest]
    return (DTR, LTR), (DTE, LTE)









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

        for i in range(len(set(trainLabel))):
            theClassOfLabel = trainData [:, trainLabel == i]
            mu = mean(theClassOfLabel)
            sigma = covarinceMartix(theClassOfLabel)
            # if model == "N":
            #     I = numpy.eye(trainData.shape[0], trainData.shape[0])
            #     sigma = sigma *  I

            means[i] = mu
            sigmas[i] = sigma

        logSJoint = numpy.zeros(( len(set(testLabel)) ,testData.shape[1]))
        for i in range(len(set(testLabel))):
            muML = means[i]
            sigmaML = sigmas[i]
            scores = logpdf_GAU_ND(testData, muML, sigmaML) + numpy.log(prior)
            logSJoint[i,:] =  scores

        SMarginal = scipy.special.logsumexp(logSJoint, 0)
        logPostorior = logSJoint - SMarginal

        predictedLabels = logPostorior.argmax(0)

        return predictedLabels

    except Exception as e:
        print(e)

def accuracyError(testLabels, predeictedLabels):
    try:
        totalNumber = testLabels.shape[0]
        trueAssignment = sum(testLabels == predeictedLabels)
        return (trueAssignment/totalNumber) * 100, ( 1-(trueAssignment/totalNumber)) * 100

    except Exception as e:
        print(e)