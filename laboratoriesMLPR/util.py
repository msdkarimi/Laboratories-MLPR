import array

import numpy
import scipy

def toCol(theArray:numpy.array):
    try:
        return theArray.reshape(theArray.size, 1)
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


def kfold(samples: numpy.array, foldNumber: int, folds: int = 4):
    try:
        sampleSize = samples.shape[1]
        eachBin = sampleSize // folds

        startIndex = foldNumber * eachBin
        endIndex = startIndex + eachBin

        allrRangesIndex = set(range(sampleSize))
        testIndex = set(range(startIndex, endIndex))
        trainIndex = allrRangesIndex - testIndex

        getTestChunk = samples[:, list(testIndex)]
        getTrainChunk = samples[:, list(trainIndex)]

        return getTrainChunk, getTestChunk

    except Exception as e:

        print(e)


def PCA(samples: numpy.array, covarianceMatrix: numpy.array, m: int = 4)->tuple:

    try:
        U, s, _ = numpy.linalg.svd(covarianceMatrix)
        P = U[:, 0:m]
        mapped = numpy.dot(P.T, samples)
        return mapped, P

    except Exception as e:
        print(e)

def LDA(samples: numpy.array, labels: numpy.array, m: int = 2):

    try:
        sW, sB = getSW_SB(samples, labels)
        s, U = scipy.linalg.eigh(sB, sW)
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