import array

import numpy

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

