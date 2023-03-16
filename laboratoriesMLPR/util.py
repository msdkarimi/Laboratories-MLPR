import array

import numpy

def toCol(theArray:numpy.array):
    return theArray.reshape(theArray.size, 1)

def mean(samples: numpy.array):
    return samples.mean(1).reshape(samples.shape[0], 1)

def centerSampels (samples: numpy.array):
    mean = samples.mean(1).reshape(samples.shape[0], 1)
    return samples - mean


def kfold(samples: numpy.array, foldNumber: int, folds: int = 4):
    try:
        sampleSize = samples.shape[1]
        eachBin = sampleSize // folds

        startIndex = foldNumber * eachBin
        endIndex = startIndex + eachBin

        allrRangesIndex = set(range(sampleSize))
        testIndex = set(range(startIndex, endIndex))
        trainIndex = list(allrRangesIndex - testIndex)

        getTestChunk = samples[:, list(testIndex)]
        getTrainChunk = samples[:, trainIndex]

        return getTrainChunk, getTestChunk

    except Exception as e:

        print(e)

