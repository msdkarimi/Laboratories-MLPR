import numpy

def toCol(theArray:numpy.array):
    return theArray.reshape(theArray.size, 1)

def mean(samples: numpy.array):
    return samples.mean(1).reshape(samples.shape[0], 1)

def centerSampels (samples: numpy.array):
    mean = samples.mean(1).reshape(samples.shape[0], 1)
    return samples - mean

