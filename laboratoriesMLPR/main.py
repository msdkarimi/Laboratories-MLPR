import numpy
import load
import plot
import util
import matplotlib.pyplot as plt
if __name__ == '__main__':

    data = load.Load("Data/iris.csv")
    data.prior = 1/3
    logSJoint_MVG = numpy.load("laboratories result/lab5 solution/g/logSJoint_MVG.npy")
    logPosterior_MVG = numpy.load("laboratories result/lab5 solution/g/logPosterior_MVG.npy")
    print(logPosterior_MVG)
    logMarginal_MVG = numpy.load("laboratories result/lab5 solution/g/logMarginal_MVG.npy")

    trainSet, testSet = util.split_db_2to1(data.samples, data.labels)

    log_postterior = util.model_MVG(*trainSet, *testSet, data.prior, model= "N")
    print(log_postterior)