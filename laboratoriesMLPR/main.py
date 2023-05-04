import numpy
import load
import plot
import util
import matplotlib.pyplot as plt
if __name__ == '__main__':

    data = load.Load("Data/iris.csv")
    data.prior = 1/3
    logSJoint_T = numpy.load("laboratories result/lab5 solution/tied/logSJoint_TiedMVG.npy")
    logPosterior_T = numpy.load("laboratories result/lab5 solution/tied/logPosterior_TiedMVG.npy")
    logMarginal_T = numpy.load("laboratories result/lab5 solution/tied/logMarginal_TiedMVG.npy")
    predictedGT = logPosterior_T.argmax(axis=0)

    trainSet, testSet = util.split_db_2to1(data.samples, data.labels)
    logPostorior, SMarginal, logSJoint = util.model_MVG(*trainSet, *testSet, data.prior, model="T")
    predicted = logPostorior.argmax(axis=0)
    print(logPosterior_T[1])
    print("-----------------------------------------------------------------------------------------------------")
    print(logPostorior[1])
    print(sum(predicted==predictedGT))