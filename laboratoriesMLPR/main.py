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
    logMarginal_MVG = numpy.load("laboratories result/lab5 solution/g/logMarginal_MVG.npy")

    # trainSet, testSet = util.split_db_2to1(data.samples, data.labels)
    #
    # peredictions = util.model_MVG(*trainSet, *testSet, data.prior, model= "N")
    # _, testLabels = testSet
    # acc, err = util.accuracyError(testLabels, peredictions)

    k = 3
    predictedLabels = []
    originalLabels = []

    for i in range(k):
        trainSet, testSet = util.kfold(data.samples, data.labels, i, k)
        d, l = trainSet
        _, labels = testSet
        peredictions = util.model_MVG(*trainSet, *testSet, data.prior, model="G")
        predictedLabels.append(peredictions)
        originalLabels.append(labels)



    originalLabels = numpy.hstack(originalLabels)
    predictedLabels = numpy.hstack(predictedLabels)
    print(1-(sum(originalLabels ==predictedLabels )/150))
