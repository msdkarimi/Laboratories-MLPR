import numpy
import load
import plot
import util
from util import MyKFold
import matplotlib.pyplot as plt
if __name__ == '__main__':

    data = load.Load("Data/iris.csv")

    kf = MyKFold(n_splits=4)
    kf.kfSplite(data.samples, data.labels)
    kf.allFolds[0]

    predictions = list()
    GT = list()
    for train, test in kf.allFolds:
        t,l = test
        GT.append(l)
        logPostorior, SMarginal, logSJoint = util.model_MVG(*train, *test, data.prior, model="G")
        predictions.append(logPostorior.argmax(axis=0))

    print(numpy.hstack(predictions))
    print(numpy.hstack(GT))
    print( sum(numpy.hstack(GT) == numpy.hstack(predictions)))






    # data.prior = 1/3
    # logSJoint_T = numpy.load("laboratories result/lab5 solution/tied/logSJoint_TiedMVG.npy")
    # logPosterior_T = numpy.load("laboratories result/lab5 solution/tied/logPosterior_TiedMVG.npy")
    # logMarginal_T = numpy.load("laboratories result/lab5 solution/tied/logMarginal_TiedMVG.npy")
    # predictedGT = logPosterior_T.argmax(axis=0)
    #
    # trainSet, testSet = util.split_db_2to1(data.samples, data.labels)
    # logPostorior, SMarginal, logSJoint = util.model_MVG(*trainSet, *testSet, data.prior, model="T")
    # predicted = logPostorior.argmax(axis=0)
    # print(logPosterior_T[1])
    # print("-----------------------------------------------------------------------------------------------------")
    # print(logPostorior[1])
    # print(sum(predicted==predictedGT))