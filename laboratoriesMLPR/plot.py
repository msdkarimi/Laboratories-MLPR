import numpy
import matplotlib.pyplot as plot
def scatter(samples, labels):
    pass

def histopram(samples, labels):
    class0Mask = labels == 0
    class1Mask = labels == 1
    class2Mask = labels == 2


    samplesClass0 = samples[:, class0Mask]
    samplesClass1 = samples[:, class1Mask]
    samplesClass2 = samples[:, class2Mask]

    for feature in range(samples.shape[0]):
        plot.hist(samplesClass0[feature,:], density=True, bins=10, alpha = 0.4, label="Iris-setosa")
        plot.hist(samplesClass1[feature,:], density=True, bins=10, alpha = 0.4,label="Iris-versicolor")
        plot.hist(samplesClass2[feature,:], density=True, bins=10, alpha = 0.4,label="Iris-virginica")
        plot.legend
        plot.show()
        plot.savefig('hist_%d.png' % feature+1)