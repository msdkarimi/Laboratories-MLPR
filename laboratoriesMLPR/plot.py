import numpy
import matplotlib.pyplot as plot

def scatter(samples, labels):
    class0Mask = labels == 0
    class1Mask = labels == 1
    class2Mask = labels == 2

    samplesClass0 = samples[:, class0Mask]
    samplesClass1 = samples[:, class1Mask]
    samplesClass2 = samples[:, class2Mask]

    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
    }

    for i in range(samples.shape[0]):
        for j in range(samples.shape[0]):
            if i == j:
                continue
            else:
                plot.scatter(samplesClass0[i, :], samplesClass0[j, :], label="Iris-setosa")
                plot.scatter(samplesClass1[i, :], samplesClass1[j, :], label="Iris-versicolor")
                plot.scatter(samplesClass2[i, :], samplesClass2[j, :], label="Iris-virginica")
                plot.xlabel(hFea[i])
                plot.ylabel(hFea[j])
                plot.legend()
                plot.show()



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
        plot.legend()
        plot.show()
        # plot.savefig('hist_%d.png' % feature+1)