import numpy
import load
import plot
import util
import matplotlib.pyplot as plt
if __name__ == '__main__':
    try:
        data = load.Load("Data/iris.csv")
        # plot.histopram(data.samples, data.labels)
        # plot.scatter(data.samples, data.labels)
        # pca4 = numpy.load("laboratories result/IRIS_PCA_matrix_m4.npy")

        # mapped, P = util.PCA(data.samples, util.covarinceMartix(data.samples), 4)
        # tst = numpy.dot(P, mapped)
        # print(tst[:,149]-data.samples[:,149])

        lda2 = numpy.load("laboratories result/IRIS_LDA_matrix_m2.npy")
        print(lda2)
        d, r= util.LDA(data.samples, data.labels)
        print(r)








    except Exception as e:

        print(f"exception in main->>> {e}")