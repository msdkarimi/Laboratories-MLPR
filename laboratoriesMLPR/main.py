import numpy

import load
import plot
import util
if __name__ == '__main__':
    try:
        data = load.Load("Data/iris.csv")
        # plot.histopram(data.samples, data.labels)
        # plot.scatter(data.samples, data.labels)
        pca4 = numpy.load("laboratories result/IRIS_PCA_matrix_m4.npy")

        mapped, P = util.PCA(data.samples, util.covarinceMartix(data.samples), 4)
        print(P)
        print(mapped.shape)
        plot.scatter(mapped, data.labels)




    except Exception as e:
        print(e)