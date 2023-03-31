import numpy
import load
import plot
import util
import matplotlib.pyplot as plt
if __name__ == '__main__':
    try:
        data = load.Load("Data/iris.csv")

        XND = numpy.load("laboratories result/lab4 solution/X1D.npy")

        mu = util.mean(XND)
        C = util.covarinceMartix(XND)

        # mu = numpy.ones((1, 1)) * 1.0
        # C = numpy.ones((1, 1)) * 2.0

        ll = util.logpdf_GAU_ND(XND, mu, C)

        plt.figure()
        plt.hist(XND.ravel(), bins=100, density=True)
        XPlot = numpy.linspace(-8, 12, 1000)
        plt.plot(XPlot.ravel(), numpy.exp(util.logpdf_GAU_ND(util.toRow(XPlot), mu, C)))
        plt.show()

        print(sum(ll))


    except Exception as e:
        print(f"exception in main->>> {e}")