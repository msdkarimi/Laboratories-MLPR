import numpy
import load
import plot
import util
import matplotlib.pyplot as plt
if __name__ == '__main__':
    try:
        data = load.Load("Data/iris.csv")


        plt.figure()
        XPlot = numpy.linspace(-8, 12, 1000)
        m = numpy.ones((1, 1)) * 1.0
        C = numpy.ones((1, 1)) * 2.0
        plt.plot(XPlot.ravel(), numpy.exp(util.logpdf_GAU_ND(util.toRow(XPlot), m, C)))
        plt.show()




    except Exception as e:
        print(f"exception in main->>> {e}")