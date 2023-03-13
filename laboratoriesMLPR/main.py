import load
import plot
if __name__ == '__main__':
    data = load.Load("Data/iris.csv")
    plot.histopram(data.samples, data.labels)

