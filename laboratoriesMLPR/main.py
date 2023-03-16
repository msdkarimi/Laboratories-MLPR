import load
import plot
import util
if __name__ == '__main__':
    try:
        data = load.Load("Data/iris.csv")
        # plot.histopram(data.samples, data.labels)
        # plot.scatter(data.samples, data.labels)
        tr, tes = util.kfold(data.samples, 1)
        print(tes.shape)

    except Exception as e:
        print(e)