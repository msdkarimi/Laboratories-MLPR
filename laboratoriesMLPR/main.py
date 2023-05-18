import numpy
import load
import plot
import util
from util import MyKFold
import matplotlib.pyplot as plt
if __name__ == '__main__':

    data = load.Load("Data/iris.csv")

    d, l = util.load_iris_binary(data.samples, data.labels)
    (t, tl), (te, tel) = util.split_db_2to1(d,l)

    targetPrior = 0.5
    fpC = 1
    fnC = 1

    pi_tilde = (targetPrior * fnC) / (targetPrior * fnC + ((1 - targetPrior) * fpC))

    scores, pLabels = util.train_logReg_prior_weighted(t, tl, te, priorTarget= targetPrior, lambdaa=0.001)

    log_odds =  numpy.log(targetPrior/(1-targetPrior))
    llr = scores - log_odds

    predicted_ = llr.ravel() > numpy.log(pi_tilde / (1 - pi_tilde))

    err = 100 - (sum(tel == predicted_) / te.shape[1]) * 100
    print(err)

    cm, tpR, fpR, fnR, tnR = util.confusion_matrix(tel, predicted_)
    print(cm)
    print(f"tpR= {tpR}, tnR = {tnR}, fpR= {fpR}, fnR= {fnR}")
    dcf = util.normal_dcf(fnR, fpR, fnC, fpC, targetPrior)
    print(dcf)

