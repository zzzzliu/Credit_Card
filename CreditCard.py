# -*- coding: utf-8 -*-
import numpy as np
import argparse
import random
from time import time
from pyspark import SparkContext


def logisticFunction(beta, x):
    """
    Given a numpy array beta, a numpy array x, return the logistic function value
    :param beta: a numpy array β
    :param x: a numpy array x
    :return: logistic value
    """

    # print "in logisticFunction", beta.dot(x)

    np.seterr(all='ignore')
    return 1.0 / (1 + np.exp(-1.0 * beta.dot(x)))


def logisticLoss(beta,x,y):
    """  Given a numpy array beta, a numpy array x, and a binary value y in {-1,+1}, compute the logistic loss

                    l(β;x,y) = log( 1.0 + exp(-y * <β,x>) )

         The input is:
            - beta: a numpy array β
            - x: a numpy array x
            - y: a binary value in {-1,+1}

         The return is:
            - l: a double value

    """
    # print "in LogisticLoss", beta.dot(x)

    np.seterr(all='ignore')
    return np.log(1.0 + np.exp(-1.0 * y * beta.dot(x)))


def gradLogisticLoss(beta,x,y):
    """   Given a numpy array beta, a numpy array x, and a binary value y in {-1,+1}, compute the compute the
          gradient of the logistic loss

              ∇l(B;x,y) = -y / (1.0 + exp(y <β,x> )) * x

         The input is:
            - beta: a numpy array β
            - x: a numpy array x
            - y: a binary value in {-1,+1}

         The return is:
            - ∇l: a numpy array

    """

    # print "in gradLogisticLoss", beta.dot(x)

    np.seterr(all='ignore')
    return (-1.0 * y) / (1.0 + np.exp(y * beta.dot(x))) * x


def lineSearch(fun, x, grad, fx, gradNormSq, a=0.2, b=0.6):
    """ Given function fun, a current argument x, and gradient grad=∇fun(x),
        perform backtracking line search to find the next point to move to.
        (see Boyd and Vandenberghe, page 464).

        Both x and grad are presumed to be SparseVectors.

        Inputs are:
            - fun: the objective function f.
            - x: the present input (a Sparse Vector)
            - grad: the present gradient
            - fx: precomputed f(x)
            - grad: precomputed ∇f(x)
            - Optional parameters a,b  are the parameters of the line search.

        Given function fun, and current argument x, and gradient grad=∇fun(x), the function finds a t such that
        fun(x - t * ∇f(x)) <= f(x) - a * t * <∇f(x),∇f(x)>

        The return value is the resulting value of t.
    """
    t = 1.0
    while fun(x - t * grad) > fx - a * t * gradNormSq:
        t = b * t
    return t


def readBeta(input):
    """
        Read a vector β from CSV file input
    """
    with open(input,'r') as fh:
        str_list = fh.read()\
                     .strip()\
                     .split(',')
        return np.array( [float(val) for val in str_list] )


def writeBeta(output,beta):
    """
        Write a vector β to a CSV file ouptut
    """
    with open(output,'w') as fh:
        fh.write(','.join(map(str, beta.tolist()))+'\n')


def readDataRDD(input_file,spark_context):
    """  Read data from an input file. Each line of the file has the form

         V0, V1, ..., V29, y

         where Vs are features and y is binary lable 0 or 1

         The return value is an RDD containing tuples of the form

         (x,y) where x is numpy array and y is binary lable -1 or 1

    """
    return spark_context.textFile(input_file) \
                        .map(eval) \
                        .map(lambda values: (values[:-1], values[-1])) \
                        .map(lambda (features, label): (np.array([float(x) for x in features]), float(label))) \
                        .map(lambda (features, label): (features, -1 if label < 1 else 1))


def totalLossRDD(dataRDD,beta,lam = 0.0):
    """  Given a numpy array beta and a RDD dataset, compute the regularized total logistic loss :

                   L(β) = Σ_{(x,y) in data}  l(β;x,y)  + λ ||β ||_2^2

        Inputs are:
            - dataRDD: a RDD of key-value pairs of the form (x,y), where x is a numpy array and y is a binary value
            - beta: a numpy array β
            - lam: the regularization parameter λ
    """

    np.seterr(all='ignore')
    loss = dataRDD.map(lambda (x, y): logisticLoss(beta, x, y)) \
                  .reduce(lambda x, y: x + y)
    return loss + lam * beta.dot(beta)


def gradTotalLossRDD(dataRDD,beta,lam = 0.0):
    """  Given a numpy array beta and a RDD dataset, compute the gradient of regularized total logistic loss :

                  ∇L(β) = Σ_{(x,y) in data}  ∇l(β;x,y)  + 2λ β

        Inputs are:
            - dataRDD: a RDD of key-value pairs of the form (x,y), where x is a numpy array and y is a binary value
            - beta: a numpy array β
            - lam: the regularization parameter λ
    """

    np.seterr(all='ignore')
    loss = dataRDD.map(lambda (x, y): gradLogisticLoss(beta, x, y)) \
                  .reduce(lambda x, y: x + y)
    return loss + 2 * lam * beta


def test(dataRDD,beta):
    """ Output the Area Under the Curve (AUC) of the Receiver Operating Characteristic (ROC) Curve, as a evaluation of

        the prediction of labels in a RDD dataset under a given β.

        The True Positive Rate (TPR), False Positive Rate (FPR) are defined in terms of the following sets:

                 P = datapoints (x,y) in data for which f(<β,x>) > t
                 N = datapoints (x,y) in data for which f(<β,x>) <= t

                 TP = datapoints in (x,y) in P for which y=+1
                 FP = datapoints in (x,y) in P for which y=-1
                 TN = datapoints in (x,y) in N for which y=-1
                 FN = datapoints in (x,y) in N for which y=+1

        For #XXX the number of elements in set XXX, the TPR and FPR of parameter vector β over data with respect of the
        discrimination threshold t are defined as:

                 TPR(β,data,t) = #TP / #P
                 FPR(β,data,t) = #FP / #N
                 Recall = #TP / (#TP + #FN)

        Inputs are:
             - dataRDD: an testing RDD containing pairs of the form (x,y)
             - beta: numpy array β

        The return:
             - AUC
    """
    scoresLabels = dataRDD.map(lambda (x, y): (logisticFunction(beta, x), y)).sortByKey().cache()
    P = dataRDD.filter(lambda (x, y): y == 1).count()
    N = dataRDD.filter(lambda (x, y): y == -1).count()
    # print P
    # print N

    auc = 0.0
    count = 0.0
    for score, label in scoresLabels.collect():
        if label < 0:
            count += 1.0 / N
        else:
            auc += count / P

    scoresLabels.unpersist()
    return auc


def trueOrFalse(x, y, beta, tao):
    if logisticFunction(beta, x) > tao:
        if y > 0:
            return "TP"
        else:
            return "FP"
    else:
        if y < 0:
            return "TN"
        else:
            return "FN"


def ROCandPrecionRecall(dataRDD, beta):
    taos = np.arange(0,1,0.01)
    Recall = []
    Precision = []
    FPR = []
    for tao in taos:
        data = dataRDD.map(lambda (x, y): trueOrFalse(x, y, beta, tao)).countByValue()
        TP = data.get("TP", 0)
        FP = data.get("FP", 0)
        TN = data.get("TN", 0)
        FN = data.get("FN", 0)

        Recall.append(1.0 * TP / (TP + FN))
        Precision.append(1.0 * TP / (TP + FP))
        FPR.append(1.0 * FP / (FP + TN))
    return taos, Recall, Precision, FPR


def train(dataRDD, lam, beta_0, gain, power, max_iter = 100, eps = 0.1, test_data = None):
    """ Train a logistic classifier from deta.

    The function minimizes:

               L(β) = Σ_{(x,y) in data}  l(β;x,y)  + λ ||β||_2^2

        using gradient descent.

        Inputs are:
            - dataRDD: a RDD of key-value pairs of the form (x,y), where x is a numpy array and y is a binary value
            - lam: the regularization parameter λ
            - beta_0 (optional): an initial sparse vector β_0
            - max_iter (optional): the maximum number of iterations
            - eps (optional): the tolerance ε
            - test_data (optional): RDD data over which model β is tested in each iteration w.r.t. AUC

        The return values are:
            - beta: the trained β, as a numpy array
            - gradNorm: the norm ||∇L(β)||_2
            - k: the number of iterations

    """
    k = 0
    gradNorm = 2 * eps
    beta = beta_0
    fh1 = open(args.output + '_save_time_' + str(args.N), 'w')
    fh2 = open(args.output + '_save_gradNorm_' + str(args.N), 'w')
    start = time()
    while k < max_iter and gradNorm > eps:
        k += 1
        obj = totalLossRDD(dataRDD, beta, lam)

        grad = gradTotalLossRDD(dataRDD, beta, lam)
        gradNormSq = grad.dot(grad)
        gradNorm = np.sqrt(gradNormSq)

        # fun = lambda x: totalLossRDD(dataRDD, x, lam)
        # gamma = lineSearch(fun, beta, grad, obj, gradNormSq)
        gamma = gain / k ** power
        beta = beta - gamma * grad

        print 'k = ', k, '\tt = ', time() - start, '\tL(β_k) = ', obj, '\t||∇L(β_k)||_2 = ', gradNorm, \
                '\tgamma = ', gamma

        fh1.write(str(time() - start))
        fh1.write('\n')
        fh2.write(str(gradNorm))
        fh2.write('\n')
        # print beta
    fh1.close()
    fh2.close()
    if test_data:
        auc = test(test_data, beta)
        return beta, auc
    else:
        return beta, None


if __name__ == "__main__":
    # settings
    parser = argparse.ArgumentParser(description='Parallel Logistic Regression', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data', help='Data Directory')
    parser.add_argument('folds', type=int, help='Number of folds')
    parser.add_argument('--epsilon', default=0.01, type=float, help="Desired objective accuracy")
    parser.add_argument('--gain', default=0.001, type=float, help="Gain")
    parser.add_argument('--power', default=0.2, type=float, help="Gain Exponent")

    parser.add_argument('--lamstart', default=0, type=int, help="Regularization parameter (start) for user features")
    parser.add_argument('--lamend', default=50, type=int, help="Regularization parameter (end) for user features")
    parser.add_argument('--laminterval', default=1, type=int, help="Regularization parameter (interval) for user features")
    parser.add_argument('--bestlam', default=1, type=int, help="Best lambda after Cross Validation")

    parser.add_argument('--maxiter', default=100, type=int, help='Maximum number of iterations')
    parser.add_argument('--N', default=40, type=int, help='Parallelization Level')
    parser.add_argument('--output', default=None)

    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')

    args = parser.parse_args()
    sc = SparkContext(appName='Parallel Logistic Regression')
    if not args.verbose: sc.setLogLevel("ERROR")

    # generate k-folds
    # folds = readDataRDD(args.data, sc).randomSplit(weights = [1.0 / args.folds] * args.folds)
    folds = {}
    for k in range(args.folds):
        folds[k] = readDataRDD(args.data + "/fold" + str(k), sc)

    if args.bestlam:
        """
        k = random.randint(0, args.folds - 1)
        train_folds = [folds[j] for j in range(args.folds) if j is not k]
        trainRDD = train_folds[0]
        for fold in train_folds[1:]:
            trainRDD = trainRDD.union(fold)
        trainRDD.repartition(args.N).cache()
        testRDD = folds[k].repartition(args.N).cache()

        beta_0 = np.zeros(29)
        beta, auc = train(trainRDD, args.bestlam, beta_0, args.gain, args.power, args.maxiter, args.epsilon, testRDD)
        taos, Recall, Precision, FPR = ROCandPrecionRecall(testRDD, beta)

        print 'Saving taos in', args.output + '_save_taos_' + str(args.bestlam)
        fh = open(args.output + '_save_taos_' + str(args.bestlam), 'w')
        fh.write(str(taos.tolist()))
        fh.close()
        print 'Saving Recall in', args.output + '_save_Recall_' + str(args.bestlam)
        fh = open(args.output + '_save_Recall_' + str(args.bestlam), 'w')
        fh.write(str(Recall))
        fh.close()
        print 'Saving Precision in', args.output + '_save_Precision_' + str(args.bestlam)
        fh = open(args.output + '_save_Precision_' + str(args.bestlam), 'w')
        fh.write(str(Precision))
        fh.close()
        print 'Saving FPR in', args.output + '_save_FPR_' + str(args.bestlam)
        fh = open(args.output + '_save_FPR_' + str(args.bestlam), 'w')
        fh.write(str(FPR))
        fh.close()
        trainRDD.unpersist()
        testRDD.unpersist()
        """

        trainRDD = folds[0]
        for i in range(1, args.folds):
            trainRDD.union(folds[i])
        trainRDD.repartition(args.N).cache()
        beta_0 = np.zeros(29)
        beta, auc = train(trainRDD, args.bestlam, beta_0, args.gain, args.power, args.maxiter, args.epsilon)
        trainRDD.unpersist()
        print 'Saving Beta in', args.output + '_save_Beta_' + str(args.bestlam)
        fh = open(args.output + '_save_Beta_' + str(args.bestlam), 'w')
        fh.write(str(beta))
        fh.close()

    else:
        for lam in range(args.lamstart, args.lamend + args.laminterval, args.laminterval):
            # cross validation
            cross_val_auc = []
            for k in range(args.folds):
                train_folds = [folds[j] for j in range(args.folds) if j is not k]
                trainRDD = train_folds[0]
                for fold in train_folds[1:]:
                    trainRDD = trainRDD.union(fold)
                trainRDD.repartition(args.N).cache()
                testRDD = folds[k].repartition(args.N).cache()
                Mtrain = trainRDD.count()
                Mtest = testRDD.count()
                print("Initiating fold %d with %d train samples and %d test samples" % (k, Mtrain, Mtest))


                beta_0 = np.zeros(29)
                beta, auc = train(trainRDD, lam, beta_0, args.gain, args.power, args.maxiter, args.epsilon, testRDD)
                print 'AUC for fold %d' %k, auc

                cross_val_auc.append(auc)
                trainRDD.unpersist()
                testRDD.unpersist()

            if args.output is None:
                print "%d-fold cross validation AUC is: %f " % (args.folds, np.mean(cross_val_auc))
            else:
                print 'Saving trained auc in', args.output + '_save_auc_' + str(lam)
                fh = open(args.output + '_save_auc_' + str(lam), 'w')
                fh.write(str(np.mean(cross_val_auc)))
                fh.close()
