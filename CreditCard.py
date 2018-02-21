# -*- coding: utf-8 -*-
import numpy as np, argparse, os
from time import time
from pyspark import SparkContext
from shutil import rmtree
from LogisticRegression import *


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def test(dataRDD, beta):
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


def train(dataRDD, lam, beta_0, gain, power, max_iter=100, eps=0.1):
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
    fh1 = open(args.output + 'save_time_' + str(args.N), 'w')
    fh2 = open(args.output + 'save_gradNorm_' + str(args.N), 'w')
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
    return beta


if __name__ == "__main__":
    # settings
    parser = argparse.ArgumentParser(description='Parallel Logistic Regression', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data', help='Data Directory')
    parser.add_argument('output', default='output', help='output directory')
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
    parser.add_argument('--testData', help='Testing data')

    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')

    args = parser.parse_args()
    sc = SparkContext(appName='Parallel Logistic Regression')
    if not args.verbose:
        sc.setLogLevel("ERROR")

    # generate k-folds
    # folds = readDataRDD(args.data, sc).randomSplit(weights = [1.0 / args.folds] * args.folds)
    folds = {}
    for k in range(args.folds):
        folds[k] = readDataRDD(args.data + "/fold" + str(k), sc)

    ensure_dir(args.output)

    if args.bestlam:
        # if we found the best lambda from previous training and cross-validation,
        # then we can train on best beta and test our model.
        trainRDD = folds[0]
        for i in range(1, args.folds):
            trainRDD.union(folds[i])
        trainRDD.repartition(args.N).cache()
        beta_0 = np.zeros(29)
        beta = train(trainRDD, args.bestlam, beta_0, args.gain, args.power, args.maxiter, args.epsilon)
        trainRDD.unpersist()
        print 'Saving Beta in', args.output + 'save_Beta_' + str(args.bestlam)
        fh = open(args.output + 'save_Beta_' + str(args.bestlam), 'w')
        fh.write(str(beta))
        fh.close()

        if args.testData:
            testData = readDataRDD(args.data + "/fold" + args.testData, sc)
            auc = test(testData, beta)
            print "The testing result: AUC = " + auc

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
                print 'Saving trained auc in', args.output + 'save_auc_' + str(lam)
                fh = open(args.output + 'save_auc_' + str(lam), 'w')
                fh.write(str(np.mean(cross_val_auc)))
                fh.close()
