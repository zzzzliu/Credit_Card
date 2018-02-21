import numpy as np


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
        return np.array([float(val) for val in str_list])


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