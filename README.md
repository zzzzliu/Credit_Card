# Credit_Card
 Credit Card Fraud Detection Project based on `Pyspark` and `Logistic
 Regression`.

## Acknowledgement
* Some helper functions are based on [Northeastern University EECE
5698 - Parallel Processing for Data Analytics](http://catalog.northeastern.edu/course-descriptions/eece/)
* The idea and dataset of this project is based on [Kaggle Challenge -
Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Dataset
>The datasets contains transactions made by credit cards in September
2013 by european cardholders. This dataset presents transactions that
occurred in two days, where we have 492 frauds out of 284,807 transactions.
The dataset is highly unbalanced, the positive class (frauds) account for
0.172% of all transactions. It contains only numerical input variables
which are the result of a PCA transformation. Unfortunately, due to
confidentiality issues, we cannot provide the original features and more
background information about the data.

>Features V1, V2, ... V28 are the principal components obtained with PCA,
the only features which have not been transformed with PCA are 'Time' and
'Amount'. Feature 'Time' contains the seconds elapsed between each transaction
and the first transaction in the dataset. The feature 'Amount' is the
transaction Amount, this feature can be used for example-dependant
cost-senstive learning. Feature 'Class' is the response variable and it
takes value 1 in case of fraud and 0 otherwise.

>Given the class imbalance ratio, we recommend measuring the accuracy
using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix
accuracy is not meaningful for unbalanced classification.

All data have been shuffled and split as 5-fold, saved in `splitCSV/` folder.

## Algorithm
For this highly-unbalanced problem, the `AUC` and `Precision-Recall` is
utilized for the analysis of classifying performance. Since the number
of features is not so much, we employed `Logistic Regression with Ridge
Penalty` to build the classifier.

## Training
Since the dataset had been split as 5 folds, we use 4 folds for training
and cross validation, and the last one for testing. To train the model:
```
$ python CreditCard.py "splitCSV/" "output/" 4 --lamstart 0 --lamend 50
```
One shoud note that there are several parameters for the program, introduced as
below:
* Required
    ```
    data: Input data Directory
    output: Output directory
    folds: Number of folds
    ```
    So the first 3 parameters in the command line should be `data`,
    `output` and `folds`.

* Optional
    ```
    --epsilon: Desired objective accuracy.
    --gain: Gain for gradient descent
    --power: Gain Exponent

    --lamstart: Regularization parameter (start) for user features
    --lamend: Regularization parameter (end) for user features
    --laminterval: Regularization parameter (interval) for user features
    --bestlam: Best lambda after Cross Validation

    --maxiter: Maximum number of iterations
    --N: Parallelization Level
    --testData: Test data
    ```
    The epsilon controls the accuracy of final gradient descent stage. If
    the change of loss function is smaller than epsilon, we exit iteration.

    The gain and the power controls the speed of gradient descent and
    line search.

    At train stage, the user has to specify `lamstart`, `lamend`, and
    `laminterval`. The program will iterate lambda from `lamstart` to
    `lamend` with interval `laminterval`. It will also save all AUCs in
    `output` directory with each corresponding lambda, so we can find best lambda.
    One should note that the cross validation is performed in training
    stage. Given the input as 4 folds, we use 3 folds as training set and
    1 fold as cross-validation set iteratively. And calculate the average
    of 4 result AUCs to get the final result. This technique gives us a
    deeper insight about the data when they are highly unbalanced.

    After finding the best lambda (note lambda is just a scalar), the user
    can use the `bestlam` argument and the program will go to the test stage.

    The `maxiter` makes users have ability to control the max numbers of
    iterations. Because sometimes we cannot find the global optima (e.g.
    bad `gain` and `power` settings), we have to guarantee the program
    can exit normally.

    The `N` controls the level of parallelization. Recall the idea of
    MapReduce, `N` will partitions the fold as `N` parts, which will utilize
    the parallel-computation mechanism of Spark, and speed up the calculation
    of each fold.

    Finally, the `testData` is the index of the fold to be used as testing
    data.

## Testing

After training stage, we get the best lambda. But for logistic regression,
there is another significant parameter to be measured - Beta. Here, we
based on the best lambda, and use the first 4 folds (together, not cross
validation this time) to train the best beta. Then test our model on the testing set.

At previous training stage, every time we use 3 folds for training and 1 fold for
cross-validation iteratively, and got 4 AUCs. It seems we can also get best
beta at this point - average 4 betas. But this is not a good idea. Instead,
we train the best beta at testing stage and utilize all 4 folds to measure
it.

To train the Beta and test the model

```
$ python CreditCard.py "splitCSV/" "output/" 4 --bestlam 30 --testData 5
```

This will use all first 4 folds as training set to train the Beta, and use
the fifth fold as the test set. The best lambda is set `30` in this example.
In practical use, it should be got from previous training stage - the
lambda with highest average AUC.
