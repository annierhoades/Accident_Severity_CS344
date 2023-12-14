import argparse
import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.feature_selection as skf
from sklearn import preprocessing
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

import random


def test_knn(df, y):
    """
    Perform basic knn with sklearn.

    Parameters
    ----------
    df : pandas dataframe mxn
        All data
    y : vector
        Labels associated with df samples
    """

    # Flatten y
    y = np.ravel(y)

    # Create a new kNN model
    knn = KNeighborsClassifier()

    # Split into train / test -> 0.8 / 0.2
    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)
    #print(df.shape, " vs ", x_train.shape, " and ", x_test.shape)

    # Make sure that the computer doesnt get mad
    x_train=np.ascontiguousarray(x_train)
    x_test=np.ascontiguousarray(x_test)
    y_train=np.ascontiguousarray(y_train)
    y_test=np.ascontiguousarray(y_test)

    # Train the model -> load the training data
    knn.fit(x_train, y_train)

    # Predict the test data
    y_pred = knn.predict(x_test)

    # Check the classification report
    print(classification_report(y_test, y_pred))


def test_knn2(df, y, metricz, neighborz):
    """
    Perform knn with different distance metrics and k values.

    Parameters
    ----------
    df : pandas dataframe mxn
        All data
    y : vector
        Labels associated with df samples
    metricz : String value
        Distance metric
    neighborz : int value
        k value
    """

    # Flatten y
    y = np.ravel(y)

    # Create a new kNN model
    knn = KNeighborsClassifier(n_neighbors=neighborz,metric=metricz)

    # Split into train / test -> 0.8 / 0.2
    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)
    #print(df.shape, " vs ", x_train.shape, " and ", x_test.shape)

    # Make sure that the computer doesnt get mad
    x_train=np.ascontiguousarray(x_train)
    x_test=np.ascontiguousarray(x_test)
    y_train=np.ascontiguousarray(y_train)
    y_test=np.ascontiguousarray(y_test)

    # Train the model -> load the training data
    knn.fit(x_train, y_train)

    # Predict the test data
    y_pred = knn.predict(x_test)

    # Check the classification report
    print(classification_report(y_test, y_pred))


def tune(df, y):
    """
    Test out knn with different distance metrics and k values.

    Parameters
    ----------
    df : pandas dataframe mxn
        All data
    y : vector
        Labels associated with df samples
    """

    # Flatten y
    y = np.ravel(y)

    # Make sure that the computer doesnt get mad
    df=np.ascontiguousarray(df)
    y=np.ascontiguousarray(y)

    # Choose a random set of 10% of the sample indices
    indices = random.sample(range(df.shape[0]), int(df.shape[0]*0.1))
    #print(len(indices), ' vs ', df.shape[0])

    # Save the subset of samples
    df_val = np.take(df, indices, 0)
    y_val = np.take(y, indices, 0)

    # Change these hyperparameters to tune
    metric = ['euclidean'] # ['manhattan'] ['minkowski']
    n_neighbors = list(range(1,802,100))

    # Test out the different metrics
    for m in metric:
        for neigh in n_neighbors:
            #print("parameters: ", metric, " and ", neigh)
            test_knn2(df=df_val, y=y_val, metricz=m, neighborz=neigh)


def main():
    """
    Main file to run from the command line.
    """
    # load the data

    # the entire dataset -> just for testing
    # xFull = pd.read_csv('US_Accidents_March23.csv')

    # the pre-processed dataset (result of running preprocessing.py)
    xPre = pd.read_csv('data.csv')
    y = pd.read_csv('y.csv')


    # plot the target distribution
    '''
    xPre['target'] = y

    fig,ax = plt.subplots()
    sb.countplot(data=xPre, x='target')
    plt.show()
    '''

    # basic knn
    test_knn(xPre,y)

    # test diff hyperparameters
    #tune(xPre,y)

    # test other hyperparameters
    '''
    indices = random.sample(range(xPre.shape[0]), int(xPre.shape[0]*0.1))
    print(len(indices), ' vs ', xPre.shape[0])

    df_val = np.take(xPre, indices, 0)
    y_val = np.take(y, indices, 0)
    test_knn2(xPre, y, 'euclidean', 10)
    test_knn2(xPre, y, 'euclidean', 25)
    test_knn2(xPre, y, 'euclidean', 50)
    '''


if __name__ == "__main__":
    main()
