import argparse
import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.feature_selection as skf
from sklearn import preprocessing
import seaborn as sb
import matplotlib.pyplot as plt


def cal_corr(df):
    """
    Print the feature pairs which have a high Pearson correlation.

    This program prints the feature pairs rather than automatically removing
    them so that you may choose which feature of the pair to delete.

    Parameters
    ----------
    df : pandas dataframe mxn
        All data
    """
    # calculate the correlation matrix
    correlation_matrix = df.corr()

    correlated_features = [] # list to hold the highly correlated feature pairs
    features = {} # dict to hold all the feature names

    # loop that forms a dict with all the variables {featureName: [correlatedMatch(es),marker]}
    z= 0
    for row1 in correlation_matrix:
        features[z] = [row1,0]
        z+=1

    # loop that goes through the correlation matrix and checks for values > 0.7
    i = 0
    for row in correlation_matrix:
        j = 0
        curr = correlation_matrix[row]
        for elem in curr:
            if abs(elem) >= 0.70:
                if row!=str(features[j][0]):
                    if type(features[j][1]) != list:
                        correlated_features.append(str(row)+" / "+str(features[j][0]))
                        features[j][1] = [row]
                        features[i][1] = [features[j][0]]
                    elif not (row in features[j][1]):
                        correlated_features.append(str(row)+" / "+str(features[j][0]))
                        features[j][1].append(row)
                        features[i][1].append(features[j][0])
            j+=1
        i+=1


    # loop that prints the correlated features
    for x in correlated_features:
        print(x)


def cal_corr_target(df, y):
    """
    Print the features which have a low Pearson correlation with the target.

    This program prints the feature pairs rather than automatically removing
    them so that you may decide what to do.

    Parameters
    ----------
    df : pandas dataframe mxn
        All data
    y : pandas dataframe nx1
        Target vector
    """
    # combine the dataframe and the target vector
    df['y'] = y

    # calculate the correlation matrix
    correlation_matrix = df.corr()

    correlated_features = [] # list to hold the low correlated feature pairs
    features =[] # list to hold all the feature names

    # loop that forms a list with all the variable names
    for row1 in correlation_matrix:
        features.append(row1)


    curr = correlation_matrix['y'] # only the correlation values between y and other variables
    # loop that goes through the correlation values, checking for < 0.01
    q = 0
    for elem in curr:
        if abs(elem) < 0.01:
            print(features[q], ": ", elem)
            correlated_features.append(features[q])
        q += 1

    # loop that prints the correlated features
    for x in correlated_features:
        print(x)



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

    # size comparisons
    #print(xPre.shape, " vs ", y.shape)
    #print(xFull.shape, " vs ", xPre.shape)


    # Pearson correlation calculations
    #cal_corr(xPre)
    #cal_corr_target(xPre,y)


    # ----- drop features + write to csv ------

    # these are the features we found, feel free to edit with your own
    xPre = xPre.drop(columns=[
        'Civil_Twilight',
        'Nautical_Twilight',
        'Astronomical_Twilight',
        'Wind_Chill(F)',
        'Bump',
        'Roundabout',
        'Give_Way'
    ])


    # convert to csv
    print('to csv')
    xPre.to_csv('data.csv', index=False)




if __name__ == "__main__":
    main()
