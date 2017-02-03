import pandas as pd
import numpy as np
import time
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

DATA_DIR = "/home/asu/Projects/kaggle/digit_recognizer/data/"
TRAINING_DATA = DATA_DIR + "train.csv"
LABEL_COLUMN = 0
FIRST_DATA_COLUMN = 1


def readData(path, split = 0.3):

    data = pd.read_csv(path);
    features = data.iloc[:,1:]
    y = data.iloc[:,0]
    scaler = preprocessing.MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(features), columns = features.columns)
    splitted = train_test_split(X, y, test_size = split, stratify = y)
    Xtrain = splitted[0]
    Xtest = splitted[1]
    ytrain = splitted[2]
    ytest = splitted[3]
    return Xtrain, ytrain, Xtest, ytest

def logistic_model(Xtrain, ytrain, Xtest, ytest):
    '''Return one v rest logistic model with regularization'''

    # Getting about 91.3% classification accuracy for 3 fold validation across
    # regularization range of [0.03,0.3,3,30,300]
    # Best regulariation: 0.3

    # parameters = {'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]}
    parameters = {'C':[0.01,0.03]}

    logit = LogisticRegression(C = 1, n_jobs = -1, solver = 'sag', max_iter = 100)
    fitted_model = GridSearchCV(estimator = logit, param_grid = parameters, n_jobs = 4)
    fitted_model.fit(Xtrain, ytrain)
    return fitted_model

def neural_net(Xtrain, ytrain):
    '''Neural network'''

    # Clocking in at about 96.5% for a 1 hidden layer with 500 nodes
    # start = time.time()

    network = MLPClassifier((500,), activation="logistic", max_iter=500, early_stopping = True,
            validation_fraction = 0.1)
    network.fit(Xtrain, ytrain)
    # end = time.time()
    # print (end-start)
    return network

def SVM(Xtrain, ytrain, Xtest, ytest):
    '''One v rest SVM'''
    # start = time.time()

    # Getting about 92.8% off the shelf
    #parameters = {'C': [0.03, 0.3, 3, 30, 300], 'kernel': ['rbf', 'poly'], 'gamma': [0.003,0.03,0.3,3,30]}

    svm = SVC()
    # fitted_model = GridSearchCV(estimator = svm, param_grid = parameters, n_jobs = 8)
    svm.fit(Xtrain, ytrain)
    # end = time.time()
    # print (end-start)
    return svm

if __name__ == '__main__':
    Xtrain, ytrain, Xtest, ytest = readData(TRAINING_DATA)
