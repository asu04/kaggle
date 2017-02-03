import pandas as pd
import numpy as np
import time
import json
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from matplotlib.backends.backend_pdf import PdfPages

DATA_DIR = "/home/asu/Projects/kaggle/digit_recognizer/data/"
TRAINING_DATA = DATA_DIR + "train.csv"
TUNING_DIR = "/home/asu/Projects/kaggle/digit_recognizer/tuning/"
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

def writeAndPlot(function):

    def wrapper(*args, **kwargs):

        fitted_model = function(*args, **kwargs)
        with open(TUNING_DIR + function.__name__ + ".txt", "w") as writefile:
            json.dump(fitted_model.best_estimator_.get_params(), writefile, indent = 0)

        cv = fitted_model.cv_results_
        plotData = {'average_test_score':[]}

        for i in range(0, len(cv['params'])):
            for key in cv['params'][i]:

                if not key in plotData:
                    plotData[key] = []

                plotData[key].append(cv['params'][i][key])

            folds = 3
            scores = []

            if not fitted_model.cv == None:
                folds = fitted_model.cv
            for j in range(0,folds):
                scores.append(cv["split"+str(j)+"_test_score"][i])
            plotData['average_test_score'].append(np.mean(scores))

        plotDataFrame = pd.DataFrame.from_dict(plotData)
        columns = set(plotDataFrame.columns)
        columns.remove('average_test_score')
        pp = PdfPages(TUNING_DIR + function.__name__ + ".pdf")
        for col in columns:
            fig, ax = plt.subplots(figsize = (8,6))
            for i, group in plotDataFrame.groupby([x for x in columns if not (x == col)]):
                group.plot(x = col, y = 'average_test_score', ax = ax, label = i)
            plt.legend()
            pp.savefig()
        pp.close()

    return wrapper

@writeAndPlot
def logistic_model(Xtrain, ytrain, Xtest, ytest):
    '''Do grid search for logistic regression and write params to file'''

    # Getting about 91.3% classification accuracy for 3 fold validation across
    # regularization range of [0.03,0.3,3,30,300]
    # Best regulariation: 0.3

    # parameters = {'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]}
    parameters = {'C': [0.01,0.03,0.1], 'max_iter': [50,100]}

    logit = LogisticRegression(C = 1, n_jobs = -1, solver = 'sag')
    fitted_model = GridSearchCV(estimator = logit, param_grid = parameters, n_jobs = -1)
    fitted_model.fit(Xtrain, ytrain)
    return fitted_model

def neural_net(Xtrain, ytrain):
    '''Neural network'''

    # Clocking in at about 96.5% for a 1 hidden layer with 500 nodes
    # start = time.time()

    parameters = {'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01]}

    network = MLPClassifier((500,), activation="logistic", max_iter=500, early_stopping = True,
            validation_fraction = 0.1)
    fitted_model = GridSearchCV(estimator = network, param_grid = parameters, n_jobs = -1)
    fitted_model.fit(Xtrain, ytrain)
    return network

def SVM(Xtrain, ytrain, Xtest, ytest):
    '''One v rest SVM'''

    # Getting about 92.8% off the shelf
    parameters = {'C': [0.03, 0.3, 3, 30, 300], 'kernel': ['rbf', 'poly'], 'gamma': [0.003,0.03,0.3,3,30]}
    svm = SVC()
    fitted_model = GridSearchCV(estimator = svm, param_grid = parameters, n_jobs = 8)
    fitted_model.fit(Xtrain, ytrain)
    return fitted_model

if __name__ == '__main__':
    Xtrain, ytrain, Xtest, ytest = readData(TRAINING_DATA)
