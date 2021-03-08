import pandas as pd
import numpy as np
from numpy import mean, std
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import rcParams

def one_hot_encode(df, categoric_col_name, new_col_names):

    a =pd.get_dummies(list(df[categoric_col_name]))
    a.columns = new_col_names
    df = pd.concat([df, a], axis = 1) # could remove one of these, but having them all is clearer and little benefit to removing them
    df = df.drop([categoric_col_name], axis = 1)
    
    return df

def select_k_best(df, k= 10):
    y = df['label']
    X = df.drop(['label', 'index'], axis = 1)
    bestfeatures = SelectKBest(score_func=chi2, k=k) #this is adapted from the docs
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Feature','Score']  #naming the dataframe columns
    print(featureScores.nlargest(k,'Score'))  #print 10 best features
    
    return list(featureScores.nlargest(k,'Score')['Feature'])

def scale_features(df):
    col_names = list(df.columns)
    col_names.remove('label')
    print(col_names)
    for col in col_names:
        #scaler = StandardScaler()
        scaler = MinMaxScaler(feature_range=(0, 0.5))
        df[col] = scaler.fit_transform(df[col].to_numpy().reshape(-1, 1))#
    # this is kept as a separate func as I originally has the scalers as saving as pickles 
    # which is required for deployment
    # also easier for playing with processing different columns
    return df

def plot_correlations(df):
    rcParams['figure.figsize'] = 20, 14
    plt.matshow(df.corr(), cmap='gnuplot')
    plt.yticks(np.arange(df.shape[1]), df.columns)
    plt.xticks(np.arange(df.shape[1]), df.columns)
    plt.colorbar()
    plt.show()


def calc_mean_std(nums):
    mean, stddev = np.mean(nums), np.std(nums)
    mean = np.around(mean, decimals=2)
    stddev = np.around(stddev, decimals=2)
    return mean, stddev
