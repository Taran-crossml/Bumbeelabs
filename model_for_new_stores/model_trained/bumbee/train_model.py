# import the required libraries.
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm      # libraries to get the confidence interval.
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import seaborn as sns
from yellowbrick.regressor import CooksDistance
import sklearn
import pickle
from sklearn import linear_model
import openpyxl


def make_directory(path, folder_name):
    make_path = path + folder_name
    try:
        os.mkdir(make_path)
    except Exception as e:
        pass

def insert_day(data):
    data=data[(data["total"]!=0)&(data["count_randomised"]!=0)].reset_index().fillna(0)
    data['time'] = pd.to_datetime(data['time'])
    data['day'] = data['time'].dt.day_name()
    data.drop(['perc_randomised', 'perc_vendor','apple','not_apple'], inplace= True, axis=1)
    data = pd.get_dummies(data)
    data.rename(columns = {'day_Friday':'Friday', 'day_Monday':'Monday', 'day_Tuesday':'Tuesday', 'day_Wednesday':'Wednesday',
                        'day_Thursday':'Thursday','day_Saturday':'Saturday','day_Sunday':'Sunday'}, inplace = True)
    return data


def clean_data(df):
    df['time'] = pd.to_datetime(df['time'])
    df=df[(df["total"]!=0)&(df["count_vendor"]!=0)].reset_index().fillna(0)
    df=df.drop(["index"], axis=1)
    return df

def outlier_detection(df, feature):
    total = len(df)
    print("Before Data size: ", total, " Before skewness: ",df[feature].skew())
    y=df["total"]
    X=df[['count_vendor','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']]
    # X=df[[feature]]
    visualizer = CooksDistance()
    visualizer.fit(X, y)
    i_less_influential = (visualizer.distance_ <= visualizer.influence_threshold_)
    X_li, y_li = X[i_less_influential], y[i_less_influential]
    outliers=df[~df.index.isin(X_li.index)]
    df=df.drop(outliers.index)
    df=df.reset_index(drop=True)
    print("After Data size: ",len(df), " Total Outlies Removed: ", total-len(df), " After skewness: ",df[feature].skew())
    return df

def split_train_test(df):
    X_train,y_train=df[['count_vendor','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']].values,df[['total']].values
    return X_train, y_train

def model(X_train, y_train):
    model=linear_model.LinearRegression()
    model=model.fit(X_train, y_train)
    return model

def lin_reg_with_CI(df,indep_dep_var):
    # model trained on the train data set.
    model = smf.ols(indep_dep_var,df)
    results =model.fit()
    return results



def metrics(test_df, sc_y_test, y_test, yhat):
   
      # invert the transformation
    yhat = yhat.reshape(-1, 1)
    # print("before ++++++++++++++++++++++",yhat)
    yhat = sc_y_test.inverse_transform(yhat) 
    # print("after++++++++++++++++++++++",yhat)
      # Create new columns
    output = np.array(yhat)
    test_df['predicted'] = output
    test_df['predicted'] = test_df['predicted'].apply(np.ceil) 
    return test_df
 
 

def preprocessing(data,store):
    print(data)
    train_data=clean_data(data)
    train_data= outlier_detection(train_data, 'count_vendor')
    X_train, y_train=split_train_test(train_data)
    # result=model(X_train, y_tnrain)
    result=lin_reg_with_CI(train_data,'total ~ count_vendor+Monday+Tuesday+Wednesday+Thursday+Friday+Saturday')
    print("summary",result.summary())   
    filename = 'finalized_model.sav'
    pickle.dump(result, open(filename, 'wb'))
    return data

if __name__ == '__main__':
    time='24H_67'
    store='marsta_oob'
    make_directory('results/', store)
    output_path = "results/" + store + "/"
    data=pd.read_csv(r'bumbee/'+time+''+store+'.csv')
    data=data.drop(["Unnamed: 0"], axis=1)
    data =insert_day(data)
    data = preprocessing(data,store) 
   
   