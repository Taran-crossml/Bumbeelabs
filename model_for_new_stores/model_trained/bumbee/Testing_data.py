import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

def make_directory(path, folder_name):
    make_path = path + folder_name
    try:
        os.mkdir(make_path)
    except Exception as e:
        pass

def clean_data(df):
    df['time'] = pd.to_datetime(df['time'])
    df=df[(df["count_vendor"]!=0)&(df["count_randomised"]!=0)].reset_index().fillna(0)
    df=df.drop(["index"], axis=1)
    return df

def insert_day(data):
    data=data[(data["count_vendor"]!=0)&(data["count_randomised"]!=0)].reset_index().fillna(0)
    data['time'] = pd.to_datetime(data['time'])
    data['day'] = data['time'].dt.day_name()
    # data.drop(['perc_randomised', 'perc_vendor', 'count_vendor', 'apple','not_apple'], inplace= True, axis=1)
    data = pd.get_dummies(data)
    data.rename(columns = {'day_Friday':'Friday', 'day_Monday':'Monday', 'day_Tuesday':'Tuesday', 'day_Wednesday':'Wednesday',
                        'day_Thursday':'Thursday','day_Saturday':'Saturday','day_Sunday':'Sunday'}, inplace = True)
    return data


def test_set(df1):
    X_test=df1[['count_vendor']].values
    return X_test

def model_test(X_test,data):
    # print(data)
    # print(X_test)
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    alpha = .05
    test_predictions = loaded_model.get_prediction(data).summary_frame(alpha)
    test_predictions.reset_index(inplace = True, drop = True)
    # print(test_predictions)
    data['Camera_count_prediction'] =test_predictions['mean'].apply(np.ceil)
    data['upper_limit']=test_predictions['obs_ci_upper'].apply(np.ceil)
    data['lower_limit']=test_predictions['obs_ci_lower'].apply(np.ceil)
    data.drop(["level_0","apple","not_apple","perc_randomised","perc_vendor",'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], inplace= True, axis=1)
    # print(data)
    return data

def line_plot(data):
    plt.figure(figsize=(25,7))
    plt.plot(data['time'],data['Camera_count_prediction'],color='blue',linewidth = '0.9',label='Camera_count_prediction')
    plt.plot(data['time'],data['upper_limit'],color='red',linewidth = '0.9',label='upper bound',linestyle = 'dashed')
    plt.plot(data['time'],data['lower_limit'],color='green',linewidth = '0.9',label='lower bound',linestyle = 'dashed')
    plt.fill_between(data['time'],data['upper_limit'], data['lower_limit'], color='#F2F2F2')
    plt.xlabel('Date')
    plt.ylabel('Footfall')
    plt.xticks(rotation=90)
    plt.legend(loc="upper right",prop={'size': 10})
    plt.title('Predictions_of_'+store+'')
    plt.savefig('results/'+store+'/''visualization_'+store+'', pad_inches=0.11, bbox_inches='tight', dpi=300)
    plt.close() 

def preprocessing(testing_data):
    test_data=clean_data(testing_data)
    X_test=test_set(test_data)
    data=model_test(X_test,test_data)
    data=data.drop(["count_randomised","count_vendor"], axis=1)
    data.to_csv('results/'+store+'/''Predicted_results_'+store+".csv", index=False)
    line_plot(data)


if __name__ == '__main__':
    #   oob_arninge   oob_hornsgatan    oob_kungsholmen
    store='oob_kungsholmen' 
    testing_file='oob_kungsholmen_220131-220206_preprocessed.csv'
    make_directory('results/', store)
    testing_data=pd.read_csv(r'bumbee/testing_dataset''/'+testing_file+'')
    testing_data = testing_data.rename(columns = {'node_time': 'time'}, inplace = False)
    # testing_data=testing_data.drop(["Unnamed: 0"], axis=1)
    data =insert_day(testing_data)
    data = preprocessing(data) 
   
   
    
   