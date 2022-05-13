import pandas as pd
import numpy as np
import pickle

def clean_data(df):
    df['time'] = pd.to_datetime(df['time'])
    df=df[(df["total"]!=0)&(df["count_vendor"]!=0)].reset_index().fillna(0)
    df=df.drop(["index"], axis=1)
    return df

def test_set(df):
    y_test=df[['total']].values
    return y_test

def model_test(y_test,data):
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    alpha = .05
    test_predictions = loaded_model.get_prediction(data).summary_frame(alpha)
    test_predictions.reset_index(inplace = True, drop = True)
    data['predicted'] =test_predictions['mean'].apply(np.ceil)
    data['upper_limit']=test_predictions['obs_ci_upper'].apply(np.ceil)
    data['lower_limit']=test_predictions['obs_ci_lower'].apply(np.ceil)
    return data
    
def preprocessing(testing_data):
    test_data=clean_data(testing_data)
    y_test=test_set(test_data)
    data=model_test(y_test,test_data)
    data=data.drop(["perc_randomised","perc_vendor","apple","not_apple","count_randomised","count_vendor"], axis=1)
    data.rename({'total': 'Camera_count'}, axis=1, inplace=True)
    data.to_csv('Predicted_results_'+store+".csv", index=False)

if __name__ == '__main__':
    store='inferno_odenplan' 
    testing_file='test.csv'
    testing_data=pd.read_csv(testing_file)
    testing_data=testing_data.drop(["Unnamed: 0"], axis=1)
    data = preprocessing(testing_data) 
   
    
   