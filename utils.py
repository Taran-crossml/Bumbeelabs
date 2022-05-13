import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import json
import statsmodels.formula.api as smf
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

def is_dir(d):
    '''
    Checks if directory is present or not
    '''
    if os.path.isdir(d) and d not in ['.ipynb_checkpoints','.vscode','__pycache__']:
        return True
    else:
        return False

def get_health_stats(df, assumptions,dir_name):
    '''
    Generates sanity report for a given df

    Parameters:
    df: dataframe
    assumpotions: python dict
    dir_name: name of the company dir
    '''
    df['time'] = pd.to_datetime(df['time'])
    nobs = df.shape[0]
    q1 = (df['time'].dt.hour>assumptions['working_hours']['start'])&(df['time'].dt.hour<=assumptions['working_hours']['end'])
    working_hours = df[q1].shape[0]
    wifi_missing = working_hours - df[q1].dropna().shape[0]
    useful_obs = (nobs-wifi_missing)
    perc_total = round(useful_obs/nobs,2)*100
    count_zero = len(df[df['total'] == 0])
    return [dir_name,nobs,working_hours,wifi_missing,useful_obs,perc_total, count_zero]

def generate_stats(base_path,data_folders,assumptions):
    '''
    Generates report for a given base_directory containing company sub directories

    Parameters:
    base_path: main path for stores sub_dirs, usually arranged weekwise
    data_folders: list of company dir names in base_path
    assumptions: python dict, eg assumptions = {'working_hours':{'start':7,'end':23}}
    '''
    stats = []
    for folder in tqdm(data_folders):
        path = os.path.join(base_path,folder,f'{folder}_preprocessed.csv')
        #print(path)
        df = pd.read_csv(path)
        stats.append(get_health_stats(df,assumptions,folder))
    cols = ['name','nobs','working_hours','wifi_missing','useful_obs','perc_total','count_zero']
    result =pd.DataFrame(stats,columns=cols)
    return result  


def clean_aggregate_data(base_path,folder,assumptions,part='full',percentile=0.50):
    '''
    Produces a shuffled df for a given store dir

    Parameters:
    base_path: Main dir with company wise sub directories
    folder: Company dir
    assumptions: Python dict, eg assumptions = {'working_hours':{'start':7,'end':23}}
    part: full, uppper (top 50 percentile), lower (bottom 50 percentile)
    '''
    if part not in ['full','upper','lower']:
        raise "part should be either full, upper or lower"
    path = os.path.join(base_path,folder,f'{folder}_preprocessed.csv')
    print("File Path: ", path)
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    q1 = (df['time'].dt.hour>assumptions['working_hours']['start'])&(df['time'].dt.hour<=assumptions['working_hours']['end'])
    if part == 'full':
        res = df[q1].dropna().sample(frac=1,random_state=42).reset_index(drop=True) 
    if part == 'upper':
        q2 = df['total']>df['total'].quantile(percentile)
        res = df[q1][q2].dropna().sample(frac=1,random_state=42).reset_index(drop=True)
    if part == "lower":
        q2 = df['total']<=df['total'].quantile(percentile)
        res = df[q1][q2].dropna().sample(frac=1,random_state=42).reset_index(drop=True)
    return res


def create_train_test(df,frac=0.70):
    '''
    Needed for statsmodels
    '''
    train_idx = round(df.shape[0]*frac)
    test_idx = train_idx+1
    df = df.sort_values(by = 'time')
    #df.style.hide_index()

    #print(df[test_idx:][['time','total','count_randomised' ,'count_vendor',  'hour']])
    test_df = df[test_idx:][['time','total','count_randomised' ,'count_vendor', 'hour']]
    #output = df[test_idx:][['time','total','count_randomised' ,'count_vendor',  'hour']]
    #output.to_excel("output.xlsx", index=False)
    return df[:train_idx] , df[test_idx:], test_df

def x_y_split(train,test):
    '''
    Needed for Sklearn models
    '''
    #print("TRAIN: \n", train)
    #print("train.drop(['time','total'],axis=1).values: \n", train.drop(['time','total'],axis=1).values)
    df_new = train.drop(['time', 'total'], axis=1)
    #print("df_new: \n", df_new)
    #print("TEST: \n", test.values)
    X_train,X_test,y_train,y_test = train.drop(['time','total','apple','not_apple'],axis=1).values,test.drop(['time','total','apple','not_apple'],axis=1).values,train['total'].values,test['total'].values
    return X_train,X_test,y_train,y_test





def run_model_pipeline(training_config, minute, file_name, test_df):
    '''
    Uses training config to run the complete model pipeline and produce model iterations

    Parameters:
    training_config: python dict, eg training_config = {
   "tree_regressor":{
      "data":{
         "train":[
            X_train,
            y_train
         ],
         "test":[
            X_test,
            y_test
         ]
      },
      "parameters":{
         "max_depth":4,
         "criterion":"mse"
      }
   },
   "linear_regressor":{
      "data":{
         "train":train,
         "test":test
      },
      "variations":[
         "total~perc_randomised",
         "total~perc_randomised+hour",
         "total~hour"
      ]
   }

    '''
    x, y = training_config['tree_regressor']['data']['train']
    x_t, y_t = training_config['tree_regressor']['data']['test']
    train = training_config['linear_regressor']['data']['train']
    test = training_config['linear_regressor']['data']['test']
    
    params = training_config['tree_regressor']['parameters']
    params_string = json.dumps(params)
    reg = DecisionTreeRegressor(**params)
    #reg = SVR(kernel='linear') # just using support vector machine regression for experiment
    reg.fit(x,y)
    tree_preds = reg.predict(x_t)
    errors = np.abs(tree_preds-y_t)
    mae = errors.mean()
    tree_result = pd.Series(errors).quantile(np.arange(0,1,0.1))
    report = {"tree_"+params_string:tree_result}
    report['tree_mae'] = mae

    for variation in training_config['linear_regressor']['variations']:
        variation_name = variation.replace("~","_").replace('+', '_')
        print("variation_name: ", variation_name)
        results = smf.ols(variation,data=train).fit()
        reg_preds = results.predict(test)
        errors = np.abs(reg_preds-y_t)
        all_results = results.predict(test)
        p_values = []
        for data in list(zip(y_t, reg_preds)):
            print(round(data[1], 2))
            p_values.append(round(data[1], 2))
        test_df[variation_name] = p_values
        print('*'*100)
        
        mae = errors.mean()

        reg_result = pd.Series(errors).quantile(np.arange(0,1,0.1))
        report['reg_'+variation] = reg_result
        report['reg_mae '+variation] = mae
    report['train_size'] = x.shape[0]
    report['test_size'] = x_t.shape[0]
    report_df = pd.DataFrame(report)
    test_df.to_excel(minute + "/" + file_name + ".xlsx", index=False)
    #report_df.to_excel(minute + "/" + "Tree_" + file_name + ".xlsx")
    #test_df.to_excel(minute+"/"+file_name+".xlsx", index=False)
    return report_df, test_df

def merge_store_data(folders,sub_folders,dest_folder):
    mapping = {}
    curr_dir = os.getcwd()
    for folder in folders:
        mapping[folder]=os.listdir(folder)
    for sub_folder in tqdm(sub_folders):
        flag = True
        for folder in folders:
            if sub_folder not in mapping[folder]:
                flag = False
        if flag:
            dfs = [pd.read_csv(os.path.join(folder,sub_folder,f'{sub_folder}_preprocessed.csv')) for folder in folders]
            consolidated = pd.concat([df for df in dfs], axis=0)
            if not os.path.isdir(os.path.join(dest_folder,sub_folder)):
                os.mkdir(os.path.join(dest_folder,sub_folder))
            consolidated.to_csv(os.path.join(dest_folder,sub_folder,f'{sub_folder}_preprocessed.csv'),index=False)



