# Initial libraries
import math 
import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from sklearn import tree
from tqdm.std import TRLock
from logging import Formatter
import matplotlib.pyplot as plt


# learning Algorithms
import xgboost as xg
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Scoring Metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Internal Libraries
from visualization import line_chart
from weather import merge_weather
from nn_merged_util import get_folders, feature_combinations, make_directory, concatenate_all_data,\
      insert_day_hour, normalization

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

def outlier_detection(df, feature):
      total = len(df)
      # print("Before Data size: ", total, " Before skewness: ",df[feature].skew())
      Q1 = df[feature].quantile(0.25)
      Q3 = df[feature].quantile(0.75)
      IQR = Q3 - Q1
      day_outlier_index = df[(df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR))].index
      df = df.drop(day_outlier_index)
      # print("After Data size: ",len(df), " Total Outlies Removed: ", total-len(df), " After skewness: ",df[feature].skew())
      return df

def xy_split(train, test, feature):
    '''
    Create split for train and text features
    '''
    X_train = train.drop(train.columns.difference(feature), axis=1).values
    y_train = train.drop(train.columns.difference(['total']), axis=1).values # train['total'].values
    X_test = test.drop(train.columns.difference(feature), axis=1).values
    y_test = test.drop(train.columns.difference(['total']), axis=1).values # test['total'].values
    return X_train, y_train, X_test, y_test

def create_train_test(df,frac=0.70, single_store=False):
    '''
    Needed for statsmodels
    '''
    train_idx = round(df.shape[0]*frac)
    test_idx = train_idx+1
    df = df.sort_values(by = 'time')
    #df.style.hide_index()

    #print(df[test_idx:][['time','total','count_randomised' ,'count_vendor',  'hour']])
    if single_store is False:
          test_df = df[test_idx:][['time', 'total', 'count_randomised' ,'count_vendor', 'hour']]
    else:
          test_df = df[test_idx:][['time', 'total', 'val_count_5', 'val_count_8','val_count_10', 'count_randomised' ,'count_vendor', 'hour']]
    #output = df[test_idx:][['time','total','count_randomised' ,'count_vendor',  'hour']]
    #output.to_excel("output.xlsx", index=False)
    return df[:train_idx] , df[test_idx:], test_df

def create_boxplot(df, feature, fig_name):
      df.boxplot(column=feature, return_type='axes')
      plt.savefig(fig_name, pad_inches=0.11, bbox_inches='tight', dpi=300)
      plt.close()

def create_pairplot(df, fig_name):
      g = sns.pairplot(data=df[['total','count_vendor', 'count_randomised', 'day']], diag_kind='kde')   
      # Map a density plot to the lower triangle
      g.map_lower(sns.kdeplot, cmap = 'Reds')
      plt.savefig(fig_name, pad_inches=0.11, bbox_inches='tight', dpi=300)
      plt.close()

def algo(X_train, y_train, X_test, model):
      # new_df['X_test'] = X_test[feature[0]]
      # new_df['y_test'] = np.array(y_test)

      regr = TransformedTargetRegressor(regressor=model, transformer=QuantileTransformer(output_distribution='normal'))
      # Fit X and y
      regr.fit(X_train, y_train)
      yhat = regr.predict(X_test)

      return yhat    

def z_score(intensity):
      mean_int = np.mean(intensity)
      std_int = np.std(intensity)
      z_scores = (intensity - mean_int) / std_int
      return z_scores

def outliers(data, intensity, feature):
    # 1. temporary dataframe
    df = data.copy(deep = True)

    # 2. Select a level for a Z-score to identify and remove outliers
    zscore = np.abs(stats.zscore(df[feature]))
    data['zscore'] = zscore

    # 3. Select rows with zscore less than intensity
    data = data[data['zscore']<intensity]

    return data

def preprocessing(data,store_name, date):
      # Insert Day column
      data = insert_day_hour(data, DropZeroFootfall=True, outlier_removal=True)

      # Pairplot to see the Pairplot graphs
      create_pairplot(data, 'results/'+date+"/"+"visualization/Before_normality (" + store_name +").png")

      # Select only those rows where count_randomised, count_vendor and total is greater than 0
      data = data[data['count_randomised']>0]
      data = data[data['count_vendor']>0]
      data = data[data['total'] > 0]       
      
      # print(data.isnull().sum().sort_values(ascending=False) / data.shape[0])
      create_pairplot(data, 'results/'+date+"/"+"visualization/After_normality (" + store_name +").png")
            
      create_boxplot(data, feature, 'results/'+date+"/"+"visualization/Before_boxplot_"+ str(feature) + "_(" + store_name + ").png")
      
      # Outliers detection

      # insert new columns for outlier removal
      data['outliers'] = data['total']/data['count_randomised']
      data['outliers'] = data['outliers'].abs()
      data = outlier_detection(data, 'count_randomised')
      data = outlier_detection(data, 'count_vendor')
      data = outlier_detection(data, 'total')
      data = outlier_detection(data, 'outliers')

      #Using the log1p function applies log(1+x) to all elements of the column
      
      # Log transformation for gaussian distribution
      # data["total"] = np.log1p(data["total"])
      # data["count_randomised"] = np.log1p(data["count_randomised"])
      # data["count_vendor"] = np.log1p(data["count_vendor"])

      # print("data: ",data) 
      create_boxplot(data, feature, 'results/'+date+"/"+"visualization/After_boxplot_"+ str(feature) + "_(" + store_name + ").png")

      # sort the data
      data = data.sort_values(by='time')

      
      # threshold = 3
      # print("total data before z_score: ", len(data))
      # data = outliers(data, threshold, feature='total')
      # data = outliers(data, threshold, feature='count_randomised')
      # print("total data after z_score: ", len(data))

      # Rolling function implementation
      # rolling_output = data[feature[0]].rolling(3, win_type ='gaussian').mean(std=data[feature[0]].std()) # win_type: 'gaussian'
      # data[feature[0]] = rolling_output

      # Convert total into categorical data
      # data['total'] = pd.cut(data['total'], 15, labels=np.arange(15))

      return data

def metrics(test_df, normal_type, sc_y_test, y_test, yhat, store, total_weeks):
      r2 = r2_score(y_test, yhat)
      mae = mean_absolute_error(y_test, yhat)
      mse = mean_squared_error(y_test, yhat)
      rmse = np.sqrt(mean_squared_error(y_test, yhat))   
      
      # invert the transformation
      if normal_type is not False:
            yhat = yhat.reshape(-1, 1)
            yhat = sc_y_test.inverse_transform(yhat) 
      
      # Create new columns
      output = np.array(yhat)
      test_df['predicted_cal'] = output
      test_df['predicted_cal'] = test_df['predicted_cal'].apply(np.ceil)
      test_df['residual'] = test_df['predicted_cal'] - test_df['total']
      test_df['residual'] = test_df['residual'].abs()
      test_df['ground_truth']= np.where((test_df['residual']<12), "TRUE", "FALSE")
      test_df['acc'] = (test_df['residual']/test_df['total'])
      
      #test_df = test_df[:100]

      #test_df.to_excel("results/23_10_2021/regression_pred_StandardScaler_MINMAX(60min).xlsx", index=False)
      fal_val= len(test_df[test_df['ground_truth']=='FALSE'])
      tru_val = len(test_df[test_df['ground_truth']=='TRUE'])      
      avg_acc = (tru_val/(tru_val+fal_val))
      avg_acc = "{:.1%}".format(avg_acc)
      avg_res = round(math.ceil(test_df['residual'].sum() / len(test_df['residual'])), 2)
      # avg_pred = test_df['predicted_cal'].sum()/len(test_df)

      # Calculate overall error percentage
      error_perc = abs(((test_df['total'].sum() - test_df['predicted_cal'].sum())/(test_df['total'].sum())))
      error_perc = "{:.1%}".format(error_perc)

      # Metrics to evaluate your model
      test_df['acc'] = test_df['acc'].map('{:.1%}'.format)
      # print("R2: ",r2, " MAE: ", mae, " MSE: ", mse)

      df2 = {'store_name': store,
            'actual_footfall': test_df['total'].sum(),
            'predicted_footfall': test_df['predicted_cal'].sum(),
            'error (%)': error_perc,
            'total_weeks': total_weeks,
            'MAE': mae,
            'MSE':mse,
            'RMSE':rmse,
            'r2':r2,
            'avg_residual': avg_res,
            'avg_acc' : avg_acc                
            }

      return test_df, df2

def correlation(df, store_name):
      correlations = df.corr()
      coor = sns.heatmap(correlations, annot=True, fmt='.2f', cmap='RdYlGn')
      plt = coor.get_figure()
      plt.savefig('results/'+date+"/"+"visualization/correlation/" + store_name + ".png", pad_inches=0.11, bbox_inches='tight', dpi=300)
      plt.clf()

def get_result_excel(feature_dict, filename):
    '''
    Save Summary Excel sheet in Summary folder in given summary path name
    '''
    writer = pd.ExcelWriter(filename, engine="xlsxwriter")
    workbook = writer.book
    
    for feature_name, store_dict in feature_dict.items():      
      row = 0 
      set_header = True
      for store_name, normalization_dict in store_dict.items():            
            for normal_type, model_dict in normalization_dict.items():                   
                  for model_name, metrics_dict in model_dict.items():
                        test_df = metrics_dict['test_df']
                        test_df = test_df.rename(columns={'total': 'camera_count'})
                        test_df['store_name'] = store_name
                        test_df['algorithm'] = model_name
                        test_df['normal_type'] = normal_type
                        
                        # set time column
                        test_df['time'] = pd.to_datetime(test_df['time'])
                        test_df['time'] = test_df['time'].dt.date

                        # print("Feature Namel: ", feature_name, " Store: ", store_name, \
                        #       " normal_type: ", normal_type, " model: ", model_name)
                        # print(test_df[['total', 'predicted_cal']][:2])

                        if set_header is True:
                              test_df.to_excel(writer, startrow=row, index=False, sheet_name=feature_name)
                        else: 
                              test_df.to_excel(writer, startrow=row, index=False, sheet_name=feature_name, header=False)

                        # heading_name = feature_name + "_" + store_name + "_" + normal_type + "_" + model_name
                        
                        # Create a format to use in the merged range.
                        # merge_format = workbook.add_format({
                        # 'bold': 1,
                        # 'border': 1,
                        # 'align': 'center',
                        # 'valign': 'vcenter',
                        # 'fg_color': 'yellow'})
                        
                        worksheet = writer.sheets[feature_name]
                        # worksheet.merge_range(row+2, 0, row+2, len(test_df.columns)-1, heading_name, merge_format)
                        if set_header is True:
                              row += (len(test_df) + 1)
                        else:
                              row += len(test_df)
                        set_header = False
    writer.save()

def new_result_excel(feature_dict, filename):
    '''
    Save Summary Excel sheet in Summary folder in given summary path name
    '''
    writer = pd.ExcelWriter(filename, engine="xlsxwriter")
    workbook = writer.book    

    for feature_name, feat_dict in feature_dict.items():      
      row = 0 
      #print(feat_dict)
      for overall_key, df in feat_dict.items():  
             
            # print("overall_key: ", overall_key)       
            # test_df = feat_dict[overall_key]
            # print(df[['total', 'predicted_cal']][:2])

            # Write to sheet
            df.to_excel(writer, startrow=row + 3, index=False, sheet_name=feature_name)
            
            # Create a format to use in the merged range.
            merge_format = workbook.add_format({
            'bold': 1,
            'border': 1,
            'align': 'center',
            'valign': 'vcenter',
            'fg_color': 'yellow'})
            
            worksheet = writer.sheets[feature_name]
            worksheet.merge_range(row+2, 0, row+2, len(df.columns)-1, overall_key, merge_format)
            row += (len(df) + 4)
    writer.save()

def get_summary_excel(feature_dict, filename):
    '''
    Save Summary Excel sheet in Summary folder in given summary path name
    '''
    writer = pd.ExcelWriter(filename, engine="xlsxwriter")
    workbook = writer.book
    FLAG = True
    
    column_list = []
    for feature_name, store_dict in feature_dict.items():
      row = 0    
      for store_name, normalization_dict in store_dict.items():                         
            for normal_type, model_dict in normalization_dict.items():
                  df = pd.DataFrame(columns=column_list)                   
                  for model_name, metrics_dict in model_dict.items():
                        removed_test_df = metrics_dict.pop('test_df')
                        # test_df = metrics_dict['test_df']
                        # print("Model: ", model_name, " ,dict: ",metrics_dict)
                        if FLAG is True:
                              column_list = metrics_dict.keys()
                              FLAG = False
                              

                        # Write to sheet
                        df = df.append(metrics_dict, ignore_index=True)
                  df.to_excel(writer, startrow=row + 3, index=False, sheet_name=feature_name)
                  heading_name = store_name + "_" + feature_name + "_" + normal_type
                        
                  # Create a format to use in the merged range.
                  merge_format = workbook.add_format({
                  'bold': 1,
                  'border': 1,
                  'align': 'center',
                  'valign': 'vcenter',
                  'fg_color': 'yellow'})
                  
                  worksheet = writer.sheets[feature_name]
                  worksheet.merge_range(row+2, 0, row+2, len(column_list)-1, heading_name, merge_format)
                  row += (len(df) + 4)
    writer.save()

if __name__ == '__main__':
      start_time = time.time()
      # Set results output path
      date = '10_12_2021 (without_outliers) (60min & -67)'
      make_directory('results/', date)
      output_path = "results/" + date + "/"
      make_directory('results/'+date+"/", "visualization")
      make_directory('results/'+date+"/"+"visualization/","correlation")
      # Control learning parameters
      base_path = 'data'
      time = '60min_67' # '60min' '30min' '24H' actual_folder -> '24H_67'
      all_feat = ['temperature','count_randomised', 'hour', 'count_vendor'] # Take weather as a feature , 'count_vendor', 'hour', 'day'
      # store = 'inferno_odenplan' # inferno_taby inferno_odenplan
      #file_path = base_path + "/" + store + "/" + store + "_" + time
      normal_type_list = ['RobustScaler', 'MinMaxScaler', 'StandardScaler']  # , 'RobustScaler', 'MinMaxScaler', 'StandardScaler'
      max_depth = 2
      single_store = False # control variable for validating inferno store data only
      algo_dict = {'LinearRegression':LinearRegression(),
                  'DecisionTreeRegressor': DecisionTreeRegressor(),
                  'RandomForestRegressor': RandomForestRegressor(),
                  'XGBRFRegressor': xg.XGBRFRegressor(max_depth=max_depth),
                  'GradientBoostingRegressor':GradientBoostingRegressor(max_depth=max_depth)}
      
      algo_list = algo_dict.keys()
      
      # Get list of all folders
      store_folders = get_folders(base_path)     
      # Get all Feature Combinations      
      all_feat = feature_combinations(all_feat)
      all_feat = [
            ['count_randomised'],
            ['count_vendor'],            
            # ['avg_temp','count_randomised'],            
            ['count_vendor', 'day'],
            ['count_randomised', 'day'],
            ['count_randomised', 'count_vendor'],
            ['count_randomised', 'count_vendor', 'day'],
            ['count_randomised', 'count_vendor', 'hour', 'day']
            # ['avg_temp','count_randomised', 'count_vendor'],
            # ['min_temp', 'max_temp','count_randomised', 'count_vendor']
      ]
      # Features for Training and Testing
      # good accuracy using ['count_randomised', 'count_vendor', 'hour'] at 30min data
      # feature = ['count_randomised', 'count_vendor', 'hour', 'day'] # also getting accuracy using this
      # feature = ['count_randomised', 'count_vendor', 'hour']  # best result using this feature        
      feature_dict = {}
      overall_feat_dict = {}

      for feature in tqdm(all_feat): # [:2]
            # print("Feature: ", feature)       
            
            # concatenate feature list to name
            concat_feature = [s.replace("_","") for s in feature]
            concat_feature = '_'.join([str(elem) for elem in concat_feature])
            concat_feature = concat_feature.replace('count','C')
            concat_feature = concat_feature.replace('ised','')
            concat_feature = concat_feature.replace('avgtemp','AvgT') #'min_temp', 'max_temp'
            concat_feature = concat_feature.replace('mintemp','MinT')
            concat_feature = concat_feature.replace('maxtemp','MaxT')

            store_dict = {}
            overall_dict = {}  
            
            store_folders = ['marsta_oob']
            if single_store is True:
                  store_folders = ['inferno_odenplan']      
            
            for store in store_folders: # [:2]
                  print("Store Name: ", store)

                  file_path = base_path + "/" + store + "/" + store + "_" + time
                  # Get list of all folders in file_path
                  date_folders = get_folders(file_path)
                  total_weeks = len(date_folders)  
                  # Concatenate all data          
                  data = concatenate_all_data(file_path, date_folders)
                  # data = merge_weather(store, data)
                  correlation(data, store)
                  
                  # Preprocess the data
                  data = preprocessing(data, store, date) 
                  
                  # integrate validation count
                  if single_store is True:
                        val_df = pd.read_csv("val_data/val_data.csv")
                        data['start_time'] = data['time'].astype('str')
                        data = pd.merge(data,val_df,left_on="start_time",right_on="start_time", how='inner').drop(['start_time','visit_id'],axis=1)
                        line_chart(data[['time', 'total', 'count_randomised', 'val_count_5', 'val_count_8','val_count_10']], 'time', 'total', \
                        None, ['val_count_5', 'val_count_8','val_count_10'], figurename=output_path+"camera_val_count.png")
                  
                  #exit()

                  # create new variable names as total_o containing main total value
                  # inflate total 
                  # data['total_o'] = data['total']
                  # data['total'] = data['total'] + (.30*data['total'])
                  # data['total'] = data['total'].astype(int)
                  # print(data[['total','total_o']])
                  # exit()
                  total_rows = len(data)
                  # print(data)
                  
                  # create train and testing groups along with test dataframe
                  train, test, test_df = create_train_test(data)
                  # print("Store: ", store, " test_size: ", len(test))

                  # Create train test                  
                  X_train, y_train, X_test, y_test = xy_split(train, test, feature)
                  # breakpoint()

                  normalization_dict = {}
                  
                  for normal_type in normal_type_list: # 
                        # print("Normalization Type: ", normal_type)
                        # Perform Normalization and Standardization
                        if normal_type is not False:
                              #print("normal_type is not false")
                              X_train_n, y_train_n, X_test_n, y_test_n, sc_X_train, sc_y_train, sc_X_test, \
                                    sc_y_test = normalization(X_train, y_train, X_test, y_test, normal_type)                        
                        
                        model_dict = {}
                        for model_name in algo_list:
                              print("Processing ", store, " using ",  model_name, " model and " , normal_type, " normalization and ", concat_feature)

                              model = algo_dict[model_name]
                              yhat = algo(X_train_n, y_train_n, X_test_n, model)

                              # Set directory path
                              make_directory(output_path, model_name)
                              make_directory(output_path + model_name + "/", concat_feature)
                              make_directory(output_path + model_name + "/" + concat_feature + "/", store)
                              
                              fig_name = "/home/gurjot/bumbeelabs/" + output_path + model_name + "/" + concat_feature + "/" + store + "/" + time + "_" + normal_type + ".png"
                              hyper_link = model_name + "/" + concat_feature + "/" + store + "/" + time + "_" + normal_type + ".png"
                              df, metrics_dict = metrics(test_df, normal_type, sc_y_test, y_test_n, yhat, store, total_weeks)
                              
                              # Plot the residuals after fitting a linear model
                              make_directory('results/'+date+"/"+"visualization/","residual_plot")
                              sns.residplot(x=df['total'], y=df['predicted_cal'], lowess=True, color="g")
                              residual_file_name = "/home/gurjot/bumbeelabs/" + output_path + "visualization" + "/" + "residual_plot/" + store + "_" + concat_feature + "_" + normal_type + "_" + model_name + " (residual).png"
                              plt.savefig(residual_file_name, pad_inches=0.11, bbox_inches='tight', dpi=300)
                              plt.close()

                              # print("output_path: ",output_path, " Concate Feat: ", concat_feature)
                              # print("Figure Name: ", fig_name)
                              # metrics_dict['image_hyperlink'] = '=HYPERLINK("%s", "%s")'.format(fig_name, store)
                              
                              metrics_dict['image_hyperlink'] = '=HYPERLINK("{}", "{}")'.format(hyper_link, store)
                              metrics_dict['total_rows'] = total_rows
                              #print("metrics_dict['image_hyperlink']: ", metrics_dict['image_hyperlink'])
                              metrics_dict['test_df'] = df.copy()
                              
                              metrics_dict['model_name'] = model_name
                              
                              # Add in metrics model_dict dictionary
                              model_dict[model_name] = metrics_dict                              
                              
                              # NEW DICTIONARY KEY
                              overall_dict_key = store + "_" + concat_feature + "_" + normal_type + "_" + model_name
                              # print("overall_dict_key: ", overall_dict_key)
                              if overall_dict_key in overall_dict:
                                    print("KEY ALREADY EXIST")
                                    exit()
                              overall_dict[overall_dict_key] = df.copy()
                              # print(df[['total', 'predicted_cal']])
                              # print(overall_dict[overall_dict_key][['total', 'predicted_cal']][:2])
                              # print(test_df)                           
                              
                              if single_store is True: 
                                    line_chart(df, 'time', 'total', 'predicted_cal', ['val_count_5', 'val_count_8','val_count_10'], figurename=fig_name)
                              else:
                                    line_chart(df, 'time', 'total', 'predicted_cal', None, figurename=fig_name)

                              # print("model_dict: ", model_dict)
                        normalization_dict[normal_type] = model_dict
                  # breakpoint()
                        # print("\n\n")
                  store_dict[store] = normalization_dict
            feature_dict[concat_feature] = store_dict
            # print("concat_feature: ", concat_feature)
            overall_feat_dict[concat_feature] = overall_dict
            
            

      file_name = output_path + "Results.xlsx"
      get_result_excel(feature_dict, file_name)
      # new_result_excel(overall_feat_dict, file_name)
      file_name = output_path + "Summary.xlsx"
      get_summary_excel(feature_dict, file_name)
      
      # print("Total Time Taken: ", (time.strptime(time.time(),"%H:%M:%S") - time.strptime(start_time,"%H:%M:%S")))