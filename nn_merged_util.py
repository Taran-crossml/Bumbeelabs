import os
import re
import math
import numpy as np
import pandas as pd
from tqdm import tqdm 
from keras.layers import Dense
import matplotlib.pyplot as plt
from weather import merge_weather
from itertools import combinations
from keras.models import Sequential
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.python.keras.optimizer_v2.adam import Adam # standard Adam optimizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def make_directory(path, folder_name):
    '''
    Create New Directory in given path

    :param path: path in which new folder is to be created
    :param folder_name: Folder to be created
    :return: Created new folder in given path
    '''
    make_path = path + folder_name
    try:
        # Make Directory with Date
        os.mkdir(make_path)
    except Exception as e:
        # print(e, " arises")
        # print(folder_name, " already exists")
        pass

def is_dir(d):
    '''
    Checks if directory is present or not
    '''
    if os.path.isdir(d) and d not in ['.ipynb_checkpoints','.vscode','__pycache__']:
        return True
    else:
        return False

def get_folders(base_path):
    '''
    get all the folders present in provided base_path 
    '''
    data_folder = []
    for folder in os.listdir(base_path):
        if is_dir(os.path.join(base_path, folder)):
            data_folder.append(folder)
    return data_folder

def insert_day_hour(data, DropZeroFootfall=True, outlier_removal=True):
    '''
    Insert day column in dataframe and select if user want to
    drop the rows where total (footfall) is less than 0. By default rows will be dropped.

    :param data: dataframe
    :param DropZeroFootfall: True: Drop rows with footfall less than 0 and False: Do-not drop rows
    :return: return pre-processed data
    '''
    # remove rows with total footfall less than 0
    # and insert day column and convert into features
    if DropZeroFootfall is True:
        # Select rows with footfall greater than 0
        data = data[data['total'] > 0]
    data['hour'] = pd.to_datetime(data['time'])
    data['hour'] = data['hour'].dt.hour
    data['day'] = data['time'].astype('datetime64[D]')
    data['day'] = data['day'].dt.day_name()
    data['day'].replace(to_replace=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'],
                        value=[1, 2, 3, 4, 5, 6, 7], inplace=True)    
    if outlier_removal is True:
        data = outlier_detection(data)
    return data

def concatenate_all_data(file_path, date_folders):
    '''
    Return concatenated preprocessed data into single dataframe
    '''
    dfs = [pd.read_csv(os.path.join(file_path, folder, f'{folder}_preprocessed.csv')) for folder in date_folders]
    data = pd.concat([df for df in dfs], axis=0)
    data.sort_values(by='time')
    return data

def outlier_saving_dropping(df, day_outlier_index, day, save=False):
    '''
    Save the outliers in file and drop the outliers index 

    return the clean dataframe without outliers 
    '''
    if len(day_outlier_index) >= 1:
        if save is True:
            if not os.path.isdir("results/outliers"):
                #  Make Directory with Date\
                os.mkdir("results/outliers")
            file_name = "results/outliers/" + day + ".xlsx"
            writer = pd.ExcelWriter(file_name, engine="xlsxwriter")
            row = 0
            day_df = df.loc[day_outlier_index]
            text = day + 'Outliers'
            day_df.to_excel(writer, startrow=row + 3)
            worksheet = writer.sheets['Sheet1']
            worksheet.write(row + 2, 0, text)
            row += (len(day_df) + 4)
            writer.save()
        df = df.drop(day_outlier_index)
        return df
    else:
        return df

def OutlierRemoval(df, feature='count_randomised'):
    total = len(df)
    print("Before Data size: ", total, " Before skewness: ",df[feature].skew())
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    day_outlier_index = df[(df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR))].index
    df = df.drop(day_outlier_index)
    print("After Data size: ",len(df), " Total Outlies Removed: ", total-len(df), " After skewness: ",df[feature].skew())
    return df

def outlier_detection(df, outlier_variable='count_randomised'):
    '''
    Function to detect outliers in the dataframe
    '''
    # print("Full DF: ", len(df))
    df = df.reset_index(drop=True)
    # print("Rest DF: ", len(df))
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in days:
        day_df = df[df['day'] == day]
        Q1 = day_df[outlier_variable].quantile(0.25)
        Q3 = day_df[outlier_variable].quantile(0.75)
        IQR = Q3 - Q1
        # next step will help to convert outliers into true values
        day_outlier_index = day_df[(day_df[outlier_variable] < (Q1 - 1.5 * IQR)) | (day_df[outlier_variable] > (Q3 + 1.5 * IQR))].index
        #print("Outliers index: ", day_outlier_index)
        df = outlier_saving_dropping(df, day_outlier_index, day)
    return df

def create_train_test(df,frac=0.70):
    '''
    Needed for statsmodels
    '''
    train_idx = round(df.shape[0]*frac)
    test_idx = train_idx+1
    df = df.sort_values(by = 'time')
    #df.style.hide_index()

    #print(df[test_idx:][['time','total','count_randomised' ,'count_vendor',  'hour']])
    test_df = df[test_idx:][['time', 'total', 'count_randomised' ,'count_vendor', 'hour']]
    #output = df[test_idx:][['time','total','count_randomised' ,'count_vendor',  'hour']]
    #output.to_excel("output.xlsx", index=False)
    return df[:train_idx] , df[test_idx:], test_df

def xy_split(train, test, feature):
    '''
    Create split for train and text features
    '''
    X_train = train.drop(train.columns.difference(feature), axis=1).values
    y_train = train.drop(train.columns.difference(['total']), axis=1).values # train['total'].values
    X_test = test.drop(train.columns.difference(feature), axis=1).values
    y_test = test.drop(train.columns.difference(['total']), axis=1).values # test['total'].values
    return X_train, y_train, X_test, y_test

def normalization(X_train, y_train, X_test, y_test, normal_type):
    '''
    Perform standardization/normalization using different feature scaling methods 
    '''
    if (normal_type == 'StandardScaler'): 
        print("we are in StandardScaler")
        sc_X_train = StandardScaler()
        X_train = sc_X_train.fit_transform(X_train)

        sc_y_train = StandardScaler()
        y_train = sc_y_train.fit_transform(y_train)

        sc_X_test = StandardScaler()
        X_test = sc_X_test.fit_transform(X_test)

        sc_y_test = StandardScaler()
        y_test = sc_y_test.fit_transform(y_test)
    if (normal_type == 'RobustScaler'): 
        print("we are in RobustScaler")
        sc_X_train = RobustScaler()
        X_train = sc_X_train.fit_transform(X_train)

        sc_y_train = RobustScaler()
        y_train = sc_y_train.fit_transform(y_train)

        sc_X_test = RobustScaler()
        X_test = sc_X_test.fit_transform(X_test)

        sc_y_test = RobustScaler()
        y_test = sc_y_test.fit_transform(y_test)
    if (normal_type == 'MinMaxScaler'): 
        print("we are in MinMaxScaler")
        sc_X_train = MinMaxScaler()
        X_train = sc_X_train.fit_transform(X_train)

        sc_y_train = MinMaxScaler()
        y_train = sc_y_train.fit_transform(y_train)

        sc_X_test = MinMaxScaler()
        X_test = sc_X_test.fit_transform(X_test)

        sc_y_test = MinMaxScaler()
        y_test = sc_y_test.fit_transform(y_test)
    
    return X_train, y_train, X_test, y_test, sc_X_train, sc_y_train, sc_X_test, sc_y_test

def Kfold_baseline_model(X_train, L1_neurons, L2_neurons, learning_rate=0.001):
    '''
    Kfold baseline method
    '''
    def bm():
        model = Sequential()
        model.add(Dense(L1_neurons, input_dim=X_train.shape[1], activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(L2_neurons, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(1, activation='linear'))
        opt = Adam(learning_rate=learning_rate)
        model.compile(loss='mse', optimizer='adam')
        return model
    return bm

def kfold(X_train, y_train, L1_neurons, L2_neurons, batch_size, epoch, learning_rate):
    '''
    Keras regression implemention
    return trained model and calculated returns
    '''
    model = KerasRegressor(build_fn=Kfold_baseline_model(X_train, L1_neurons, L2_neurons, learning_rate), nb_epoch=epoch, batch_size=batch_size, verbose=False)
    kfold = KFold(n_splits=10)
    results = cross_val_score(model, X_train, y_train, cv=kfold)
    return model, results

def line_plt(test_df, col_name, error, feature_list, fig_name, time, L1_neurons, L2_neurons, epoch, normal_type, learning_rate, batch_size):
    '''
    Plot line chart for various features and test data
    '''
    test_df['time'] = pd.to_datetime(test_df['time'])

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))

    ax[0].plot(test_df["time"], test_df.total, linewidth=1, color='#e9ecef')
    ax[0].plot(test_df["time"], test_df[col_name], linewidth=1, color='#0a9396')

    # Set tick parameters and Make y axis tics not visible
    ax[0].tick_params(axis='x', direction='in', length=3, color="black", labelsize=8)
    ax[0].tick_params(axis='y', direction='in', length=3, color="black", labelsize=8)

    fonts = {"fontname": "Times New Roman", "fontsize": 8}
    ax[0].set_xlabel('', fontdict=fonts)
    ax[0].set_ylabel('', fontdict=fonts)

    ax[0].text(0.88, 0.88, str('MAE: %.3f' % error), horizontalalignment='center',
            verticalalignment='center',
            transform=ax[0].transAxes)
    
    ax[0].legend(['Camera Count', 'Predicted Count'],
                bbox_to_anchor=(0.0, 1.00, 1, 0.3),
                loc='center',
                ncol=3,
                frameon=False)
    
    ax[1].axis('off')

    # Insert Experiment Hyperparamter's
    fig.text(0.15, 0.40, 'Feature: ' + str(feature_list))
    fig.text(0.15, 0.37, 'Minutes: ' + str(time))
    fig.text(0.15, 0.34, 'L2_neurons: ' + str(L2_neurons))
    fig.text(0.15, 0.31, 'L1_neurons: ' + str(L1_neurons))
    if normal_type is not False:
        fig.text(0.15, 0.28, 'Normalization: ' + normal_type) # last y = 0.5
    else:
        fig.text(0.15, 0.28, 'Normalization: ' + "False")
    fig.text(0.15, 0.25, 'Epochs: ' + str(epoch))
    fig.text(0.15, 0.22, 'Learning Rate: ' + str(learning_rate))
    fig.text(0.15, 0.19, 'Batch Size: ' + str(batch_size))
    plt.savefig(fig_name, pad_inches=0.11, bbox_inches='tight', dpi=600)

def feature_combinations(feat):
    '''
    Feature combinations are generated

    :param feat: List of columns
    :return: Combinations of features
    '''
    # List of all combinations of Features
    all_feat = sum([list(map(list, combinations(feat, i))) for i in range(len(feat) + 1)], [])
    # Remove empty list from all_feat
    all_feat.remove([])
    return all_feat

def get_single_excel(all_test_df, file_name):
    '''
    Create single Excel File with various worksheets containing predictions results of various feature combinations 
    '''
    writer = pd.ExcelWriter(file_name, engine="xlsxwriter")
    workbook = writer.book    
    for feature_name, feature_dict in all_test_df.items():
        row = 0
        for store_name, store_dict in feature_dict.items():
            store_df = store_dict['test_df']
            error = store_dict['MAE']
            total_weeks = store_dict['total_weeks']

            store_df.to_excel(writer, startrow=row + 3, index=False, sheet_name=feature_name)
            
            average_res = math.ceil(store_df['residual'].sum() / len(store_df['residual']))
            error = round(error, 2) 
            heading_name = store_name + " Average Residual Value: " + str(average_res) + " Error: " + str(error) + " TOTAL_WEEK: " + str(total_weeks)

            # Create a format to use in the merged range.
            merge_format = workbook.add_format({
                'bold': 1,
                'border': 1,
                'align': 'center',
                'valign': 'vcenter',
                'fg_color': 'yellow'})
            
            worksheet = writer.sheets[feature_name]
            worksheet.merge_range(row+2, 0, row+2, len(store_df.columns)-1, heading_name, merge_format)
            row += (len(store_df) + 4)
    writer.save()    
    
def get_summary_excel(all_test_df, summary_filename):
    '''
    Save Summary Excel sheet in Summary folder in given summary path name
    '''
    summary_writer = pd.ExcelWriter(summary_filename, engine="xlsxwriter")
    workbook = summary_writer.book
    row = 0
    column_list = []
    FLAG = True
    for feature_name, feature_dict in all_test_df.items():       
        summary_df = pd.DataFrame(columns=column_list) 
        # summary_df = pd.DataFrame(columns=['store_name', 'actual_footfall', 'predicted_footfall', 
        # 'error_perc','residual', 'avg_residual', 'MAE', 'r2','total_weeks', 'accuracy_12'])
        for store_name, store_dict in feature_dict.items():  
            removed_test_df = store_dict.pop('test_df')
            if FLAG is True:
                column_list = store_dict.keys()
                FLAG = False
            
            # Write to sheet
            summary_df = summary_df.append(store_dict, ignore_index=True)        

        summary_df['residual'] = summary_df['residual'].abs()

        summary_df.to_excel(summary_writer, startrow=row + 3, index=False)
        # Create a format to use in the merged range.
        merge_format = workbook.add_format({
            'bold': 1,
            'border': 1,
            'align': 'center',
            'valign': 'vcenter',
            'fg_color': 'yellow'})
        worksheet = summary_writer.sheets['Sheet1']
        worksheet.merge_range(row+2, 0, row+2, len(summary_df.columns)-1, str(feature_name), merge_format)    
        row += (len(summary_df) + 4)
    summary_writer.save()

def run_process(base_path, time, all_feat, L1_neurons, L2_neurons, epoch, batch_size, \
    learning_rate, normal_type, outlier, output_path, no_of_training_rows, run):
    '''
    Process to train baseline model and generate graphs ans excel sheets    
    '''
    store_folders = get_folders(base_path)
    all_feat = feature_combinations(all_feat)
    all_feat = [
            ['count_randomised'],
            ['count_vendor'],
            ['count_randomised', 'hour'],            
            ['avg_temp','count_randomised', 'hour'],
            ['count_randomised', 'hour', 'count_vendor'],
            ['avg_temp','count_randomised', 'hour', 'count_vendor'],
            ['hour', 'count_vendor'],
            ['min_temp', 'max_temp','count_randomised', 'hour', 'count_vendor'],
      ]

    all_feat_dict = {}
    for feature in tqdm(all_feat):
        print("Feature: ", feature)

        # concatenate feature list to name and shorten the feature name as Excel sheet name can contain maximum 31 characters
        concat_feature = [s.replace("_","") for s in feature]
        concat_feature = '_'.join([str(elem) for elem in concat_feature])
        concat_feature = concat_feature.replace('count','C')
        concat_feature = concat_feature.replace('ised','')
        concat_feature = concat_feature.replace('avgtemp','AvgT') #'min_temp', 'max_temp'
        concat_feature = concat_feature.replace('mintemp','MinT')
        concat_feature = concat_feature.replace('maxtemp','MaxT')

        print ("concat_feature: ", concat_feature)
        all_test_df_dict = {}
        for store in store_folders:
            print("STORE NAME: ", store)
            # Set Pattern and extract Store Name
            # pattern = '(.+)_preprocessed'
            # store_name = re.search(pattern, store)
            file_path = base_path + "/" + store + "/" + store + "_" + time

            # Get list of all folders
            date_folders = get_folders(file_path)            
            data = concatenate_all_data(file_path, date_folders)
            data = merge_weather(store, data)
            data = insert_day_hour(data, DropZeroFootfall=True, outlier_removal=True)
            data = OutlierRemoval(data)
            data = OutlierRemoval(data, feature='total')
            # data = data[data['
            # count_randomised']>0]
            # data = data[data['total'] > 0]
            data.sort_values(by='time')
            # Total weeks used for training_testing
            total_weeks = len(date_folders)
            total_rows = len(data)

            print(store ," total data after Preprocessing: ", len(data))

            # create train and testing groups along with other dataframe
            train, test, test_df = create_train_test(data)
            
            # take first 100 values in test
            # test = test[:no_of_training_rows]
            # test_df = test_df[:no_of_training_rows]

            X_train, y_train, X_test, y_test = xy_split(train, test, feature)
            # print(X_train, y_train, y_test)

            # Perform normalization
            if normal_type is not False:
                # print("normal_type is not false")
                X_train, y_train, X_test, y_test, sc_X_train, sc_y_train, sc_X_test, sc_y_test = normalization(X_train, y_train, X_test, y_test,normal_type)
            
            model, results = kfold(X_train, y_train, L1_neurons, L2_neurons, batch_size, epoch, learning_rate)
            
            # evaluate on test set
            model.fit(X_train, y_train)
            yhat = model.predict(X_test)

            # calculate MAE
            mae = mean_absolute_error(y_test, yhat)
            r2 = r2_score(y_test, yhat)
            mse = np.sqrt(mean_squared_error(y_test, yhat))  
            print("Store Name: ", store, 'MAE: %.3f' % mae, "Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
            
            # invert the transformation
            if normal_type is not False:
                yhat = yhat.reshape(-1, 1)
                yhat = sc_y_test.inverse_transform(yhat)                    
            
            output = np.array(yhat)
            input = np.array(X_test)

            test_df['predicted_cal'] = output
            test_df['residual'] = test_df['predicted_cal'] - test_df['total']
            test_df['residual'] = test_df['residual'].abs()
            test_df['ground_truth']= np.where((test_df['residual']>12), "TRUE", "FALSE")
            #new_df.to_excel("results/23_10_2021/regression_pred_StandardScaler_MINMAX(60min).xlsx", index=False)
            
            # Calculate overall error percentage and other statistics
            error_perc = abs(((test_df['total'].sum() - test_df['predicted_cal'].sum())/(test_df['total'].sum())))
            error_perc = "{:.1%}".format(error_perc)

            fal_val= len(test_df[test_df['ground_truth']=='FALSE'])
            tru_val = len(test_df[test_df['ground_truth']=='TRUE'])
            acc = (fal_val/(tru_val+fal_val)) * 100
            average_res = math.ceil(test_df['residual'].sum() / len(test_df['residual']))

            # Set figure name
            image_folder = time + "_L" + str(L1_neurons) + "_L" + str(L2_neurons) + "_E" + str(epoch) + "_B" + str(batch_size) + "_L" + str(learning_rate - int(learning_rate)) + "_" + normal_type
            make_directory(output_path, image_folder)
            make_directory(output_path + image_folder + "/", concat_feature)
            fig_name = output_path + image_folder + "/" + concat_feature + "/" + store + "_" + str(epoch) + ".png"

            # add dataframe and errort to dictionary
            all_test_df_dict[store] = {'test_df':test_df,                                                                                  
                                        'store_name': store,
                                        'actual_footfall': math.ceil(test_df['total'].sum()),
                                        'predicted_footfall': math.ceil(test_df['predicted_cal'].sum()),
                                        'error_perc': error_perc,
                                        'total_weeks': total_weeks,
                                        'total_rows:': total_rows,
                                        'MAE':mae, 
                                        'r2':r2,
                                        'MSE':mse,
                                        'residual': math.ceil(test_df['total'].sum() - test_df['predicted_cal'].sum()),
                                        'avg_residual': math.ceil(test_df['residual'].sum() / len(test_df['residual'])),
                                        # 'accuracy_12' : acc,
                                        'image_hyperlink':'=HYPERLINK("{}", "{}")'.format(image_folder + "/" + concat_feature + "/" + store + "_" + str(epoch) + ".png", store)
                                        }
                        
            line_plt(test_df, 'predicted_cal', mae, feature, fig_name, time, L1_neurons, L2_neurons, epoch, normal_type, learning_rate, batch_size)
        all_feat_dict[concat_feature] = all_test_df_dict


    # print(all_feat_dict)
    if normal_type is not False:
        excel_file_name = output_path + "AllResults/" + "AllResult_" + time + "_L" + str(L1_neurons) + "_L" + str(L2_neurons) + "_E" + str(epoch) + "_B" + str(batch_size) + "_L" + str(learning_rate - int(learning_rate)) + "_" + normal_type + ".xlsx"
        # summary_filename = output_path + "Summary/L" + str(learning_rate) + "/" + time + "_L" + str(L1_neurons) + "_L" + str(L2_neurons) + "_E" + str(epoch) + "_B" + str(batch_size) + "_L-" + str(learning_rate - int(learning_rate)) + "_" + normal_type + ".xlsx"
        summary_filename = output_path + "/" + time + "_L" + str(L1_neurons) + "_L" + str(L2_neurons) + "_E" + str(epoch) + "_B" + str(batch_size) + "_L-" + str(learning_rate - int(learning_rate)) + "_" + normal_type + " (Summary).xlsx"

    else:
        excel_file_name = output_path + "AllResults/" + "AllResult_" + time + "_L" + str(L1_neurons) + "_L" + str(L2_neurons) + "_E" + str(epoch) + "_B" + str(batch_size) + "_L" + str(learning_rate - int(learning_rate)) + "_" + "NO_OPT.xlsx"
        # summary_filename = output_path + "Summary/L" + str(learning_rate) + "/" + time + "_L" + str(L1_neurons) + "_L" + str(L2_neurons) + "_E" + str(epoch) + "_B" + str(batch_size) + "_L-" + str(learning_rate - int(learning_rate)) + "_" + "NO_OPT.xlsx"
        summary_filename = output_path + "/" + time + "_L" + str(L1_neurons) + "_L" + str(L2_neurons) + "_E" + str(epoch) + "_B" + str(batch_size) + "_L-" + str(learning_rate - int(learning_rate)) + "_" + "NO_OPT (Summary).xlsx"

    print("excel_file_name: ", excel_file_name)
    print("summary_filename: ",summary_filename)
    # Generate Summary and Result Excel
    get_single_excel(all_feat_dict, excel_file_name)
    get_summary_excel(all_feat_dict, summary_filename)