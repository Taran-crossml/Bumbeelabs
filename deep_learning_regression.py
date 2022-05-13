from utils import is_dir, get_health_stats, generate_stats, clean_aggregate_data, create_train_test,x_y_split, run_model_pipeline, merge_store_data
import os
import re
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import SGD
from tabulate import tabulate
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from visualization import line_chart, prediction_line_chart
from sklearn.metrics import mean_absolute_error
from scipy.signal import lfilter
from itertools import combinations
from datetime import datetime
# Feature Scaling (ignore possible warnings due to conversion of integers to floats)
from sklearn.preprocessing import StandardScaler

def regression_model(X_train, X_test, Y_train, Y_test):
    model = Sequential()
    model.add(Dense(50, activation='sigmoid', input_dim=1)) # activation='tanh'
    #model.add(Dense(50, activation='sigmoid')) # ,activation='relu'
    #model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(1, activation='softmax')) # , activation='sigmoid'
    model.compile(optimizer='adam', # sgd
                  loss='mean_squared_error', # sparse_categorical_crossentropy categorical_crossentropy
                  metrics=['accuracy']
                  ) # optimizer='adam'

    print(model.summary())
    history = model.fit(X_train, Y_train, epochs=2, batch_size=6, validation_data=(X_test, Y_test))  #
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(score)
    print("Accuracy", score[1])

def file_path():
    data_folders = []
    base_path = r"merged (210813-210829 and 210830-210905)_5min"
    for d in os.listdir(base_path):
        if is_dir(os.path.join(base_path, d)):
            data_folders.append(d)
    #print(data_folders)

    for folder in tqdm(data_folders):
        path = os.path.join(base_path, folder, f'{folder}_preprocessed.csv')
        print(path)

def correlations():
    assumptions = {'working_hours': {'start': 7, 'end': 23}}
    data_folders = []
    base_path = r"merged (210813-210829 and 210830-210905)_5min"
    for d in os.listdir(base_path):
        if is_dir(os.path.join(base_path, d)):
            data_folders.append(d)
    print("folder: ", data_folders)

    # initialze the excel writer
    writer = pd.ExcelWriter('visualization/correlation_mapping/5min/All_correlations(210813-210829 and 210830-210905)_5min.xlsx',
                            engine='xlsxwriter')

    all_df = {}
    for data_folder in data_folders:
        data = clean_aggregate_data(base_path, data_folder, assumptions)
        correlations = data.corr()
        all_df[data_folder] = correlations
        # correlations.to_csv("visualization/correlation_mapping/"+data_folder+".csv")
        coor = sns.heatmap(correlations)
        plt = coor.get_figure()
        plt.savefig("visualization/correlation_mapping/5min/" + data_folder + ".png", pad_inches=0.11, bbox_inches='tight',
                    dpi=300)
        plt.clf()

    # now loop thru and put each on a specific sheet
    for sheet, frame in all_df.items():  # .use .items for python 3.X
        frame.to_excel(writer, sheet_name=sheet)

    # critical last step
    writer.save()

def mac_read():
    mac_file = r"mac_folder/macaddress.io-db.csv"
    mac_info = pd.read_csv(mac_file)
    mac_info = mac_info[:10]
    mac_info['oui'].str.slice(0, 8)
    print(mac_info['oui'])

def merged_files(minute):
    assumptions = {'working_hours': {'start': 7, 'end': 23}}
    data_folders = []
    #minute = "60min"
    base_path = r"inferno_odenplan_allweeks/inferno_odenplan_" + minute
    # base_path = r"inferno_odenplan_allweeks"
    for d in os.listdir(base_path):
        if is_dir(os.path.join(base_path, d)):
            data_folders.append(d)
    return base_path, data_folders, assumptions

def resample_wifi(wifi, time_limit):
    wifi.index = wifi['node_time']
    # time_limit = '2min'
    wifi_res_count = wifi[['node_time', 'mac', 'randomised', 'is_vendor']].drop_duplicates() \
        .resample(time_limit).sum().reset_index(). \
        rename(columns={'randomised': 'count_randomised', 'is_vendor': 'count_vendor'}) \
        [['count_randomised', 'count_vendor']]

    wifi_res_perc = wifi[['node_time', 'mac', 'randomised', 'is_vendor']].drop_duplicates() \
        .resample(time_limit).mean().reset_index(). \
        rename(columns={'randomised': 'perc_randomised', 'is_vendor': 'perc_vendor'})

    apple_count = wifi[['node_time', 'mac', 'randomised', 'is_vendor', 'apple']].drop_duplicates() \
        .resample(time_limit).sum().reset_index()[['apple']]

    not_apple_count = wifi[['node_time', 'mac', 'randomised', 'is_vendor', 'not_apple']].drop_duplicates() \
        .resample(time_limit).sum().reset_index()[['not_apple']]
    # wifi_resampled['hour'] = pd.to_datetime(wifi['node_time']).dt.hour

    wifi_resampled = pd.concat([wifi_res_perc, wifi_res_count, apple_count, not_apple_count], axis=1)

    #wifi_resampled.to_excel("experiment/wifi_resampled.xlsx", index=False)
    return wifi_resampled

def bumbee_internal():
    df = pd.read_csv('bumbee_closed_env/survey_samples_2021.09.21.csv')
    df['node_time'] = pd.to_datetime(df['node_time'])
    df['node_time'] = df['node_time'].dt.strftime("%Y-%m-%d %H:%M:%S")
    # print(wifi['node_time'])
    df['node_time'] = pd.to_datetime(df['node_time'])

    #print(df[['node_time']])

    mac_info = pd.read_csv('mac_folder/macaddress.io-db.csv')
    df['oui'] = df['mac'].str.slice(0, 8)
    mac_info['oui'] = mac_info['oui'].str.slice(0, 8)
    mac_info['oui'] = mac_info['oui'].str.lower()

    df = df.merge(mac_info, on='oui', how='left')
    df['randomised'] = df['companyName'].isnull().astype('int')
    df['is_vendor'] = 1 - df['companyName'].isnull().astype('int')

    df.loc[df['companyName'] == 'Apple, Inc', 'apple'] = 1
    df.loc[(df['companyName'] != 'Apple, Inc') & (~df['companyName'].isnull()), 'not_apple'] = 1
    df['hour_min'] = pd.to_datetime(df['node_time'])
    df['hour_min'] = df['hour_min'].dt.strftime("%H:%M:00")
    df.to_excel('bumbee_closed_env/device_merged_and_companies.xlsx')
    time = '30s'
    df = resample_wifi(df, time)
    print(df.columns)
    apple_name = "bumbee_closed_env/apple_not_apple" + time + ".png"
    vendor_random_name = "bumbee_closed_env/randomised_vendor" + time + ".png"

    line_chart(df, "node_time", "apple", "not_apple", figurename=apple_name)
    line_chart(df, "node_time", "count_vendor", "count_randomised", figurename=vendor_random_name)

    df.to_excel("bumbee_closed_env/merged_processed.xlsx")
    # df = df[df['rssi'] > -67]

# define base model
def baseline_model(X_train, y_train, X_test, y_test, L1_neurons, L2_neurons, epoch):
    # define the keras model
    model = Sequential()
    # standard 20 neurons | Original activation 'relu'
    model.add(Dense(L1_neurons, input_dim=X_train.shape[1], activation='relu', kernel_initializer='he_normal'))
    # model.add(Dropout(0.5))
    # standard 10 neurons | Original activation 'relu'
    model.add(Dense(L2_neurons, activation='relu', kernel_initializer='he_normal'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))  # Original activation 'linear' ,activation='linear'
    # compile the keras model
    # According to Kingma and Ba (2014) Adam has been developed for
    # "large datasets and/or high-dimensional parameter spaces".
    model.compile(loss='mse', optimizer='adam')
    # fit the keras model on the dataset
    model.fit(X_train, y_train, epochs=epoch, batch_size=12, verbose=2)
    # evaluate on test set
    yhat = model.predict(X_test)
    error = mean_absolute_error(y_test, yhat)
    print('MAE: %.3f' % error)
    output = np.array(yhat)
    input = np.array(X_test)
    return input, output, error

def lfilter_normalization(df,col_name):
    n = 3  # larger n gives smoother curves
    b = [1.0 / n] * n  # numerator coefficients
    a = 1  # denominator coefficient
    y_total = lfilter(b, a, df[col_name])
    return y_total

def change_col_name(folder_name, features):
    feature = list(map(lambda st: str.replace(st, "_", ""), features))
    col_name = '_'.join([str(elem) for elem in feature])
    col_name = col_name + "_" + folder_name
    print("col_name", col_name, " ,folder: ", folder_name)

def merge_inferno(time):
    ### Merge store data for two weeks
    dest_folder = "inferno_weeks_merged"
    folders = ["inferno_odenplan_allweeks/inferno_odenplan_30min"]
    sub_folders = ["210601-210606", "210607-210616",
                   "210617-210627", "210627-210706",
                   "210813-210829", "210830-210905"]
    #merge_store_data(folders, sub_folders, dest_folder)

def merge_all_week_data(minute):
    base_path = r'inferno_odenplan_allweeks/inferno_odenplan_' + minute
    data_folders = next(os.walk(base_path))[1]
    dfs = []
    for folder in tqdm(data_folders):
        path = os.path.join(base_path, folder, f'{folder}_preprocessed.csv')
        dfs.append(pd.read_csv(path))
    consolidated = pd.concat([df for df in dfs], axis=0)
    return consolidated

def clean_data(df, assumptions):
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    q1 = (df['time'].dt.hour > assumptions['working_hours']['start']) & (
            df['time'].dt.hour <= assumptions['working_hours']['end'])
    res = df[q1].dropna().sample(frac=1, random_state=42).reset_index(drop=True)
    res = res.sort_values(by='time')
    return res

def xy_split(train, test, feature):
    X_train = train.drop(train.columns.difference(feature), axis=1).values
    y_train = train.drop(train.columns.difference(['total']), axis=1).values # train['total'].values
    X_test = test.drop(train.columns.difference(feature), axis=1).values
    y_test = test.drop(train.columns.difference(['total']), axis=1).values # test['total'].values
    return X_train, y_train, X_test, y_test

def outlier_detection(df):
    print("Full DF: ", len(df))
    df = df.reset_index(drop=True)
    print("Rest DF: ", len(df))
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in days:
        day_df = df[df['day'] == day]
        Q1 = day_df['total'].quantile(0.25)
        Q3 = day_df['total'].quantile(0.75)
        IQR = Q3 - Q1
        # next step will help to convert outliers into true values
        day_outlier_index = day_df[(day_df['count_randomised'] < (Q1 - 1.5 * IQR)) | (day_df['count_randomised'] > (Q3 + 1.5 * IQR))].index
        #print("Outliers index: ", day_outlier_index)
        df = outlier_saving_dropping(df, day_outlier_index, day)
    return df

def outlier_saving_dropping(df, day_outlier_index, day, save= False):
    if len(day_outlier_index) >= 1:
        if save is True:
            file_name = "visualization/" + day + ".xlsx"
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

def get_folders(base_path):
    data_folder = []
    for folder in os.listdir(base_path):
        if is_dir(os.path.join(base_path, folder)):
            data_folder.append(folder)
    return data_folder

def get_single_excel(all_test_df, file_name, summary_filename):
    writer = pd.ExcelWriter(file_name, engine="xlsxwriter")
    workbook = writer.book
    row = 0
    for store_name, store_df in all_test_df.items():
        store_df.to_excel(writer, startrow=row + 3, index=False)
        worksheet = writer.sheets['Sheet1']
        #print("Store_name: ", store_name, " Sum: ", store_df['pos_residual'].sum())
        average_res = math.ceil(store_df['pos_residual'].sum() / len(store_df['pos_residual']))
        store_name = store_name + " Average Residual Value: " + str(average_res)

        # Create a format to use in the merged range.
        merge_format = workbook.add_format({
            'bold': 1,
            'border': 1,
            'align': 'center',
            'valign': 'vcenter',
            'fg_color': 'yellow'})

        worksheet.merge_range(row+2, 0, row+2, len(store_df.columns)-1, store_name, merge_format)
        #worksheet.write(row + 2, 0, store_name)
        row += (len(store_df) + 4)
    writer.save()

    summary_df = pd.DataFrame(columns=['store_name', 'actual_footfall', 'predicted_footfall', 'residual', 'avg_residual'])
    for store_name, store_df in all_test_df.items():
        df2 = {'store_name': store_name,
               'actual_footfall': math.ceil(store_df['total'].sum()),
               'predicted_footfall': math.ceil(store_df['predicted_cal'].sum()),
               'residual': math.ceil(store_df['total'].sum() - store_df['predicted_cal'].sum()),
               'avg_residual': math.ceil(store_df['pos_residual'].sum() / len(store_df['pos_residual']))
               }
        summary_df = summary_df.append(df2, ignore_index=True)
    summary_df['residual'] = summary_df['residual'].abs()

    writer = pd.ExcelWriter(summary_filename, engine="xlsxwriter")
    workbook = writer.book    
    row = 0
    summary_df.to_excel(writer, startrow=row + 1, index=False)
    # Create a format to use in the merged range.
    merge_format = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'fg_color': 'yellow'})
    worksheet = writer.sheets['Sheet1']
    worksheet.merge_range(row, 0, row, len(summary_df.columns) - 1, str(summary_filename), merge_format)    
    writer.save()
    #summary_df.to_excel(summary_filename, index=False)

# define base model
def Kfold_baseline_model(X_train, L1_neurons=30, L2_neurons=20):
    def bm():
        model = Sequential()
        model.add(Dense(L1_neurons, input_dim=X_train.shape[1], activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(L2_neurons, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model
    return bm

def all_nn_train():
    base_path = 'preprocessed files_14102021'
    time = '30min'
    feature = ['count_randomised']
    epoch = 100
    batch_size= 12
    normalization = True
    flag = 1
    store_folders = get_folders(base_path)
    print(store_folders)
    base_pth = "visualization/merged_results_nn/"
    all_test_df = {}
    print("Feature: ", feature)
    for store in store_folders:
        #print("Store Name: ", store)
        pattern = '(.+)_preprocessed'
        store_name = re.search(pattern, store)
        # print(store_name.group(1))
        file_path = base_path + "/" + store + "/" + store_name.group(1) + "_" + time
        date_folders = get_folders(file_path)
        dfs = [pd.read_csv(os.path.join(file_path, folder, f'{folder}_preprocessed.csv')) for folder in date_folders]
        data = pd.concat([df for df in dfs], axis=0)
        data = data[data['total'] > 0]  # Select rows with footfall greater than 0
        data['hour'] = pd.to_datetime(data['time'])
        data['hour'] = data['hour'].dt.hour
        #print(data.columns)
        data['day'] = data['time'].astype('datetime64[D]')
        data['day'] = data['day'].dt.day_name()
        data['day'].replace(to_replace=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'],
                            value=[1, 2, 3, 4, 5, 6, 7], inplace=True)
        # drop outliers
        data = outlier_detection(data)

        # create train and testing groups along with other dataframe
        train, test, test_df = create_train_test(data)

        # take first 100 values in test
        test = test[:50]
        test_df = test_df[:50]

        X_train, y_train, X_test, y_test = xy_split(train, test, feature)
        # print(X_train, y_train, y_test)

        if normalization is True:
            sc_X_train = StandardScaler()
            X_train = sc_X_train.fit_transform(X_train)

            sc_y_train = StandardScaler()
            y_train = sc_y_train.fit_transform(y_train)

            sc_X_test = StandardScaler()
            X_test = sc_X_test.fit_transform(X_test)

            sc_y_test = StandardScaler()
            y_test = sc_y_test.fit_transform(y_test)

        #print(train.columns)

        model = KerasRegressor(build_fn=Kfold_baseline_model(X_train), nb_epoch=epoch, batch_size=batch_size, verbose=False)
        kfold = KFold(n_splits=10)
        results = cross_val_score(model, X_train, y_train, cv=kfold)
        # print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

        # evaluate on test set
        model.fit(X_train, y_train)

        yhat = model.predict(X_test)

        if normalization is True:
            yhat = yhat.reshape(-1, 1)
            print("yhat: ", yhat)
            yhat = sc_y_test.inverse_transform(yhat)
            

        error = mean_absolute_error(y_test, yhat)
        print("Store Name: ", store, 'MAE: %.3f' % error, "Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
        output = np.array(yhat)
        input = np.array(X_test)

        # input, output, error = baseline_model(X_train, y_train, X_test, y_test, 20, 10, epoch)

        col_name = 'predicted_cal'
        test_df['time'] = pd.to_datetime(test_df['time'])
        test_df[col_name] = output
        # pp_df.index = pp_df['time']
        fig, ax = plt.subplots(figsize=(10, 3))

        ax.plot(test_df["time"], test_df.total, linewidth=1, color='#e9ecef')
        ax.plot(test_df["time"], test_df[col_name], linewidth=1, color='#0a9396')

        # Set tick parameters and Make y axis tics not visible
        ax.tick_params(axis='x', direction='in', length=3, color="black", labelsize=8)
        ax.tick_params(axis='y', direction='in', length=3, color="black", labelsize=8)

        fonts = {"fontname": "Times New Roman", "fontsize": 8}
        ax.set_xlabel('', fontdict=fonts)
        ax.set_ylabel('', fontdict=fonts)

        ax.text(0.88, 0.88, str('MAE: %.3f' % error), horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
        ax.legend(['Camera Count', 'Predicted Count'],
                  bbox_to_anchor=(0.0, 1.00, 1, 0.3),
                  loc='center',
                  ncol=3,
                  frameon=False)

        name = base_pth + store_name.group(1) + "_" + str(epoch)
        test_df['residual'] = test_df['predicted_cal'] - test_df['total']
        test_df['pos_residual'] = test_df['residual'].abs()
        #test_df.to_excel(name+".xlsx", index=False)
        all_test_df[store_name.group(1)] = test_df
        file_name = base_pth + store_name.group(1) + "_" + str(epoch) + ".png"
        plt.savefig(file_name, pad_inches=0.11, bbox_inches='tight', dpi=600)

    single_excel_file_name = base_pth + "all_results" + str(epoch) + "epochs_" + str(batch_size) + "batch" + ".xlsx"
    summary_filename = base_pth + str(epoch) + "epochs_" + str(batch_size) + "batch" + "_summary.xlsx"
    print("summary_filename: ", summary_filename)
    print("summary_filename: ",summary_filename)
    get_single_excel(all_test_df, single_excel_file_name, summary_filename)

if __name__ == '__main__':
    #file_path()
    #correlations()
    #mac_read()
    all_nn_train()
    exit()
    # Experiment Function for internal wifi sheet
    # bumbee_internal()
    # exit()
    #merge_inferno()
    outlier = False
    all_data = merge_all_week_data('30min')
    assumptions = {'working_hours': {'start': 7, 'end': 23}}
    all_data = clean_data(all_data, assumptions)
    if outlier is True:
        all_data = outlier_detection(all_data)


    # Select rows with footfall greater than 0
    all_data = all_data[all_data['total'] > 0]
    print(all_data)

    test_df = pd.read_csv('210914-211005/inferno_odenplan/inferno_odenplan_preprocessed.csv')
    test_df = clean_data(test_df, assumptions)
    test_df = test_df[test_df['total'] > 0]  # Select rows with footfall greater than 0
    test_df = test_df[:100]

    feature = ['count_randomised']
    X_train = all_data.drop(all_data.columns.difference(feature), axis=1).values
    y_train = all_data['total'].values
    print(len(X_train))
    X_test = test_df.drop(test_df.columns.difference(feature), axis=1).values
    y_test = test_df['total'].values
    input, output, error = baseline_model(X_train, y_train, X_test, y_test, 20, 10)

    col_name = 'predicted_cal'
    test_df['time'] = pd.to_datetime(test_df['time'])
    test_df[col_name] = output
    # pp_df.index = pp_df['time']
    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(test_df["time"], test_df.total, linewidth=1, color='#e9ecef')
    ax.plot(test_df["time"], test_df[col_name], linewidth=1, color='#0a9396')

    # Set tick parameters and Make y axis tics not visible
    ax.tick_params(axis='x', direction='in', length=3, color="black", labelsize=8)
    ax.tick_params(axis='y', direction='in', length=3, color="black", labelsize=8)

    fonts = {"fontname": "Times New Roman", "fontsize": 8}
    ax.set_xlabel('', fontdict=fonts)
    ax.set_ylabel('', fontdict=fonts)

    ax.text(0.88, 0.88, str('MAE: %.3f' % error), horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax.transAxes)
    ax.legend(['Camera Count', 'Predicted Count'], bbox_to_anchor=(0.0, 1.00, 1, 0.3), loc='center', ncol=3,
                 frameon=False)
    plt.savefig("on_unseen_210914-211005data_new.png", pad_inches=0.11, bbox_inches='tight', dpi=600)
    exit()

    # set training columns features
    start = datetime.now()
    all_cols = ['perc_randomised', 'perc_vendor', 'day', 'count_vendor', 'count_randomised']
    all_cols = sum([list(map(list, combinations(all_cols, i))) for i in range(len(all_cols) + 1)], [])
    all_cols.remove([])
    print(all_cols)
    time = '5min'
    base_path, data_folders, assumptions = merged_files(minute=time)

    for feature in all_cols:
        feature_list = feature
        print("Features:", feature)

        fig, ax = plt.subplots(nrows=len(data_folders)+1, ncols=1, figsize=(10, 6))
        # set plot font
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 8
        # Set plot border line width
        plt.rcParams['axes.linewidth'] = 0.5
        plt.rcParams['xtick.color'] = 'black'
        # Set tick parameters
        plt.rc('xtick', direction='in', color="black")
        plt.rc('ytick', direction='in', color="black")

        print("ax: ",ax)

        fig.subplots_adjust(bottom=0.2) # base bottom=0.3

        fonts = {"family": "Times New Roman", "size": 8} # , "weight": 'bold'
        plt.rc('font', **fonts)

        row_num = 0
        for folder in data_folders:
            print("Store Name: ", folder)
            data = clean_aggregate_data(base_path, folder, assumptions)
            # remove rows with total footfall less than 0
            # and insert day column and convert into features
            data = data[data['total']>0] # Select rows with footfall greater than 0
            data['day'] = (data['time']).astype('datetime64[D]')
            data['day'] = data['day'].dt.day_name()
            data['day'].replace(to_replace=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'],
                                value=[1, 2, 3, 4, 5, 6, 7], inplace=True)
            #print(data[['day','time','total','count_vendor']][:10])
            # all_cols = list(data.columns) #.remove('time') #-['time']
            # all_cols.remove('time')
            # all_cols.remove('total')

            # Replace substring in list of strings

            # create train and testing groups along with other dataframe
            train, test, test_df = create_train_test(data)
            print(train.columns)

            print("Feature: ", feature)
            X_train, y_train, X_test, y_test = xy_split(train, test, feature)
            # X_train = train.drop(train.columns.difference(feature),axis=1).values
            # y_train = train['total'].values
            # X_test = test.drop(train.columns.difference(feature),axis=1).values
            # y_test = test['total'].values
            print("X_train.shape: ", X_train.shape)
            print("y_train: ", y_train.shape)
            print("\n\n")

            L1_neurons = 30
            L2_neurons = 20

            input, output, error = baseline_model(X_train, y_train, X_test, y_test, L1_neurons, L2_neurons)

            # print(input[:10], output[:10])
            # test_df[col_name] = output
            # test_df['input'] = input_text[:, 0]
            # test_df.to_excel("nn_output(per_randomized+day).xlsx")
            #exit()
            col_name = change_col_name(folder, feature)

            test_df['time'] = pd.to_datetime(test_df['time'])
            test_df[col_name] = output
            # pp_df.index = pp_df['time']

            ax[row_num].plot(test_df["time"], test_df.total, linewidth=1, color='#e9ecef')
            ax[row_num].plot(test_df["time"], test_df[col_name], linewidth=1, color='#0a9396')

            # Set tick parameters and Make y axis tics not visible
            ax[row_num].tick_params(axis='x', direction='in', length=3, color="black", labelsize=8)
            ax[row_num].tick_params(axis='y', direction='in', length=3, color="black", labelsize=8)

            fonts = {"fontname": "Times New Roman", "fontsize": 8}
            ax[row_num].set_xlabel('', fontdict=fonts)
            ax[row_num].set_ylabel('', fontdict=fonts)

            ax[row_num].text(0.88, 0.88, str('MAE: %.3f' % error) + " ,Week: "+folder, horizontalalignment='center', verticalalignment='center',
                     transform=ax[row_num].transAxes)
            row_num += 1

        #fig.text(0.1, 0.20, 'Feature: '+str(feature_list))
        fig.text(0.1, 0.15, 'Feature: '+str(feature_list))
        fig.text(0.1, 0.13, 'L1_neurons: '+str(L1_neurons))
        fig.text(0.1, 0.11, 'L2_neurons: '+str(L2_neurons)) # last 0.5

        ax[0].legend(['Camera Count', 'Predicted Count'], bbox_to_anchor=(0.0, 1.00, 1, 0.3), loc='center', ncol=3,
                     frameon=False)

        ax[len(data_folders)].axis('off')
        # fig.text(0.1, 0.20, 'Feature: '+str(feature_list))
        # fig.text(0.50, 0.20, 'Store: '+str(data_folders))
        # fig.text(0.1, 0.10, 'L1_neurons: '+str(L1_neurons))
        # fig.text(0.50, 0.10, 'L2_neurons: '+str(L2_neurons))
        # plt.tight_layout()

        # set the spacing between subplots
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

        # set Figure Names
        feature = list(map(lambda st: str.replace(st, "_", ""), feature_list))
        name = '_'.join([str(elem) for elem in feature])
        figurename = "visualization/" + time + "_" + name + ".png"

        # Save and Close the figure
        print("Figurename: ",figurename)
        plt.savefig(figurename, pad_inches=0.11, bbox_inches='tight', dpi=600)
        plt.close()

        plt.show()
    print("Total Time Taken : ",datetime.now() - start)

    # for data in list(zip(y_t, yhat)):
    #     print(round(data[1], 2))
    #

    #
    # new_input = [2.12309797]
    # # get prediction for new input
    # new_output = model.predict(new_input)
    # # summarize input and output
    # print(new_input, new_output)