import os
import numpy as np
import pandas as pd
from pandas import ExcelWriter
import matplotlib.pyplot as plt
from itertools import combinations
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


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
    except:
        print(folder_name, " already exists")

def is_dir(d):
    '''
    Checks if directory is present or not
    '''
    if os.path.isdir(d) and d not in ['.ipynb_checkpoints','.vscode','__pycache__']:
        return True
    else:
        return False

def clean_data(df, assumptions):
    '''
    gives clean dataframe based on assumptions

    :param df: Dataframe
    :param assumptions: Timestamp assumption dictionary
    :return: clean dataframe based on assumption dictionary
    '''
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    q1 = (df['time'].dt.hour > assumptions['working_hours']['start']) & (
            df['time'].dt.hour <= assumptions['working_hours']['end'])
    res = df[q1].dropna().sample(frac=1, random_state=42).reset_index(drop=True)
    res = res.sort_values(by='time')
    return res

def baseline_model(X_train, y_train, X_test, y_test, L1_neurons, L2_neurons, epoch, L1_dropout=False, L2_dropout=False):
    '''
    Baseline Regression Model

    :param X_train: Training Input vector
    :param y_train: Training Target Vector
    :param X_test: Testing Input vector
    :param y_test: Testing Target vector
    :param L1_neurons: Number of Neurons in First layer
    :param L2_neurons: Number of Neurons in Second layer
    :return: input array, output array, error rate (MAE)
    '''
    # define the keras model
    model = Sequential()
    # standard 20 neurons | Original activation 'relu'
    model.add(Dense(L1_neurons, input_dim=X_train.shape[1], activation='relu', kernel_initializer='he_normal'))
    if L1_dropout is True:
        model.add(Dropout(0.5))
    # standard 10 neurons | Original activation 'relu'
    model.add(Dense(L2_neurons, activation='relu', kernel_initializer='he_normal'))
    if L2_dropout is True:
        model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))  # Original activation 'linear' ,activation='linear'
    # compile the keras model
    # According to Kingma and Ba (2014) Adam has been developed for
    # "large datasets and/or high-dimensional parameter spaces".
    model.compile(loss='mse', optimizer='adam')
    # fit the keras model on the dataset
    model.fit(X_train, y_train, epochs=epoch, batch_size=12, verbose=1)
    # evaluate on test set
    yhat = model.predict(X_test)
    r2 = r2_score(y_test, yhat)
    mae = mean_absolute_error(y_test, yhat)
    mse = np.sqrt(mean_squared_error(y_test, yhat))   
    output = np.array(yhat)
    input = np.array(X_test)
    return input, output, mae, mse, r2

def xy_split(train, test, feature):
    X_train = train.drop(train.columns.difference(feature), axis=1).values
    y_train = train['total'].values
    X_test = test.drop(train.columns.difference(feature), axis=1).values
    y_test = test['total'].values
    return X_train, y_train, X_test, y_test

def create_train_test(df,frac=0.70):
    '''
    Needed for statsmodels
    '''
    train_idx = round(df.shape[0]*frac)
    test_idx = train_idx+1
    df = df.sort_values(by = 'time')
    df.style.hide_index()
    test_df = df[test_idx:][['time','total','count_randomised' ,'count_vendor',  'hour']]
    return df[:train_idx] , df[test_idx:], test_df

def merged_files(base_path):
    '''
    Return the list of sub-folders in base-path

    :param base_path: base path directory
    :return: list of sub-directories
    '''
    data_folders = []
    for d in os.listdir(base_path):
        if is_dir(os.path.join(base_path, d)):
            data_folders.append(d)
    return base_path, data_folders

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

def insert_day(data, DropZeroFootfall=True):
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
    data['day'] = data['time'].astype('datetime64[D]')
    data['day'] = data['day'].dt.day_name()
    data['day'].replace(to_replace=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'],
                        value=[1, 2, 3, 4, 5, 6, 7], inplace=True)
    return data

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

def change_col_name(folder_name, features):
    '''
    Return column names based on feature and file name

    :param folder_name: Week name/number
    :param features: Number of features
    :return: merged column name
    '''
    feature = list(map(lambda st: str.replace(st, "_", ""), features))
    col_name = '_'.join([str(elem) for elem in feature])
    col_name = col_name + "_" + folder_name
    return col_name

def sub_plots(graph_dict, length, minute, data_folders, feature_list, folder_name, L1_neurons, L2_neurons, L1_dropout, L2_dropout, epoch):
    '''
    Subplot generation function

    :param graph_dict: dictionary with all dataframe and vectors
    :param length: Total number of sub-plots to be generated
    :param data_folders: all data-folders
    :param feature_list: feature list
    :param L1_neurons: Number of Neurons in First layer
    :param L2_neurons: Number of Neurons in Second layer
    :param minute: minute in string format
    :param epoch: iterations of neural networks
    :return: generated sub-plots
    '''
    fig, ax = plt.subplots(nrows=length, ncols=1, figsize=(10, 6))
    # set plot font
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 8
    # Set plot border line width
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.color'] = 'black'
    # Set tick parameters
    plt.rc('xtick', direction='in', color="black")
    plt.rc('ytick', direction='in', color="black")
    row_num = 0
    for file, list in graph_dict.items():
        test_df = graph_dict[file]["test_df"]
        col_name = graph_dict[file]["col_name"]
        error = graph_dict[file]["error"]
        ax[row_num].plot(test_df["time"], test_df.total, linewidth=1, color='#e9ecef')
        ax[row_num].plot(test_df["time"], test_df[col_name], linewidth=1, color='#0a9396')

        # Set tick parameters and Make y axis tics not visible
        ax[row_num].tick_params(axis='x', direction='in', length=3, color="black", labelsize=8)
        ax[row_num].tick_params(axis='y', direction='in', length=3, color="black", labelsize=8)

        fonts = {"fontname": "Times New Roman", "fontsize": 8}
        ax[row_num].set_xlabel('', fontdict=fonts)
        ax[row_num].set_ylabel('', fontdict=fonts)

        ax[row_num].text(0.88, 0.88, str('MAE: %.3f' % error) + " ,Week: " + file, horizontalalignment='center',
                         verticalalignment='center',
                         transform=ax[row_num].transAxes)
        row_num += 1

    # Insert Experiment Hyperparamter's
    fig.text(0.1, 0.15, 'Feature: ' + str(feature_list))
    fig.text(0.1, 0.13, 'L1_neurons: ' + str(L1_neurons))
    fig.text(0.1, 0.11, 'L2_neurons: ' + str(L2_neurons))
    fig.text(0.1, 0.09, 'L1_dropout (0.5): ' + str(L1_dropout))
    fig.text(0.1, 0.07, 'L2_dropout (0.5): ' + str(L2_dropout)) # last y = 0.5
    fig.text(0.1, 0.07, 'Epochs: ' + str(epoch))

    ax[0].legend(['Camera Count', 'Predicted Count'], bbox_to_anchor=(0.0, 1.00, 1, 0.3), loc='center', ncol=3,
                 frameon=False)

    ax[len(data_folders)].axis('off')

    # set the spacing between subplots
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

    # set Figure Names
    feature_name = [s.replace("_","") for s in feature_list]
    name = '_'.join([str(elem) for elem in feature_name])

    figurename = folder_name + "/" + minute + "_" + name + ".png"

    # Save and Close the figure
    plt.savefig(figurename, pad_inches=0.11, bbox_inches='tight', dpi=600)
    plt.close()

def generate_excel(feature_name, folder_name, graph_dict):
    '''
    Generated Excel sheets for each trained Dataframe

    :param feature_name: Feature name which will be used as file name
    :param graph_dict: dictionary with all dataframe and vectors
    :return: Excel File containing Training output
    '''
    print("Writing Neural Network Output to Excel File")
    file_name = folder_name + feature_name + ".xlsx"
    writer = ExcelWriter(file_name)
    for file, value in graph_dict.items():
        test_df = graph_dict[file]["test_df"]
        test_df.to_excel(writer, sheet_name=file, index=False)
    writer.save()
    print("Writing Completed")

def outlier_saving_and_dropping(df, day_outlier_index, day, save=False):
    '''
    Outliers removal and saving

    :param df: Main Dataframe
    :param day_outlier_index: outlier index
    :param day: day for which the outliers are being removed
    :param save: False-> Donot save file. True-> Save file
    :return: return clean dataframe
    '''
    if len(day_outlier_index) >= 1:
        # Save outlier to file if you want to
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

        # Drop outlier index
        df = df.drop(day_outlier_index)
        return df
    else:
        return df

def outlier_detection(df):
    '''
    Detect Outlier for the day

    :param df: dataframe
    :return: dataframe after removing outliers
    '''
    df = df.reset_index(drop=True)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in days:
        day_df = df[df['day'] == day]
        Q1 = day_df['total'].quantile(0.25)
        Q3 = day_df['total'].quantile(0.75)
        IQR = Q3 - Q1
        # next step will help to convert outliers into true values
        day_outlier_index = day_df[(day_df['total'] < (Q1 - 1.5 * IQR)) | (day_df['total'] > (Q3 + 1.5 * IQR))].index
        #print("Outliers index: ", day_outlier_index)
        df = outlier_saving_and_dropping(df, day_outlier_index, day)
    return df

def process(base_path, feat, minute, assumptions, L1_neurons, L2_neurons, L1_dropout=False, L2_dropout=False, epoch=50, outlier_removal=False):
    '''
    Process to train baseline model and generate graphs

    :param feat: list of columns based on which combination of features will be generated
    :param minute: minute in string format
    :param assumptions: based assumption on dataframe
    :param L1_neurons: Number of Neurons in First layer
    :param L2_neurons: Number of Neurons in Second layer
    :param base_path: base path directory
    :return: generate sub-plots and Excel sheets for each input feature
    '''
    # List of all combinations of Features
    all_feat = feature_combinations(feat)
    base_path, data_folders= merged_files(base_path)

    # Create New folder in visualization to save files
    folder_name = str(L1_neurons) + "_" + str(L2_neurons) + "_" + str(L1_dropout) + "_" + str(L2_dropout)
    make_directory("visualization/", folder_name)
    folder_name = "visualization/" + folder_name + "/"

    for feature in all_feat:
        feature_list = feature
        print("Training Neural Network using ", feature_list, " Features")
        graph_dict = {}
        total_weeks = len(data_folders)
        for folder in data_folders:
            data = clean_aggregate_data(base_path, folder, assumptions)
            data = insert_day(data, DropZeroFootfall=True)
            if outlier_removal is True:
                data = outlier_detection(data)
            total_rows = len(total_rows)
            train, test, test_df = create_train_test(data)
            X_train, y_train, X_test, y_test = xy_split(train, test, feature)
            input, output, mae, mse, r2 = baseline_model(X_train, y_train, X_test, y_test, L1_neurons, L2_neurons, epoch, L1_dropout, L2_dropout)
            col_name = change_col_name(folder, feature)
            test_df['time'] = pd.to_datetime(test_df['time'])
            test_df[col_name] = output
            graph_dict[folder] = {"input":input,
                                "output":output,
                                "test_df":test_df,
                                # 'error (%)': error_perc,
                                'total_weeks': total_weeks,
                                'total_rows' : total_rows,
                                'MAE': mae,
                                'MSE':mse,
                                'r2':r2,
                                "col_name":col_name}

        feature = list(map(lambda st: str.replace(st, "_", ""), feature_list))
        feature = '_'.join([str(elem) for elem in feature])

        # Generate Experiment Excel Sheets and Subplots
        generate_excel(feature, folder_name, graph_dict)
        sub_plots(graph_dict, len(data_folders)+1, minute, data_folders, feature_list, folder_name, L1_neurons, L2_neurons, L1_dropout, L2_dropout, epoch)
        #exit()