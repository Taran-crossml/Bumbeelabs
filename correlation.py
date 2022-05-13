import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from utils import is_dir
from tqdm import tqdm
import re
import seaborn as sns
# function to make mean files in output folder min wise
def get_files(time):
    assumptions = {'working_hours': {'start': 7, 'end': 23}}
    data_folders = []
    base_path = r"/home/taran/Machine learning/bumbeelabs/forth/preprocessed files/inferno_taby_preprocessed/inferno_taby_"+time
    for d in os.listdir(base_path):
        if is_dir(os.path.join(base_path, d)):
            data_folders.append(d)
    print(data_folders)
    all_path = []
    for folder in tqdm(data_folders):
        path = os.path.join(base_path, folder, f'{folder}_preprocessed.csv')
        all_path.append(path)

    return all_path

def get_folders(base_path):
    data_folder = []
    for folder in os.listdir(base_path):
        if is_dir(os.path.join(base_path, folder)):
            data_folder.append(folder)
    return data_folder



if __name__ == '__main__':
    base_path = 'preprocessed_files'
    time='30min'
    store_folders = get_folders(base_path)
    print(store_folders)
    for store in store_folders:
        pattern = '(.+)_preprocessed'
        store_name = re.search(pattern, store)
        #print(store_name.group(1))
        file_path = base_path + "/" + store + "/" + store_name.group(1) + "_" + time
        date_folders = get_folders(file_path)
        dfs = [pd.read_csv(os.path.join(file_path, folder, f'{folder}_preprocessed.csv')) for folder in date_folders]
        consolidated = pd.concat([df for df in dfs], axis=0)
        correlations = consolidated.corr()

        coor = sns.heatmap(correlations, annot=True, fmt='.2f', cmap='RdYlGn')
        plt = coor.get_figure()
        plt.savefig("visualization/correlation/" + store_name.group(1) + ".png", pad_inches=0.11, bbox_inches='tight', dpi=300)
        plt.clf()
        print("Store Name: ", store_name.group(1), " Length: ", len(consolidated))

    # all_path = get_files(time)
    # file_counter = 1
    # for path in all_path:
    #     # print("Path: ", path)
    #     df = pd.read_csv(path)
    #     df.reset_index(inplace=True)
    #     # print("df is",df)

        # df.to_csv(r'/home/taran/Machine learning/bumbeelabs/merged1.csv', index=False)
        # file_name = 'output/' + time + '/file_' + str(file_counter) + ".csv"
        # pattern = '/home/taran/Machine learning/bumbeelabs/forth/preprocessed files/inferno_taby_preprocessed/inferno_taby_'+time+'\/(.+)\/(.+)_preprocessed.csv$'
        # text = re.search(pattern, path)
        # date = text.group(1)
        # df.to_csv(file_name, index=False)
        # file_counter+=1


