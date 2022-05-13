import numpy as np
import pandas as pd
from nn_merged_util import run_process, make_directory

if __name__ == '__main__':
    '''
    Trainig Status: 
        1. Using StandardScaler Normalization
            trained using 100 epochs 6, 12, 32, 64 for 0.01 rate
            trained using 100 epochs 6, 12, 32, 64 for 0.001 rate
            trained using 100 epochs 6, 12, 32, 64 for 0.0001 rate
        

    '''
    
    # Set results output path
    date = '15_11_2021'
    make_directory('results/', date)
    output_path = "results/" + date + "/"

    make_directory(output_path, "AllResults")
    # make_directory(output_path, "Summary")

    # Experiment control variables
    base_path = 'data'
    time = '24H'
    all_feat = ['count_randomised', 'count_vendor', 'hour', 'day']
    L1_neurons = 30
    L2_neurons = 20
    epoch = 200 # 10, 50, 100 200
    batch_size= 32 # 6, 12, 32, 64 
    learning_rate = 0.001 # 0.01, 0.001, 0.0001,  
    normal_type = 'StandardScaler' # StandardScaler, MinMaxScaler, RobustScaler or set to false if no use of normalization
    outlier = True
    
    no_of_training_rows = 100 # set the number of rows we want in test data
    run = 1

    # make_directory(output_path+"Summary/", "L"+str(learning_rate))
    
    run_process(base_path, time, all_feat, L1_neurons, L2_neurons, epoch, batch_size, learning_rate, 
    normal_type, outlier, output_path, no_of_training_rows, run)  