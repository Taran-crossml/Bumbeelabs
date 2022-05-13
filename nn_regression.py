import pandas as pd
from nn_utils import process

if __name__ == '__main__':
    minute = '30min'
    base_path = r"inferno_odenplan_allweeks/inferno_odenplan_" + minute
    assumptions = {'working_hours': {'start': 7, 'end': 23}}
    # Feature list
    #feat = ['perc_randomised', 'perc_vendor', 'day', 'count_vendor', 'count_randomised']
    feat = ['count_vendor', 'count_randomised']

    # Experiment Setting Excel Sheet
    experiment_setting_df = pd.read_excel('nn_experiment_settings.xlsx')

    for index, row in experiment_setting_df.iterrows():
        process(base_path,
                feat,
                minute,
                assumptions,
                row['L1_neurons'],
                row['L2_neurons'],
                row['L1_dropout'],
                row['L2_dropout'],
                epoch = 100,
                outlier_removal=False)
        exit()