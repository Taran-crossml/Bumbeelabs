import pandas as pd
import numpy as np
from visualization import prediction_line_chart, line_chart
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt

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

def get_single_excel(all_test_df, file_name):
    writer = pd.ExcelWriter(file_name, engine="xlsxwriter")
    workbook = writer.book
    
    row = 0    
    for exp_name, exp_df in all_test_df.items():       

        #worksheet = workbook.add_worksheet(feature_name)
        exp_df.to_excel(writer, startrow=row + 3, index=False)
        
        # Create a format to use in the merged range.
        merge_format = workbook.add_format({
            'bold': 1,
            'border': 1,
            'align': 'center',
            'valign': 'vcenter',
            'fg_color': 'yellow'})
        
        worksheet = writer.sheets['Sheet1']
        worksheet.merge_range(row+2, 0, row+2, len(exp_df.columns)-1, exp_name, merge_format)
        row += (len(exp_df) + 4)
    writer.save()    

def bumbee_internal(file):
    df = pd.read_csv(file)
    df['node_time'] = pd.to_datetime(df['node_time'])
    df['node_time'] = df['node_time'].dt.strftime("%Y-%m-%d %H:%M:%S")
    # print(wifi['node_time'])
    df['node_time'] = pd.to_datetime(df['node_time']) + pd.DateOffset(hours=2) # Add 2 hours

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
    #df['hour_min'] = pd.to_datetime(df['hour'])
    #df.to_excel('bumbee_closed_env/device_merged_and_companies.xlsx')
    return df
    # time = '30s'
    # df = resample_wifi(df, time)
    # print(df.columns)
    # apple_name = "bumbee_closed_env/apple_not_apple" + time + ".png"
    # vendor_random_name = "bumbee_closed_env/randomised_vendor" + time + ".png"
    #
    # line_chart(df, "node_time", "apple", "not_apple", figurename=apple_name)
    # line_chart(df, "node_time", "count_vendor", "count_randomised", figurename=vendor_random_name)
    #
    # df.to_excel("bumbee_closed_env/merged_processed.xlsx")
    # df = df[df['rssi'] > -67]

def rssi_analysis(data):
    # convert data into the intervals of 10 based on rssi value
    df['rssi_cat_val'] = pd.cut(data['rssi'], 10).astype(str)
    data['rssi_cat'] = pd.cut(data['rssi'], 10, labels=np.arange(10))
    print(data[['rssi_cat_val', 'rssi_cat']])
     
    # count plot on single categorical variable
    sns.set(rc={'figure.figsize':(4.7,3.27)})
    sns.countplot(y ='rssi_cat_val', data = data)
    #sns.catplot(x='rssi_categories', data=data)
    
    # Show the plot
    plt.savefig('internal_experiment/internal.png', pad_inches=0.11, bbox_inches='tight', dpi=300)
    

def internal_analytics(df, internal_mac_dict, experiment_timings):
    df['node_time'] = pd.to_datetime(df['node_time'])
    df['node_time'] = df['node_time'].dt.strftime("%Y-%m-%d %H:%M:00")
    df['node_time'] = pd.to_datetime(df['node_time'])
    df = df.sort_values(by='node_time')
    df.set_index('node_time', inplace=True)

    time_frame_dict = {}

    for experiment_name, exp_time_dict in experiment_timings.items():
        start_time = experiment_timings[experiment_name]["start_time"]
        end_time = experiment_timings[experiment_name]["end_time"]

        # Renaming Figname
        fig_name = str(experiment_name) + "_" + str(start_time+"_"+end_time) 
        fig_name = fig_name.replace("/", "")
        fig_name = fig_name.replace(":","")
        print("fig_name: ", fig_name)

        # Experiment Settings
        print("Exp Name: ", experiment_name, " Start_name: ", start_time, " End_time: ", end_time)

        # Slice data in given timeframe
        time_df = df.between_time(start_time, end_time)
        time_df['node_time'] = time_df.index
        time_df['rssi_cat_val'] = pd.cut(time_df['rssi'], 10).astype(str)
        time_df['rssi_cat'] = pd.cut(time_df['rssi'], 10, labels=np.arange(10))

        #print("time_df : ", time_df.columns)
        print(
            #"TOTAL ROWS : ", len(time_df),
            "AVG_RSSI : ", time_df['rssi'].mean(),
            "UNIQUE MAC: ", len(time_df['mac'].unique()),
            "APPLE DEVICES: ", time_df['apple'].sum(),
            " NOT_APPLE: ", time_df['not_apple'].sum()
            )

        # Apple and Non-apple count for sliced dataframe
        for make, mac in internal_mac_dict.items():
            mac = mac[:8]
            make_df = time_df[time_df['oui'] == mac]
            # count plot on single categorical variable
            try:
                sns.set(rc={'figure.figsize':(4.7,3.27)})
                sns.countplot(y ='rssi_cat_val', data = make_df)
            
                # Show the plot
                plt.savefig('internal_experiment/'+ fig_name + "_" + mac + "_" + make + '_rssicount.png', pad_inches=0.11, bbox_inches='tight', dpi=300)
                plt.close()
            except:
                pass
            print("MAC: ", mac, " Make: ", make, " devices: ", len(make_df))

                
        
        # select apple dataframes only
        time_df_new = time_df[~time_df['apple'].isnull()]
        # print("APPLE DEVICES STATISTICS")
        print("APPLE DEVICES STATISTICS: "
            "TOTAL ROWS : ", len(time_df_new),
            "AVG_RSSI : ", time_df_new['rssi'].mean(),
            "UNIQUE MAC: ", len(time_df_new['mac'].unique()),
            )

        time_frame_dict[fig_name] = time_df_new

        time = '10s'
        # print("time_df : ", time_df.columns)
        time_df = resample_wifi(time_df, time)
        # print("time_df : ", time_df.columns)

        fig_name = "visualization/" + fig_name + ".png"
        line_chart(time_df, 'node_time', 'apple', 'not_apple', fig_name)

        # Datframe Summary
        print("Total Entries in given Timeframe: ", len(time_df),
              " APPLE: ", time_df['apple'].sum(),
              " NOT_APPLE: ", time_df['not_apple'].sum(),
              " COUNT_RANDOMIZED: ", time_df['count_randomised'].sum(),
              " COUNT_VENDOR: ", time_df['count_vendor'].sum(),
              #" UNIQUE MAC: ", len(time_df['mac'].unique())
              )
        print("\n")
    #print(time_frame_dict)
    # get_single_excel(time_frame_dict, "visualization/internal_summary_ios_15.2021.09.28 (apple_device).xlsx")

if __name__ == '__main__':
    file = 'internal_experiment/survey_samples_2021.09.21.csv'
    print("File Name: ", file.split("/")[1])
    df = bumbee_internal(file)
    #rssi_analysis(df)
    
    internal_mac_dict_21Sep2021 = {'Huawei':'b8:08:d7:a8:52:69',
                         'Samsung':'a8:9f:ba:ab :79:73',
                         'Iphone':'f8:4e:73:a5:56:e8'}
    experiment_timings_21Sep2021 = {'Mobile locked/passive state': {'start_time':'16:52:00', 'end_time':'17:02:00'},
                          'Unlocked/screen on_1': {'start_time': '17:03:00', 'end_time': '17:13:00'},
                          'Mobiles locked/passive state': {'start_time': '17:19:00', 'end_time': '17:29:00'},
                          'Unlocked/screen on_2': {'start_time': '17:30:00', 'end_time': '17:40:00'},
                          'In phonecall (only f8:4e:73:a5:56:e8)': {'start_time': '17:53:00', 'end_time': '18:05:00'}}
    internal_analytics(df, internal_mac_dict_21Sep2021, experiment_timings_21Sep2021)

    file = 'internal_experiment/survey_samples_ios_15.2021.09.28.csv'
    print("File Name: ", file.split("/")[1])
    df = bumbee_internal(file)

    internal_mac_dict_28Sep2021 = {'Iphone': 'f8:4e:73:a5:56:e8'}
    experiment_timings_28Sep2021 = {'In a call': {'start_time': '14:28:00', 'end_time': '14:58:00'},
                                    'Unlocked': {'start_time': '14:59:00', 'end_time': '15:10:00'},
                                    'Locked': {'start_time': '15:11:00', 'end_time': '15:23:00'},
                                    }
    # internal_analytics(df, internal_mac_dict_28Sep2021, experiment_timings_28Sep2021)

