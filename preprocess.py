import os 
import re
import pandas as pd
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt

class Preprocess():
    def __init__(self,source_dir, mac_file,dest_dir=None):
        self.source_dir = source_dir
        self.mac_file = mac_file
        self.dest_dir = dest_dir
        
    def _generate_pairs(self):
        all_files = os.listdir(self.source_dir)
        camera_files = [x for x in all_files if x.startswith("camera")]
        camera_files.sort() 
        sample_files = [x for x in all_files if x.startswith("samples")]
        sample_files.sort()
        self.file_pairs = list(zip(camera_files,sample_files))
        return self.file_pairs

    def _generate_company_names(self):
        pattern_digits = re.compile(r"_\d+-\d+\.csv")
        names = []
        for pair in self.file_pairs:
            name = pair[0].replace("camera_","")
            name = re.sub(pattern_digits,"",name)
            names.append(name)
        self.names = names
    
    def _clean_camera(self,df, time_limit):
        df['time'] = pd.to_datetime(df['time'])
        df['time'] = df['time'].dt.strftime("%Y-%m-%d %H:%M:00")
        df['time'] = pd.to_datetime(df['time'])
        df.index = df['time']
        df_resampled = df[(df['total']<300)&(df['total']>0)].resample(time_limit).sum().dropna()
        df_resampled = df_resampled.reset_index().drop('Unnamed: 0',axis=1)
        return df_resampled
    

    
    def _clean_wifi(self, wifi, store_name, rssi_val):
        mac_info = pd.read_csv(self.mac_file)
        wifi['node_time'] = pd.to_datetime(wifi['node_time'])
        wifi['node_time'] = wifi['node_time'].dt.strftime("%Y-%m-%d %H:%M:00")
        wifi['node_time'] = pd.to_datetime(wifi['node_time'])        
        
        wifi['oui'] = wifi['device_mac'].str.slice(0,8)
        mac_info['oui'] = mac_info['oui'].str.slice(0,8)
        mac_info['oui'] = mac_info['oui'].str.lower()
        
        wifi= wifi.merge(mac_info, on='oui', how='left')

        wifi['randomised'] = wifi['companyName'].isnull().astype('int')
        wifi['is_vendor'] = 1 - wifi['companyName'].isnull().astype('int')

        wifi.loc[wifi['companyName']=='Apple, Inc', 'apple'] = 1
        wifi.loc[(wifi['companyName'] != 'Apple, Inc') & (~wifi['companyName'].isnull()), 'not_apple'] = 1

        wifi = wifi[wifi['rssi']>rssi_val] # original >-67
        
        return wifi
    
    def _resample_wifi(self,wifi, time_limit):
        wifi.index = wifi['node_time']

        wifi_res_count = wifi[['node_time','device_mac','randomised','is_vendor']].drop_duplicates()\
                            .resample(time_limit).sum().reset_index().\
                            rename(columns={'randomised':'count_randomised','is_vendor':'count_vendor'})\
            [['count_randomised','count_vendor']]

        wifi_res_perc = wifi[['node_time','device_mac','randomised','is_vendor']].drop_duplicates()\
                        .resample(time_limit).mean().reset_index().\
                        rename(columns={'randomised':'perc_randomised','is_vendor':'perc_vendor'})

        apple_count = wifi[['node_time','device_mac','randomised','is_vendor','apple']].drop_duplicates()\
                            .resample(time_limit).sum().reset_index()[['apple']]

        not_apple_count = wifi[['node_time', 'device_mac', 'randomised','is_vendor','not_apple']].drop_duplicates()\
            .resample(time_limit).sum().reset_index()[['not_apple']]

        wifi_resampled = pd.concat([wifi_res_perc,wifi_res_count,apple_count, not_apple_count],axis=1)
        return wifi_resampled

    def companycount(self, name, source_dir, wifi):
        # Company percentage visualization
        df = wifi['companyName'].dropna().value_counts(normalize=True).mul(100) # .sort_values() 
        ax = df.plot(kind="barh", figsize=(3, 16))  
        if not os.path.exists("visualization"): os.makedirs("visualization")
        if not os.path.exists("visualization/companyplot"): os.makedirs("visualization/companyplot")
        if not os.path.exists("visualization/companyplot/"+name): os.makedirs("visualization/companyplot/"+name)
        # if not os.path.exists("visualization/companyplot/"+name+"/"+source_dir): os.makedirs("visualization/companyplot/"+name+"/"+source_dir)
        plt.savefig("visualization/companyplot/"+name+"/"+source_dir+'.png', pad_inches=0.11, bbox_inches='tight', dpi=600)
        plt.close()        

    def preprocess(self, time_limit, source_dir, rssi_val):
        self._generate_pairs()
        self._generate_company_names()
        for name,pair in tqdm(zip(self.names,self.file_pairs),total = len(self.names)):
            try:
                # print("TIME_STAMP: ", time_limit, " STORE_NAME: ", name)
                path_camera = os.path.join(self.source_dir,pair[0])
                path_sample = os.path.join(self.source_dir,pair[1])
                # print("path_camera: ",path_camera)
                # print("path_sample: ",path_sample)
                camera = pd.read_csv(path_camera)
                wifi = pd.read_csv(path_sample)
                camera = self._clean_camera(camera, time_limit)
                wifi = self._clean_wifi(wifi, name, rssi_val)
                self.companycount(name, source_dir, wifi.copy())
                wifi = self._resample_wifi(wifi, time_limit)
                cols_drop = ['node_time','incoming','outgoing','hour','minute','sensor_id']
                merged_data = pd.merge(camera,wifi,left_on="time",right_on="node_time",how='inner').drop(cols_drop,axis=1)
                if self.dest_dir is not None:
                    folder = os.path.join(self.dest_dir,name)
                else:
                    folder = name
                if not os.path.isdir('data/'+name+"/"+name+"_"+time_limit+"_"+str(abs(rssi_val))):
                    os.mkdir('data/'+name+"/"+name+"_"+time_limit+"_"+str(abs(rssi_val)))
                if not os.path.isdir('data/'+name+"/"+name+"_"+time_limit+"_"+str(abs(rssi_val))+"/"+source_dir):
                    os.mkdir('data/'+name+"/"+name+"_"+time_limit+"_"+str(abs(rssi_val))+"/"+source_dir)
                #merged_data.to_csv(os.path.join(folder,f"{name}_preprocessed.csv"),index=False)
                # merged_data.to_csv(os.path.join('data',name,name+"_"+time_limit+"_"+str(abs(rssi_val)),source_dir,f"{source_dir}_preprocessed.csv"),index=False)
            except Exception as e:
                print(e, " EXCEPTION RAISED")
                pass