from datetime import date
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

class Mergedata:
    def __init__(self,week,dest_dir,dir_path,mac_file,path_sample,time_limit,rssi_val,figname):
        self.week = week
        self.dest_dir = dest_dir
        self.dir_path = dir_path
        self.mac_file = mac_file
        self.path_sample = path_sample
        self.time_limit = time_limit
        self.rssi_val = rssi_val

    def _merge_camera_files(self, files):
        dfs = []
        for file in files:
            file = self.dir_path + "/" +self.week + "/" + file
            dfs.append(pd.read_csv(file))        
        df = pd.concat(dfs)
        df = df.reset_index(drop=True)
        # df = pd.to_datetime(df['time'])
        return df

    def _clean_camera(self, df, time_limit):
        df['time'] = pd.to_datetime(df['time'])
        df['time'] = df['time'].dt.strftime("%Y-%m-%d %H:%M:00")
        df['time'] = pd.to_datetime(df['time'])
        df.index = df['time']
        df_resampled = df.resample(time_limit).sum().dropna() # [(df['total']<300)&(df['total']>0)]
        df_resampled = df_resampled.reset_index()
        return df_resampled

    def _visualize(self, entry_df, exit_df):
        # Total Calculation for entry and exit dataset        
        exit_df = exit_df.rename(columns = {'total':'total_exit'})
        entry_df = entry_df.rename(columns = {'total': 'total_entry'})
        merged_data = pd.merge(entry_df,exit_df,left_on="time",right_on="time",how='inner')
        print(merged_data)
        merged_data['time'] = merged_data['time'].dt.date
        
        # plt.figure(figsize = (10,7))
        merged_data.plot(x="time", y=["total_exit", "total_entry"], kind="bar", figsize=(16, 3))        
        plt.xticks(rotation=0)
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.savefig(figname, pad_inches=0.11, bbox_inches='tight', dpi=600)
        plt.close() 
    
    def _resample_wifi(self, wifi):
        wifi.index = wifi['node_time']

        wifi_res_count = wifi[['node_time','device_mac','randomised','is_vendor']].drop_duplicates()\
                            .resample(self.time_limit).sum().reset_index().\
                            rename(columns={'randomised':'count_randomised','is_vendor':'count_vendor'})\
            [['count_randomised','count_vendor']]

        wifi_res_perc = wifi[['node_time','device_mac','randomised','is_vendor']].drop_duplicates()\
                        .resample(self.time_limit).mean().reset_index().\
                        rename(columns={'randomised':'perc_randomised','is_vendor':'perc_vendor'})

        apple_count = wifi[['node_time','device_mac','randomised','is_vendor','apple']].drop_duplicates()\
                            .resample(self.time_limit).sum().reset_index()[['apple']]

        not_apple_count = wifi[['node_time', 'device_mac', 'randomised','is_vendor','not_apple']].drop_duplicates()\
            .resample(self.time_limit).sum().reset_index()[['not_apple']]

        wifi_resampled = pd.concat([wifi_res_perc,wifi_res_count,apple_count, not_apple_count],axis=1)
        return wifi_resampled

    def _clean_wifi(self, wifi):
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

        wifi = wifi[wifi['companyName']!='Tiger NetCom']

        wifi = wifi[wifi['rssi']>self.rssi_val] # original >-67        

        wifi.to_csv('visualization/wifi'+week+".csv", index=False)
        self.visualize_bins(wifi.copy())

        return wifi

    def _get_files(self):
        all_files = os.listdir(self.dir_path + "/" +self.week)
   
        entry_camera_files = [x for x in all_files if 'entre' in x and x.endswith("csv")]
        exit_camera_files = [x for x in all_files if 'exit' in x and x.endswith("csv")]
        entry_camera_files.sort()
        exit_camera_files.sort()
        exit_df = self._merge_camera_files(exit_camera_files)        
        entry_df = self._merge_camera_files(entry_camera_files)        

        exit_df['total'] = exit_df['incoming']-exit_df['outcoming']
        exit_df['total'] = exit_df['total'].abs()
        entry_df['total']=entry_df['incoming']-entry_df['outcoming']
        entry_df['total'] = entry_df['total'].abs()

        # Clean Camera files
        entry_df = self._clean_camera(entry_df,self.time_limit)
        exit_df = self._clean_camera(exit_df,self.time_limit)
        
        # visualize entry and exit datasets
        # self._visualize(entry_df, exit_df)
        
        return entry_df, all_files[0]

    def _generate_company_names(self, file_name):
        pattern = re.compile(r".+_camera_(.+)_\d+-\d+\.csv")    
        return pattern.match(file_name).group(1)
    
    def getuniquemac(self, wifi_n):
        # wifi_n = wifi_n.resample(self.time_limit).sum().dropna()
        wifi_n['node_time'] = wifi_n['node_time'].dt.date
        wifi_n = wifi_n[wifi_n['randomised']==0]
        wifi_n = wifi_n.sort_values(by='node_time')
        wifi_group = wifi_n.groupby('node_time')
        df = {str(time):len(df['device_mac'].unique()) for time, df in wifi_group}

        return df.values()
    
    def visualize_bins(self, wifi_n):
        # Visualize missing timestamps 
        # wifi_n['node_time'] = pd.to_datetime(wifi_n['node_time'])   
        # wifi_n['date'] = wifi_n['node_time'].dt.date
        # wifi_n['time'] = wifi_n['node_time'].dt.time
        # wifi_n['hour'] = wifi_n['node_time'].dt.hour
        # wifi_n['minute'] = wifi_n['node_time'].dt.minute
        # groups = wifi_n.groupby('date')
        # for date, df in groups:
        #     print("Date: ", date)
        #     g = df.groupby(['hour','minute'])
        #     cnt = g.device_mac.nunique()
        #     # pivot = g.pivot(index='hour', columns='minute', values='device_mac')
        #     # print(pivot)
        #     breakpoint()

        # Company percentage visualization
        # print("Week: ", week)
        # df = wifi_n['companyName'].dropna().value_counts() #  normalize=True .mul(100).sort_values()
        # print(df)
        # ax = df.plot(kind="barh", figsize=(3, 16))
        # from decimal import Decimal
        # for p in ax.patches:
        #     ax.annotate('{:.2}'.format(Decimal(str(p.get_height()))), (p.get_x(), p.get_height()))
        # plt.savefig("visualization/marsta_oob/"+week+'companynames_RSSI(' + str(self.rssi_val) + ') BIN.png', pad_inches=0.11, bbox_inches='tight', dpi=600)
        # plt.close()

        # RSSI bins visualization 
        # df = pd.cut(wifi_n['rssi'], 10)
        # df = df.value_counts(sort=False, normalize=True).mul(100)
        # ax = df.plot.bar(rot=0, color="b", figsize=(16,4))        
        # plt.savefig("visualization/marsta_oob/"+week+'_RSSI(' + str(self.rssi_val) + ') BIN.png', pad_inches=0.11, bbox_inches='tight', dpi=600)
        # plt.close()
        pass

    def process(self):  
        try:            
            camera, camera_name = self._get_files()   
            name = self._generate_company_names(camera_name)     
            wifi = pd.read_csv(self.dir_path + "/" + self.week + "/" + self.path_sample)
            wifi = self._clean_wifi(wifi) 
            # breakpoint()
            
            unqmac_count = self.getuniquemac(wifi.copy())
            
            wifi = self._resample_wifi(wifi)
            
            cols_drop = ['node_time','incoming','outcoming']
            merged_data = pd.merge(camera,wifi,left_on="time",right_on="node_time",how='inner').drop(cols_drop,axis=1)
            
            # merged_data['unq_vendor_mac'] = unqmac_count
            print(merged_data)
            
            if self.dest_dir is not None:
                folder = os.path.join(self.dest_dir, name)
            else:
                folder = name
            if not os.path.isdir('data/'+name):                
                os.mkdir('data/'+name)
            if not os.path.isdir('data/'+name+"/"+name+"_"+time_limit+"_"+str(abs(rssi_val))):
                os.mkdir('data/'+name+"/"+name+"_"+time_limit+"_"+str(abs(rssi_val)))
            if not os.path.isdir('data/'+name+"/"+name+"_"+time_limit+"_"+str(abs(rssi_val))+"/"+ self.week):
                os.mkdir('data/'+name+"/"+name+"_"+time_limit+"_"+str(abs(rssi_val))+"/"+ self.week)
            # merged_data.to_csv(os.path.join(folder,f"{name}_preprocessed.csv"),index=False)
            merged_data.to_csv(os.path.join('data',name,name+"_"+time_limit+"_"+str(abs(rssi_val)), self.week,f"{ self.week}_preprocessed.csv"),index=False)
            
            # merged_data.plot(x="time", y=["total", "count_randomised", "count_vendor", "unq_vendor_mac"], kind="bar", figsize=(16, 3))      
            # plt.legend(['Camera Count', 'Randomised Devices (count)', 'Vendor Identified (count)', 'Unique Devices (Identified)'])
            # merged_data['time'] = merged_data['time'].dt.date            
            # plt.xticks(rotation=0)
            # plt.xlabel("Date")
            # plt.ylabel("Count")
            # plt.savefig("visualization/marsta_oob/"+week+'_RSSI(' + str(rssi_val) + ') Count.png', pad_inches=0.11, bbox_inches='tight', dpi=600)
            # plt.close() 
            
        except Exception as e:
                print(e, " EXCEPTION RAISED")
                pass


if __name__ == '__main__':
    week_data = [
                '211122-211128',
                '211129-211205'                
                ]

    # week = week_data[0]
    for week in week_data:
        dir_path = r'new_store_data'    
        mac_file = r"mac_folder/macaddress.io-db.csv"
        dest_dir = r"bumbee_week_data/" + week
        time_limit = '60min'
        rssi_val = -67      
        path_sample = r'samples_marsta_oob_'+ week + '.csv'
        figname = 'visualization/marsta_oob/marsta_oob' + week + '_' + time_limit + '_RSSI(' + str(rssi_val) + ').png'
        p = Mergedata(week, dest_dir, dir_path, mac_file, path_sample, time_limit, rssi_val, figname)
        p.process()