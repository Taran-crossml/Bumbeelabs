from datetime import date
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from visualization import line_chart

class Mergedata:
    def __init__(self,week,dest_dir,dir_path,mac_file,path_sample,path_camera,time_limit,rssi_val,figname):
        self.week = week
        self.dest_dir = dest_dir
        self.dir_path = dir_path
        self.mac_file = mac_file
        self.path_sample = path_sample
        self.path_camera =path_camera
        self.time_limit = time_limit
        self.rssi_val = rssi_val

    def _merge_camera_files(self, files):
            dfs = []
            for file in files:
                file = self.dir_path + "/" +self.week + "/" + file
                dfs.append(pd.read_csv(file))
            if len(dfs) > 1:
                df = pd.concat(dfs)
                df = df.reset_index(drop=True)
            else:
                df = dfs[0]
                df = df.reset_index(drop=True)
            # df = pd.to_datetime(df['time'])
            return df

    def _clean_camera(self, df, time_limit):
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df['time'] = df['time'].dt.strftime("%Y-%m-%d %H:%M:00")
        df['time'] = pd.to_datetime(df['time'])
        df.index = df['time']
        df_resampled = df.resample(time_limit).sum().dropna() # [(df['total']<300)&(df['total']>0)]
        df_resampled = df_resampled.reset_index()
        
        return df_resampled

    def _visualize(self, entry_df, exit_df):
        # Total Calculation for entry and exit dataset        
        exit_df = exit_df.rename(columns = {'camera_count':'total_exit'})
        entry_df = entry_df.rename(columns = {'camera_count': 'total_entry'})
        # print("exit_df",exit_df)
        # print("entry_df",entry_df)
        # breakpoint()
        merged_data = pd.merge(entry_df,exit_df,left_on="time",right_on="time",how='inner')
        merged_data['time'] = merged_data['time'].dt.date
        # print("++__++__++__++",merged_data)
        # plt.figure(figsize = (10,7))
        merged_data.plot(x="time", y=["total_exit", "total_entry"], kind="bar", figsize=(16, 3))        
        plt.xticks(rotation=90)
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.savefig("visualization/ica_maxi_toftanas/enter_exit/ica_maxi_toftanas_"+week+".png", pad_inches=0.11, bbox_inches='tight', dpi=600)
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

        try:
            wifi['node_time'] = pd.to_datetime(wifi['node_time'], utc=True)
        except Exception as e:
            print("Exception raise as ", e)
            # Remove last row where the exception was raised for '20211222-20220123'
            wifi = wifi[:-1]
            wifi['node_time'] = pd.to_datetime(wifi['node_time'], utc=True)

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
        
        # Remove Tiger NetCom deviced from wifi column
        wifi = wifi[wifi['companyName']!='Tiger NetCom']

        wifi = wifi[wifi['rssi']>self.rssi_val] # original >-67        

        wifi.to_csv('visualization/wifi'+week+".csv", index=False)
        # self.visualize_bins(wifi.copy())
        # print(mac_info)
        return wifi

    def _get_files(self):
        all_files = os.listdir(self.dir_path + "/" +self.week)
        # print(all_files)
        single_df = pd.read_csv(self.dir_path + "/" + self.week + "/" + self.path_camera)
        # single_df=single_df.drop(columns="sensor_id")

        single_df['total'] = single_df['incoming']-single_df['outgoing']
        single_df['total'] = single_df['total'].abs()

        # here the sensor id 62 means entrance fron recycle room and 63 means main entrance
        i = single_df[(single_df.sensor_id == 62)].index
        j = single_df[(single_df.sensor_id == 63)].index
 
        main_ent=single_df.drop(i)
        rec_ent=single_df.drop(j)

        main_ent=main_ent.drop(columns="sensor_id")
        rec_ent=rec_ent.drop(columns="sensor_id")

        main_ent=self._clean_camera(main_ent,self.time_limit)
        rec_ent = self._clean_camera(rec_ent, self.time_limit)

        # adding the total column after saparating in entry and exit data and subsidiry is the recycle entrance 

        main_ent=main_ent.rename(columns = {'total':'main_entrance'}, inplace = False)
        rec_ent=rec_ent.rename(columns = {'total':'subsidiary_entrance'}, inplace = False)
        main_ent['subsidiary_entrance']=rec_ent['subsidiary_entrance']
        main_ent["combined_entrance"]=main_ent["main_entrance"]+rec_ent["subsidiary_entrance"]
        # main_ent=main_ent.drop(["incoming","outgoing"], axis=1)
        print(main_ent)
        # visualize entry and exit datasets
        # self._visualize(main_ent, rec_ent)
        return main_ent, all_files[1]

        # return single_df, all_files[0]

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
        wifi_n['node_time'] = pd.to_datetime(wifi_n['node_time'])   
        wifi_n['date'] = wifi_n['node_time'].dt.date
        wifi_n['time'] = wifi_n['node_time'].dt.time
        wifi_n['hour'] = wifi_n['node_time'].dt.hour
        wifi_n['minute'] = wifi_n['node_time'].dt.minute
        groups = wifi_n.groupby('date')
        for date, df in groups:
            # print("Date: ", date)
            g = df.groupby(['hour','minute'])
            cnt = g.device_mac.nunique()
            # pivot = g.pivot(index='hour', columns='minute', values='device_mac')
            # print(pivot)
            # breakpoint()

        # Company percentage visualization
        print("Week: ", week)
        df = wifi_n['companyName'].dropna().value_counts() #  normalize=True .mul(100).sort_values()
        
        ax = df.plot(kind="barh", figsize=(3, 16))
        from decimal import Decimal
        for p in ax.patches:
            ax.annotate('{:.2}'.format(Decimal(str(p.get_height()))), (p.get_x(), p.get_height()))
        plt.savefig("visualization/ica_maxi_toftanas/"+week+'companynames_RSSI(' + str(self.rssi_val) + ') BIN.png', pad_inches=0.11, bbox_inches='tight', dpi=600)
        plt.close()

        # RSSI bins visualization 
        df = pd.cut(wifi_n['rssi'], 10)
        df = df.value_counts(sort=False, normalize=True).mul(100)
        ax = df.plot.bar(rot=0, color="b", figsize=(16,4))        
        plt.savefig("visualization/ica_maxi_toftanas/"+week+'_RSSI(' + str(self.rssi_val) + ') BIN.png', pad_inches=0.11, bbox_inches='tight', dpi=600)
        plt.close()
        pass

    def process(self):  
        try:            
            camera, camera_name = self._get_files()
               
            name = "ica_maxi_toftanas"
            # print("++",name)
            wifi = pd.read_csv(self.dir_path + "/" + self.week + "/" + self.path_sample)
            
            wifi = self._clean_wifi(wifi) 
            
            # breakpoint()
            
            # unqmac_count = self.getuniquemac(wifi.copy())
            # print("wifi",wifi)
            wifi = self._resample_wifi(wifi)
            # print("wifi",wifi)
            cols_drop = ['node_time','incoming','outgoing']
            merged_data = pd.merge(camera,wifi,left_on="time",right_on="node_time",how='inner').drop(cols_drop,axis=1)
            
            # merged_data['unq_vendor_mac'] = unqmac_count
            
            
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

            merged_data=merged_data.rename(columns = {'combined_entrance':'total'}, inplace = False)
            merged_data.to_csv(os.path.join('data',name,name+"_"+time_limit+"_"+str(abs(rssi_val)), self.week,f"{ self.week}_preprocessed.csv"),index=False)
            print(merged_data)
            print("done")

            # merged_data['date'] = merged_data['time'].dt.date
            # date_grp = merged_data.groupby('date')
            # for date, df in date_grp:
            #     # print(df)
            #     figrname = "visualization/ica_maxi_toftanas/line_chart/"+ str(date) + "(week_" + self.week + ").png"
            #     line_chart(df, 'time', 'total', 'count_vendor', None, ["Camera Total", "Vendor Count"], figurename=figrname)
            #     pass


            # merged_data['date'] = merged_data['time'].dt.date
            # # print ("merged",merged_data)
            # figname = "visualization/ica_maxi_toftanas/line_chart/"+ self.week + ".png"
            # line_chart(merged_data, 'time', 'total', 'count_vendor', None, ["Camera Total", "Vendor Count"], figurename=figname)


            
        except Exception as e:
                print(e, " EXCEPTION RAISED")
                pass


if __name__ == '__main__':
    week_data = [

                # '211222-220123',
                # '220124-220130',     
                # '220131-220206',
                # '220207-220213',
                # '220214-220220',
                # '220221-220227',
                '220228-220306',
                '220307-220313'
                ]

    for week in week_data:
        dir_path = r'new_stores_week_data/ica_maxi_toftanas'
        mac_file = r"mac_folder/macaddress.io-db.csv"
        dest_dir = r"new_stores_week_data/ica_maxi_toftanas/" + week
        time_limit = '1H' # 60 min
        rssi_val = -67      
        path_sample = r'samples_ica_maxi_toftanas_'+ week + '.csv'
        path_camera =r'camera_ica_maxi_toftanas_'+ week + '.csv'
        figname = "visualization/ica_maxi_toftanas/line_chart/" + week + '_' + time_limit + '_RSSI(' + str(rssi_val) + ').png'
        p = Mergedata(week, dest_dir, dir_path, mac_file, path_sample,path_camera, time_limit, rssi_val, figname)
        p.process()