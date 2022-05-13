import pandas as pd

def read_weather():
    '''
    Read weather csv file
    '''
    weather_df = pd.read_csv('weather_store_data/weather-2021.csv')
    return weather_df

def read_storedata():
    '''
    Read store envirnment sheet
    '''
    store_df = pd.read_excel('weather_store_data/environment_sheet.xlsx')
    store_df = store_df[['locations','city']]
    return store_df

def merge_weather(store_name, df):
    '''
    Merge weather data with store data using time key
    '''
    weather_df = read_weather()
    store_df = read_storedata()
    store_dict = store_df.set_index('locations').to_dict()['city']
    weather_df['time'] = pd.to_datetime(weather_df['time'])
    weather_df = weather_df.sort_values(by = 'time')
    weather_df['time'] = weather_df['time'].dt.date
    weather_df = weather_df[weather_df['city']==store_dict[store_name]]
    weather_df1 = weather_df[['time', 'temperature', 'city']]

    # Calculate average weather
    weather_df1 = weather_df1.groupby('time').mean()
    weather_df1['avg_temp'] = weather_df1['temperature'].astype(int)
    weather_df1.drop('temperature', axis='columns', inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].dt.date
    df = pd.merge(df,weather_df1,left_on="time",right_on="time", how='inner')

    # Calculate min temparature
    weather_df2 = weather_df[['time', 'temperature', 'city']]
    min_df =  weather_df2.groupby('time').agg('min')
    min_df = min_df.rename(columns = {'temperature': 'min_temp'}, inplace = False)    
    df = pd.merge(df,min_df,left_on="time",right_on="time", how='inner')
    df.drop('city', axis='columns', inplace=True)

    # Calculate max temparature
    weather_df3 = weather_df[['time', 'temperature', 'city']]
    max_df =  weather_df3.groupby('time').agg('max')
    max_df = max_df.rename(columns = {'temperature': 'max_temp'}, inplace = False)
    df = pd.merge(df,max_df,left_on="time",right_on="time", how='inner')
    df.drop('city', axis='columns', inplace=True)

    return df

