import pandas as pd

def main():
    df = pd.read_csv('val_data/inferno_visits_2021.csv')    
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    df['start_time'] = df['start_time'].dt.strftime("%Y-%m-%d")
    df['end_time'] = df['end_time'].dt.strftime("%Y-%m-%d")

    df = df.sort_values(by='start_time')
    print(df)
    print(df.groupby('start_time', as_index=False).nunique())
    df = df.groupby('start_time', as_index=False)['visit_id'].nunique()
    print(df)

    
    df['val_count_5'] = df['visit_id'] + (5*df['visit_id'])
    df['val_count_8'] = df['visit_id'] + (8*df['visit_id'])
    df['val_count_10'] = df['visit_id'] + (10*df['visit_id'])
    df.to_csv("val_data/val_data.csv",index=False)

main()