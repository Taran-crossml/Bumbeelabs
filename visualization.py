from utils import is_dir, get_health_stats, generate_stats, clean_aggregate_data, create_train_test,x_y_split, run_model_pipeline, merge_store_data
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm
import glob
import re
import numpy as np
from scipy.signal import lfilter

def line_chart(pp_df, x, y1, y2, y3, figurename="fig.png"):
    pp_df[x] = pd.to_datetime(pp_df[x])
    pp_df.index = pp_df[x]

    # set plot font
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 5

    # Set plot border line width
    plt.rcParams['axes.linewidth'] = 0.2
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['lines.markersize'] = 2
    fig, ax1 = plt.subplots(figsize=(12, 2))
    width = 0.2

    n = 3 # larger n gives smoother curves
    b = [1.0 / n] * n # numerator coefficients
    a = 1 # denominator coefficient
    # y1 = lfilter(b, a, pp_df[y1])
    # y2 = lfilter(b, a, pp_df[y2])
    # for t,a,b in list(zip(pp_df[x].tolist(),y1,y2)):
    #     print(t, " ", a, " ", b)

    ax1.plot(pp_df[x], pp_df[y1], linewidth=0.2, color='#9d0208', marker="2")
    
    if y2 is not None:
        ax1.plot(pp_df[x], pp_df[y2], linewidth=0.2, color='#3a0ca3', marker='*')
    
    if y3 is not None:
        colors = ['#7400b8', '#06d6a0', '#ef476f']
        for i in range(0,len(y3)):
            ax1.plot(pp_df[x], pp_df[y3[i]], linewidth=0.2, color=colors[i], marker='o')
    
    if y2 is not None:
        col_names = ["Camera Total", "Predicted Count"]
    else:
        col_names = ["Camera Total", "Predicted Count"] + y3
    
    ax1.legend(col_names, ncol=len(col_names), frameon=False) #figurename

    # set labels
    fonts = {"fontname": "DejaVu Sans", "fontsize": 5, "fontweight": 'bold'}
    xlabel = 'Date'
    ylabel = 'Device Count'
    ax1.set_xlabel(xlabel, fontdict=fonts)
    ax1.set_ylabel(ylabel, fontdict=fonts)

    # set border line color
    border_color = 'black'
    ax1.spines["top"].set_color(border_color)
    ax1.spines["left"].set_color(border_color)
    ax1.spines["right"].set_color(border_color)
    ax1.spines["bottom"].set_color(border_color)

    # Set tick parameters and Make y axis tics not visible
    plt.tick_params(axis='x', direction='in', length=2, color="black")
    plt.tick_params(axis='y', direction='in', length=2, color="black")
    plt.savefig(figurename, pad_inches=0.11, bbox_inches='tight', dpi=300)
    plt.close()

def prediction_line_chart(pp_df, time_folder, store_name, column_name):
    pp_df['time'] = pd.to_datetime(pp_df['time'])
    pp_df.index = pp_df['time']

    # set plot font
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11

    # Set plot border line width
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.color'] = 'black'
    fig, ax1 = plt.subplots(figsize=(10, 2))
    width = 0.2

    #pp_df["total"] = 2 * (pp_df["total"] - pp_df["total"].min()) / (pp_df["total"].max() - pp_df["total"].min())
    #pp_df[column_name] = 2 * (pp_df[column_name] - pp_df[column_name].min()) / (pp_df[column_name].max() - pp_df[column_name].min())

    #pp_df["total"] = 2 * (pp_df["total"] - pp_df["total"].mean()) / (pp_df["total"].max() - pp_df["total"].min())
    #pp_df[column_name] = 2 * (pp_df[column_name] - pp_df[column_name].mean()) / (pp_df[column_name].max() - pp_df[column_name].min())
    # window = 11
    # order = 2
    # y_total = savgol_filter(pp_df.total, window, order)
    # y_colname = savgol_filter(pp_df[column_name], window, order)

    # poly = np.polyfit(pp_df["time"], pp_df.total, 5)
    # y_total = np.poly1d(poly)(pp_df.total)
    # poly = np.polyfit(pp_df["time"], pp_df[column_name], 5)
    # y_colname = np.poly1d(poly)(pp_df[column_name])

    n = 3 # larger n gives smoother curves
    b = [1.0 / n] * n # numerator coefficients
    a = 1 # denominator coefficient
    y_total = lfilter(b, a, pp_df.total)
    y_colname = lfilter(b, a, pp_df[column_name])


    ax1.plot(pp_df["time"], y_total, linewidth=1, color='#e9ecef')
    ax1.plot(pp_df["time"], y_colname, linewidth=1, color='#0a9396')
    #ax1.plot(pp_df["time"], pp_df.total_perc_randomised_hour, linewidth=0.8, color='#ee9b00')
    #ax1.plot(pp_df["time"], pp_df.total_hour, linewidth=0.8, color='#3a86ff')
    #ax1.legend(['Person Present',column_name], ncol=1, frameon=False) #figurename
    ax1.legend(['Camera Count','Predicted Count'], loc='center', bbox_to_anchor=(0.0, 1.00, 1, 0.1), ncol=2, frameon=False)
    # set labels
    fonts = {"fontname": "Times New Roman", "fontsize": 12, "fontweight": 'bold'}
    xlabel = 'Date'
    ylabel = 'Device Count'
    ax1.set_xlabel(xlabel, fontdict=fonts)
    ax1.set_ylabel(ylabel, fontdict=fonts)

    # set border line color
    border_color = 'black'
    ax1.spines["top"].set_color(border_color)
    ax1.spines["left"].set_color(border_color)
    ax1.spines["right"].set_color(border_color)
    ax1.spines["bottom"].set_color(border_color)

    # Set tick parameters and Make y axis tics not visible
    plt.tick_params(axis='x', direction='in', length=3, color="black")
    plt.tick_params(axis='y', direction='in', length=3, color="black")
    figurename = time_folder + "/" + column_name +"(" + store_name + ")" + ".png"
    plt.savefig(figurename, pad_inches=0.11, bbox_inches='tight', dpi=600)
    plt.close()

def bar_chart(pp_df, figurename):
    pp_df['Date'] = pd.to_datetime(pp_df.time)
    pp_df['Date'] = pp_df['Date'].dt.date

    x = pp_df.groupby(['Date']).sum()
    pp_df = x.reset_index()
    colors = ['#1d3557', '#ffb703']
    # set plot font
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11

    # Set plot border line width
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.color'] = 'black'

    fig, ax1 = plt.subplots(figsize=(13, 3))
    figsize = (10, 3)
    width = 0.2
    pp_df.plot(x="Date", y=["apple", "not_apple"], kind="bar", figsize=figsize, color=colors)
    # ax1.plot(pp_df["time"], pp_df.not_apple, linewidth=0.5, color='#3a0ca3')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # set labels
    fonts = {"fontname": "Times New Roman", "fontsize": 12, "fontweight": 'bold'}
    plt.xlabel("Data")
    plt.ylabel("Device Count")

    # set border line color
    border_color = 'black'
    ax1.spines["top"].set_color(border_color)
    ax1.spines["left"].set_color(border_color)
    ax1.spines["right"].set_color(border_color)
    ax1.spines["bottom"].set_color(border_color)

    # Set tick parameters and Make y axis tics not visible
    plt.tick_params(axis='x', direction='in', length=3, color="w")
    # plt.xticks(wrap=True)
    plt.tick_params(axis='y', direction='in', length=3, color="black")
    plt.savefig(figurename, pad_inches=0.11, bbox_inches='tight', dpi=300)
    plt.close()

def file_path():
    data_folders = []
    base_path = r"merged (210813-210829 and 210830-210905)_5min"
    for d in os.listdir(base_path):
        if is_dir(os.path.join(base_path, d)):
            data_folders.append(d)
    print(data_folders)
    paths = {}
    for folder in tqdm(data_folders):
        path = os.path.join(base_path, folder, f'{folder}_preprocessed.csv')
        paths[folder]= path
    return paths

if __name__ == '__main__':
    time_list = ['60min','2min','5min']
    for time_folder in time_list:
        file_list = glob.glob(time_folder+"/*.xlsx")
        print(file_list)
        for file in file_list:
            print("file: ", file)
            pattern = r'.+\\(.+).xlsx'
            store_name = re.search(pattern, file, re.IGNORECASE)
            store_name = store_name.group(1)
            df = pd.read_excel(file, parse_dates=[0])
            #df = df.set_index('time')
            #print(df.columns)
            #print(df.dtypes)
            #print(df)
            #df = df.rolling(window=10).mean()
            print(df)
            column_names = df.columns[5:]
            print(column_names)
            for column_name in column_names:
                prediction_line_chart(df, time_folder, store_name, column_name)

    # files = file_path()
    # for file, path in files.items():
    #     df = pd.read_csv(path)
    #     figname = 'visualization\\210813-210829 and 210830-210905\\5min\\' + file + ".png"
    #     line_chart(df, figurename=figname)
    #     #bar_chart(df, figurename=figname)




    # preprocessed_file = r'210601-210606/inferno_odenplan/inferno_odenplan_preprocessed.csv'
    # pp_df = pd.read_csv(preprocessed_file)


     # fig, ax1 = plt.subplots(figsize=(10, 10))
     # tidy = pp_df.melt(id_vars=pp_df.index.hour)
     # sns.barplot(x=pp_df.index.hour, y=pp_df.apple, hue='Variable', data=tidy, ax=ax1)
     # sns.despine(fig)
     #sns.barplot(x=pp_df.index.hour, y=pp_df.apple)
     # ax1.xaxis.set(
     #     major_locator=mdates.DayLocator(),
     #     major_formatter=mdates.DateFormatter("\n\n%A"),
     #     #minor_locator=mdates.HourLocator((0, 2)),
     #     minor_formatter=mdates.DateFormatter("%H"),
     # )
     # # Only Date
     # pp_df['date'] = pd.to_datetime(pp_df['time'])
     # pp_df['date'] = pp_df['date'].dt.strftime("%Y-%m-%d 00:00:00")
     # pp_df['date'] = pd.to_datetime(pp_df['date'])
     #
     # # Only Time
     # pp_df['hour'] = pd.to_datetime(pp_df['time'])
     # pp_df['hour'] = pp_df['hour'].dt.strftime("%H:00:00")
     # pp_df['hour'] = pd.to_datetime(pp_df['hour'])
     #print(pp_df[['date','hour']])
