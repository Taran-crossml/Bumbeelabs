from preprocess import Preprocess

week_data = ['210601-210606', 
            '210607-210616', 
            '210617-210627',
            '210627-210706', 
            # '210813-210829', # process being killed for this week
            '210830-210905',
            '210906-210913', 
            # '210914-211005', # process being killed for this week
            '211006-211011',
            '211012-211017', 
            '211018-211024', 
            '211025-211031',
            '211101-211107',
            '211108-211114',
            '211115-211121',
            '211122-211128',
            '211129-211205']

# week_data = [week_data[-1]]       

print("Total Weeks: ", len(week_data), "Last Index: ", len(week_data)-1)
# exit()

if __name__=="__main__":    
    rssi_val = -67
    time_limit = '24H' # '24H'
    # week_data = week_data[6:]
    print("week data: ", week_data)
    for i in range(len(week_data)):        
        week = week_data[i]
        print(i, " Week being processed: ", week)        
        source_dir = r"bumbee_week_data/"+ week
        mac_file = r"mac_folder/macaddress.io-db.csv"
        dest_dir = r"bumbee_week_data/" + week
        p = Preprocess(source_dir=source_dir,mac_file=mac_file,dest_dir=dest_dir)
        p.preprocess(time_limit, week, rssi_val)
