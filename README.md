# crossml-research


Preprocessing the data for Inferno stores :
There are two files used for preprocessing the data: pipepline.py and preprocessing.py.

pipeline.py :
this file call the class Preprocess from the python script preprocess.py. The week_data is a list of weeks for which the data is available. The notation
for a week being used is 'YYMMDD'-'YYMMDD' i.e. the start date- end date.
Note: The raw data for a week is need to be stored in the folders with names as  'YYMMDD'-'YYMMDD' for example '220124'-'220130'. This folder must have 4 files namely:
camera file and wifi file for both the Inferno stores.
The next thing in this code is to define the value of wifi strength used under the variable rssi_val and to define the time for which the data is required such '24H' will
provide the daily data and '1H' will provide the Hourly data.
In the last the for loop is preprocessing the data for the weeks being specified in the week_data list. 

preprocess.py:
This script contains the class Preprocess being used in pipeline.py. 
_generate_pairs() : 
                      it combines all the files as camera_file and sample_file from each week for each store.

_generate_company_pairs():
                           it generates the company names for each device.

_clean_camera: 
               this function is used to remove those rows in the data which has camera count either 0 or more than 300 and then resample the data according to the 
               'time_limit' specified.

_clean_wifi(): 
               this function deals with the 'sample_wifi' file. It compares the first 8 digits of device mac with mac_info which contains the addresses for different 
               companies. Then it classifies the devices as count_randomised or count_vendor. Also we classify the devices as 'apple' or 'not_apple'. At the end it extract
               only those devices which have rssi_val more than the specified.
           
           
                
_resample_wifi():
                  Initiall the duplicate rows have been dropped and then the data is resampled according to time_limit specified. At last all columns are concatenated.
companycount() :
                It plots the frequency of each company for each store's data.
               
                  

preprocess(): 
              This function calls all functions defined above. At the begining it generate pairs for camera and wifi data. Then it generates company names. It then add
              a file of the preprocessed data into the path specified in the 'pipeline.py' with the week names as a folder.


Setup
---------------

```
# Setup virtual environment
python3 -m venv venv

# Activate virtual env
./venv/bin/activate

# Install requirements
pip install -r requirements.txt
```