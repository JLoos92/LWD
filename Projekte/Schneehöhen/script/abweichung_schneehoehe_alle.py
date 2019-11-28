#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:03:32 2019

@author: root
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:31:24 2019

@author: root
"""

import pandas as pd
import numpy as np
import csv
import numpy as np
from pandas import DataFrame, Series
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from numpy import genfromtxt
import plotly as py
import sys
import re
import pandas as pd
from datetime import date
import datetime
import locale
import time
from functools import partial
import matplotlib.dates as mdates
import os
import matplotlib.ticker as ticker
from matplotlib.ticker import NullFormatter
from matplotlib.dates import MonthLocator, DateFormatter
from dateutil.relativedelta import relativedelta


lw_tote = pd.read_excel('Wetterstationen_Lawinentote_FINAL.xlsx', sheet_name='Wetterstationen_Lawinentote_FIN')


# Sortiert: lawinentote
sorted_lw_tote = lw_tote.sort_values(by=['Unfall_DAT'])

lw_tote['Date'] = pd.to_datetime(lw_tote.Unfall_DAT)
sorted_lw_tote = lw_tote.sort_values(by=['Stat_NAM'])




def abweichung_alle():
       
        # make listdir
        
    list_stations = os.listdir('Stations/.')
    list_stations = list_stations[:3]
    stations = []  
    logo = []
 
    for i in list_stations:
        count = 0
            
        
        with open('Stations/' + i,'r') as f:
            for line in f:
         
                count += 1
                lines = f.readlines()    
                station_name = str(lines[2])
                station = station_name.split('= ')
                station = station[1].split(':')
                station_name = station[0]
                stations.append(station_name)
            f.close()
                
        
        
        
        # pandas    
        df = pd.read_fwf(r'Stations/' + i, header=None)
        df.columns = ['Data']
        
        # Header
        df_header = df.iloc[:18]        
        
        # Data
        df_data = df.iloc[19:]
        
        df_data['Date'], df_data['HS'] = df_data['Data'].str.split(' ', 1).str
        df_data.replace(-999, np.nan)
        df_data['HS'].astype(float)
        
        df_date = pd.to_datetime(df_data['Date'])
                
        date_new = df_date.dt.date
        time_new = df_date.dt.time
       
        date_new = date_new
        data_new = df_data['HS']
    
        df_all = pd.concat([date_new, data_new.reindex(date_new.index)], axis=1)
        df_all = pd.concat([date_new, data_new], axis=1)
        df_all["HS"] = pd.to_numeric(df_all["HS"])
        df_all = df_all.replace(-999.0,np.NaN)
        df_all = df_all.replace(0.0,np.NaN)
        
        
        mean_days = df_all.groupby(['Date']).mean()
        mean_days = mean_days.reset_index() 
        
        # make mean months
        mean_days['Date'] = pd.to_datetime(mean_days['Date'])
        mean_days_dates = pd.to_datetime(mean_days['Date'])        
        mean_days = mean_days.set_index('Date')
        mean_days = mean_days.reset_index()

        #######################################################################
        # Lawinentote
        #######################################################################
        
        
        date_ind = np.where(station_name == sorted_lw_tote['Stat_NAM'])[0]
        
        dates_tote = sorted_lw_tote['Unfall_DAT'][date_ind]
        dates_tote_2 = pd.to_datetime(dates_tote, format = '%d.%m.%Y')#).dt.strftime('%d.%m.%Y')
        dates_tote_2 = dates_tote_2.to_frame()
       
        dates_tote_day = dates_tote_2['Unfall_DAT'].dt.strftime('%m/%d')
        
        dates_tote_day = dates_tote_day.to_frame()
        
        # Make dates for x-axis datetime: Unfälle
        dates_tote_day['Unfall_DAT'] = dates_tote_day['Unfall_DAT'] + '/2018'
        dates_tote_day['Unfall_DAT'] = dates_tote_day['Unfall_DAT'].astype(str)
        dates_tote_day['Unfall_DAT'] = pd.to_datetime(dates_tote_day['Unfall_DAT'],errors='coerce')
        dates_tote_day['Unfall_DAT'] = pd.to_datetime(dates_tote_day['Unfall_DAT'],format='%Y/%m/%d')        
        thresh = pd.to_datetime('2018-10-01',format='%Y-%m-%d')
        
            
        dates_tote_day['Unfall_DAT'].dt.year   
        
        new = dates_tote_day > thresh
         
        new_ind = new.index[new['Unfall_DAT']==True].tolist()
         
         # make new years
        for i in new_ind:
            dates_tote_day.loc[i] = dates_tote_day.loc[i].apply(lambda x: x.replace(year=2017))
             
        startDate = '2017-10-01'
        endDate = '2018-05-30'
        
        
        mean_days.columns = ['Date', station_name]
        mask = (mean_days['Date'] > startDate) & (mean_days['Date'] <= endDate)
        mean_days_year = mean_days.loc[mask]
        mean_days_year = mean_days_year.set_index('Date')
        logo.append(mean_days_year[station_name])
        print(mean_days_year.size)
      
    
    return logo
