#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:31:24 2019

@author: root
"""
import pandas
import pandas as pd
import numpy as np
import csv
import matplotlib
from pylab import *
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



lw_tote = pd.read_excel('Wetterstationen_Lawinentote_FINAL.xlsx', sheet_name='Wetterstationen_Lawinentote_FIN')


# Sortiert: lawinentote
sorted_lw_tote = lw_tote.sort_values(by=['Unfall_DAT'])

lw_tote['Date'] = pd.to_datetime(lw_tote.Unfall_DAT)
sorted_lw_tote = lw_tote.sort_values(by=['Stat_NAM'])




def abweichung():
       
        # make listdir
        
    list_stations = os.listdir('Stations/.')
    
    stations = []
    stats_abweichung = pd.DataFrame()  
    neuneu = []
 
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
        df_months = mean_days.groupby(mean_days['Date'].dt.strftime('%B'))['HS'].mean()
        mean_days_dates = pd.to_datetime(mean_days['Date']).dt.strftime('%Y-%m-%d')
        # Mean days
        
        mean_m_d = mean_days.groupby(mean_days['Date'].dt.strftime('%m/%d'))['HS'].mean()
        max_m_d = mean_days.groupby(mean_days['Date'].dt.strftime('%m/%d'))['HS'].max()
        max_m_d = max_m_d.reset_index()
        max_m_d = max_m_d.drop(max_m_d.index[59])
        
        min_m_d = mean_days.groupby(mean_days['Date'].dt.strftime('%m/%d'))['HS'].min()
        min_m_d = min_m_d.reset_index()
        min_m_d = min_m_d.drop(max_m_d.index[59])
        
        
        
        
        # Date for x-axis
        x_date = mean_m_d.reset_index()
        x_date['Date'] = x_date['Date'] + '/2018'
        x_date['Date'] = x_date['Date'].astype(str)
        x_date['Date'] = pd.to_datetime(x_date['Date'],errors='coerce')
        x_date['Date'] = pd.to_datetime(x_date['Date'],format='%Y/%m/%d')
        x_date = x_date[pd.notnull(x_date['Date'])]
           
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
        
            
            
        new = dates_tote_day > thresh
         
        new_ind = new.index[new['Unfall_DAT']==True].tolist()
         
         # make new years
        for i in new_ind:
            dates_tote_day.loc[i] = dates_tote_day.loc[i].apply(lambda x: x.replace(year=2017))
             
             
           
           
        
        # hydrological year    
        x_date_new = x_date.tail(92)
        x_date_new = x_date_new['Date'].map(lambda x: x.replace(year=2017))
        x_date_new = x_date_new.append(x_date['Date'].head(181)).to_frame()  
        x_date_new['Date'] = pd.to_datetime(x_date_new['Date']).dt.date
        x_date_new = x_date_new.reset_index()
        x_date_new = x_date_new.drop(columns = ['index'])
        x_date_new = x_date_new['Date'].astype(str)
        
        # hydrological year    m
        x_hs_new = x_date['HS'].tail(92)
        x_hs_new = x_hs_new.append(x_date['HS'].head(181))       
        x_hs_new = x_hs_new.to_frame().reset_index()
        x_hs_new = x_hs_new.drop(columns = ['index'])
        
        array_hs = []
        array_hs_2 = []
        stat_names = []
        
        # Make strings
        dates_tote_day['Unfall_DAT'] = pd.to_datetime(dates_tote_day['Unfall_DAT']).dt.date
        dates_tote_day =  dates_tote_day.astype(str)
        
        dates_tote_2['Unfall_DAT'] = pd.to_datetime(dates_tote_2['Unfall_DAT']).dt.date
        dates_tote_2 =  dates_tote_2.astype(str)
                
        for j,k in zip(range(dates_tote_2.size),range(dates_tote_day.size)):
            
            # avalanche deads snow height
            ind_hs = np.where(dates_tote_day.iloc[k][0]==x_date_new)
            ind_hs = ind_hs[0]
            ind_hs_2 = x_hs_new['HS'][ind_hs]
            ind_hs_2 = ind_hs_2.iloc[0]            
            array_hs_2.append(ind_hs_2)
            
            # mean of annual snow height
            ind = np.where(dates_tote_2.iloc[j][0]==mean_days_dates)
            ind = ind[0]
            ind_2 = mean_days['HS'][ind]
            ind_2 = ind_2.iloc[0]            
            array_hs.append(ind_2)
        
            # calculate difference
            difference = np.array(array_hs_2) - np.array(array_hs)
    
        # append list of lists    
        neuneu.append(difference.tolist())
            
      
       
        max_m_d_new = max_m_d.tail(92)
        max_m_d_new = max_m_d_new.append(max_m_d.head(181)) 
        
        min_m_d_new = min_m_d.tail(92)
        min_m_d_new = min_m_d_new.append(min_m_d.head(181))  
        
        # annotation
        annot_dates = dates_tote.astype(str)
        annot_dates = annot_dates.to_frame()
        annot_dates_list = annot_dates['Unfall_DAT'].tolist()
            
            
    abweichung_posneg = pd.DataFrame(neuneu).transpose()
    abweichung_posneg.columns = stations
       
    return neuneu    
            
