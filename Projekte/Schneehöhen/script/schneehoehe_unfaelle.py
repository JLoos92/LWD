#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:46:07 2019

@author: Schmulius
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


def compute_stat_mean():
     
    # Nicht korrigierte Schneehöhen (Beobachter)
        
    
    # Lechtal
    obs_1 = pd.read_csv('BEOB1.csv',skiprows=16,sep=';',names=['Datum', 'Uhrzeit', 'Wert', 'score'])
    
    # Dolomitenhütte
    obs_2 = pd.read_csv('BEOB2.csv',skiprows=16,sep=';',names=['Datum', 'Uhrzeit', 'Wert', 'score'])
    
    # Kühtai
    obs_3 = pd.read_csv('BEOB3.csv',skiprows=16,sep=';',names=['Datum', 'Uhrzeit', 'Wert', 'score'])
    
    # Nordkette
    obs_4 = pd.read_csv('BEOB4.csv',skiprows=16,sep=';',names=['Datum', 'Uhrzeit', 'Wert', 'score'])
   
    # Obergurgl
    obs_5 = pd.read_csv('BEOB5.csv',skiprows=16,sep=';',names=['Datum', 'Uhrzeit', 'Wert', 'score'])
    
    # St. Veit in Defereggen
    obs_6 = pd.read_csv('BEOB6.csv',skiprows=16,sep=';',names=['Datum', 'Uhrzeit', 'Wert', 'score'])
    
    ######################
    # Define title
    ######################
    title = 'Beobachter Lechtal'    
    
    
    # Choose station  
    obs_1['Date'] = pd.to_datetime(obs_1['Datum'],format="%d.%m.%Y").dt.date
    obs = obs_1
    obs = obs.set_index(['Date'])
    year1_ser = np.arange(1992,2019,1)
    year2_ser = np.arange(1993,2020,1)
    
    year1_str = year1_ser.tolist() 
    year2_str = year2_ser.tolist() 
    
    mean_all = []
    max_all = []
    min_all = []
    
    list_years_nums = np.arange(0,27,1)
    
    dfObj = []
    
    for i,k,j in zip(year1_ser,year2_ser,list_years_nums):
      
        # def years
        y1 = str(i)
        y2 = str(k)
         
        #obs_num = obs_num.to_frame()pd.DatetimeIndex(df.t).normalize()
        
        # Get values for season
        startdate = pd.to_datetime(y1 + "-12-01", format='%Y-%m-%d').date().strftime("%Y-%m-%d")
        startdate = datetime.datetime.strptime(startdate,"%Y-%m-%d").date()
        enddate = pd.to_datetime(y2 + "-05-01",  format='%Y-%m-%d').date().strftime("%Y-%m-%d")
        enddate = datetime.datetime.strptime(enddate,"%Y-%m-%d").date()
        
        
        
        obs_num = obs.loc[startdate:enddate]
        obs_num = obs_num['Wert']
        obs_num = obs_num.to_frame().reset_index()
        
        date = obs_num['Date']
        date = date.to_frame()
        obs_num = obs_num.drop(['Date'], axis=1)
        obs_num.columns = [str(year1_str[j])+ '_' + str(year2_str[j])]
        
        
        
        obs_num = obs_num[obs_num[str(year1_str[j])+ '_' + str(year2_str[j])].astype(str).str.isdigit()]
        obs_num = obs_num[str(year1_str[j])+ '_' + str(year2_str[j])].astype(float)
        

        dfObj.append(obs_num)

    
    # Merge dataframes
    df = pd.concat(dfObj,axis=1)    
    df['mean'] = df.mean(axis=1)
    df['min'] = df[df>0.1].min(axis=1)
    df['max'] = df.max(axis=1)
    
    df = df.iloc[:-1]
        
        


    # Lawinenverunglückte
    
    print (obs.loc[pd.DatetimeIndex([datetime.datetime.strptime('2011-12-18', '%Y-%m-%d').date()])])    
    print (obs.loc[pd.DatetimeIndex([datetime.datetime.strptime('2015-02-13', '%Y-%m-%d').date()])])    
    
    lechtal_1 = datetime.datetime.strptime('2018-12-18', '%Y-%m-%d').date()
    lechtal_1_hs = 57
    abweichung_1 = df.iloc[27]['mean'] - lechtal_1_hs
    
    lechtal_2 = datetime.datetime.strptime('2019-02-13', '%Y-%m-%d').date()
    lechtal_2_hs = 43
    abweichung_2 = df.iloc[74]['mean'] - lechtal_2_hs
    
    
    # Figure and plot
    fig, ax = plt.subplots(1)   
    ax.plot(date['Date'],df['mean'],'r-',linewidth=0.5,alpha=0.7)
    ax.plot(date['Date'],df['max'],'k-',linewidth=0.5)
    ax.plot(date['Date'],df['min'],'k-',linewidth=0.5)
    
    ax.plot(lechtal_1,lechtal_1_hs,'b*',)   
    ax.plot(lechtal_2,lechtal_2_hs,'b*') 
    
    
    ax.grid(True, linestyle = '--')
    ax.set_xlabel('Monat/Tag',fontsize=10)
    ax.set_ylabel('HS [cm]',fontsize=10)
    ax.set_title('Schneehöhe ' + title + ' seit 1992',fontsize=14)
    
    ax.tick_params(axis='both', which='major', labelsize=8,direction='in', length=6, width=3)
    
    
    # Define the date format
    date_form = DateFormatter("%m/%d")
    ax.xaxis.set_major_formatter(date_form)
    ax.legend(['Mittlere Schneehöhe'],loc='upper right',frameon=False) 
    
    
     
    #ax.fill_between(date['Date'], df['min'], df['max'], facecolor='grey', alpha=0.3)
      
       
    
    # Save figures
    path = str('figures/')
    fname_pdf = str('Station' + title + '.pdf')
    fname_png = str('Station' + title + '.png')
    
    fig.savefig(path + fname_pdf, format = 'pdf',dpi=1000,bbox_inches = 'tight') 
    fig.savefig(path + fname_png, format = 'png',dpi=1000,bbox_inches = 'tight') 
    
       
    return abweichung_1, abweichung_2
  

# Lawinentote: korrelation schneehöhe und abweichung vom mittel

lw_tote = pd.read_excel('Wetterstationen_Lawinentote_FINAL.xlsx', sheet_name='Wetterstationen_Lawinentote_FIN')


# Sortiert: lawinentote
sorted_lw_tote = lw_tote.sort_values(by=['Unfall_DAT'])

lw_tote['Date'] = pd.to_datetime(lw_tote.Unfall_DAT)
sorted_lw_tote = lw_tote.sort_values(by=['Stat_NAM'])




def compute_station_corrected():
       
        # make listdir
        
    list_stations = os.listdir('Stations/.')
    
    stations = []
        
    #for i in list_stations:
        
    # for i in list_stations:
    count = 0
    
    
    with open('Stations/SAGA2.smet', 'r') as f:
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
    df = pd.read_fwf(r'Stations/SAGA2.smet', header=None)
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
    dates_tote_2 = pd.to_datetime(dates_tote).dt.strftime('%Y-%m-%d')
    dates_tote_2 = dates_tote_2.to_frame()
   
    
    dates_tote_day = pd.to_datetime(dates_tote).dt.strftime('%m/%d')
    dates_tote_day = dates_tote_day.to_frame()
    
    # Make dates for x-axis datetime: Unfälle
    dates_tote_day['Unfall_DAT'] = dates_tote_day['Unfall_DAT'] + '/2018'
    dates_tote_day['Unfall_DAT'] = dates_tote_day['Unfall_DAT'].astype(str)
    dates_tote_day['Unfall_DAT'] = pd.to_datetime(dates_tote_day['Unfall_DAT'],errors='coerce')
    dates_tote_day['Unfall_DAT'] = pd.to_datetime(dates_tote_day['Unfall_DAT'],format='%Y/%m/%d')
    
    thresh = pd.to_datetime('2018-10-31',format='%Y-%m-%d')
    
        
        
    new = dates_tote_day > thresh
     
    new_ind = new.index[new['Unfall_DAT']==True].tolist()
     
     # make new years
    for i in new_ind:
        dates_tote_day.loc[i] = dates_tote_day.loc[i].apply(lambda x: x.replace(year=2017))
         
         
       
       
    
    array_hs = []
    for j in range(dates_tote_2.size):
        
        ind = np.where(dates_tote_2.iloc[j][0]==mean_days_dates)
        ind = ind[0]
        ind_2 = mean_days['HS'][ind]
        ind_2 = ind_2.iloc[0]
        
        array_hs.append(ind_2)
        
     
#        
    # hydrological year    
    x_date_new = x_date.tail(92)
    x_date_new = x_date_new['Date'].map(lambda x: x.replace(year=2017))
    x_date_new = x_date_new.append(x_date['Date'].head(181))  
    
     # hydrological year    
    x_hs_new = x_date['HS'].tail(92)
    x_hs_new = x_hs_new.append(x_date['HS'].head(181))    
#   
    max_m_d_new = max_m_d.tail(92)
    max_m_d_new = max_m_d_new.append(max_m_d.head(181)) 
#    
    min_m_d_new = min_m_d.tail(92)
    min_m_d_new = min_m_d_new.append(min_m_d.head(181))  
#    
    # annotation
    annot_dates = dates_tote.astype(str)
    annot_dates = annot_dates.to_frame()
    annot_dates_list = annot_dates['Unfall_DAT'].tolist()
    
    # Plots
    fig, ax = plt.subplots(1)   
    ax.plot(x_date_new,x_hs_new,'k-',linewidth=0.7, label = 'Mittlere Schneehöhe')
    ax_2 = ax.plot(x_date_new,max_m_d_new['HS'],'k-',linewidth=0.5,alpha=0.5,label = '_nolegend_')
    ax_3 = ax.plot(x_date_new,min_m_d_new['HS'],'k-',linewidth=0.5,alpha=0.5,label = '_nolegend_')
    ax.plot(dates_tote_day,array_hs,'b*',linewidth=0.5,alpha=0.9,label = 'Lawinentote')
    
    # fill inbetween
    x_axis = x_date_new.values
    ax.fill_between(x_axis, min_m_d_new['HS'], max_m_d_new['HS'], facecolor='grey', alpha=0.2)
    xlim_1 = pd.to_datetime('2018-10-31',format = '%Y-%m-%d')
    xlim_2 = pd.to_datetime('2018-06-30',format = '%Y-%m-%d')
      
    # labels titles ticks
    ax.set_xlabel('Monat/Tag',fontsize=10)
    ax.set_ylabel('HS [m]',fontsize=10)
    ax.set_title('Schneehöhe ' + station_name,fontsize=14)   
    ax.tick_params(axis='both', which='major', labelsize=8,direction='in', length=6, width=3)
   # ax.set_xlim(xlim_2,xlim_1)
    #ax1.legend(['Mittlere Schneehöhe'],loc='upper right',frameon=False) 
    leg = ax.legend(frameon = True)
    
    
    #################################
#    ax.annotate('Test', (mdates.date2num(dates_tote_day.iloc[2]), array_hs[1]), xytext=(15, 15), 
#            textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
    
    
    #ax.text(dates_tote_day.iloc[2],array_hs[1], 'hallooo')
#    for i,type in enumerate(annot_dates_list):
#        ax.text(dates_tote_day.iloc[i],array_hs[i], type, fontsize=9)
#    
    ax.annotate(annot_dates_list[1],dates_tote_day.iloc[1],array_hs[1])
   
    
    # Define the date format
    date_form = DateFormatter("%m/%d")
    ax.xaxis.set_major_formatter(date_form)
    fig.tight_layout()
      
    # Save figures
    path = str('figures/')
    fname_pdf = str('Station' + station_name + '.pdf')
    fname_png = str('Station' + station_name + '.png')    
    fig.savefig(path + fname_pdf, format = 'pdf',dpi=1000,bbox_inches = 'tight') 
    fig.savefig(path + fname_png, format = 'png',dpi=1000,bbox_inches = 'tight') 
    

