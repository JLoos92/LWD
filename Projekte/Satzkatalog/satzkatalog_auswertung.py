#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:53:07 2019

@author: Schmulius
"""

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
from matplotlib import rc,rcParams





# Auswertung Satzkatalog; Daten einladen
data_sentence = pd.read_excel('Satzkatalog_Rohdaten.xlsx',sheet_name='Sentence IDs')
data_bulletins = pd.read_excel('Satzkatalog_Rohdaten.xlsx',sheet_name='Bulletins')
data_bulletins = data_bulletins.drop_duplicates(subset='BulletinId')

#Read activity highlights
activity_highlights = data_bulletins['AvActivityHighlightIdsDe']
activity_comments = data_bulletins['AvActivityCommentIdsDe']
snowpack_structure = data_bulletins['SnowpackStructureCommentIdsDe']
tendency = data_bulletins['TendencyCommentIdsDe']

#Split first
sentence_id = activity_highlights.str.split('[',expand=True)[0].to_frame()
activity_comments_id = activity_comments.str.split('[',expand=True)[0].to_frame()
snowpack_structure_id = snowpack_structure.str.split('[',expand=True)[0].to_frame()
tendency_id = tendency.str.split('[',expand=True)[0].to_frame()



##Split second and so on
sentence_id_2 = activity_highlights.str.findall(r"\d+").dropna()
activity_comments_id_2 = activity_comments.str.findall(r"\d+").dropna()
snowpack_structure_id_2 = snowpack_structure.str.findall(r"\d+").dropna()
tendency_id_2 = tendency.str.findall(r"\d+").dropna().reset_index()


# Count
list_2 = []
for i in range(tendency_id_2.size):
    new = list(map(int, tendency_id_2.TendencyCommentIdsDe[i]))
    list_2.append(new)

# Flat list
flat_list = []
for sublist in list_2:
    for item in sublist:
        flat_list.append(item)


#Flatlistarray
flatlist_array =  np.asarray(flat_list)
flatlist_array = flatlist_array[(flatlist_array<517)]

df_tend = pd.DataFrame(flatlist_array)
df_tend.columns = ['ID']

# counts
df_tend['freq'] = df_tend.groupby('ID')['ID'].transform('count')
df_tend_counts = df_tend.drop_duplicates().sort_values(by = 'freq')




# To pandas frame
sentence_id.columns = ['ID']
activity_comments_id.columns = ['ID']
snowpack_structure_id.columns = ['ID']
tendency_id.columns = ['ID']


# counts
sentence_id['freq'] = sentence_id.groupby('ID')['ID'].transform('count')
sentence_id_counts = sentence_id.drop_duplicates().sort_values(by = 'freq')

activity_comments_id['freq'] = activity_comments_id.groupby('ID')['ID'].transform('count')
activity_comments_id_counts = activity_comments_id.drop_duplicates().sort_values(by = 'freq')

snowpack_structure_id['freq'] = snowpack_structure_id.groupby('ID')['ID'].transform('count')
snowpack_structure_id_counts = snowpack_structure_id.drop_duplicates().sort_values(by = 'freq')

tendency_id['freq'] = tendency_id.groupby('ID')['ID'].transform('count')
tendency_id_counts = tendency_id.drop_duplicates().sort_values(by = 'freq')


all_counts = sentence_id_counts.append(activity_comments_id_counts,ignore_index=True)
all_counts = all_counts.append(snowpack_structure_id_counts,ignore_index=True)
all_counts = all_counts.append(tendency_id_counts)


# Sum all counts
all_counts = all_counts.groupby('ID').sum()
all_counts = all_counts.sort_values(by='freq')

# Sentence with 15th most occurence
#all_counts_tail = all_counts.tail(15)
all_counts_tail = all_counts
all_counts_tail = all_counts_tail.sort_index(axis=0)
all_counts_tail = all_counts_tail.reset_index()
all_counts_tail['ID'] = all_counts_tail.ID.astype(int)

all_counts_tail = all_counts_tail.sort_values(by='ID')
all_counts_tail = all_counts_tail.set_index('ID')

# Index sentences
index_nums = all_counts_tail.index

# new
sentence_name_id = data_sentence[['sentence_id', 'name']]


# select sentences real
real_sentences_tail = data_sentence[data_sentence['sentence_id'].isin(all_counts_tail.index)]
real_sentences_tail = real_sentences_tail.set_index('sentence_id')
real_sentences_tail = real_sentences_tail['name']


# concat
all_names = pd.concat([all_counts_tail,real_sentences_tail],axis=1)
all_names = all_names.sort_values(by='freq')
all_names = all_names.groupby('name').sum().reset_index()
all_names = all_names.sort_values(by='freq',ascending = False)
#all_names = all_names.set_index('freq')



# Plots
# activate latex text rendering

fig, ax = plt.subplots(figsize=(2,12))

ax.barh(all_names['name'],all_names['freq'], height = 1.2) # Plots
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Counts')
ax.set_ylabel('Sätze')
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_title('Verwendete Sätze (Satzkatalog 18/19)')
ax.set_ylim(0,all_names['name'].size)


plt.show()

path = str('figures/')
fname_pdf = (str('Satz_auswertung') + '.pdf')
fname_png = (str('Satz_auswertung') + '.png')    
fig.savefig(path + fname_pdf, format = 'pdf',dpi=1000,bbox_inches = 'tight') 
fig.savefig(path + fname_png, format = 'png',dpi=1000,bbox_inches = 'tight') 