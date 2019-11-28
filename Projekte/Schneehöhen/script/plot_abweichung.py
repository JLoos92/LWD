#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:39:42 2019

@author: root
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from abweichung_schneehoehe import abweichung


delta_dead = abweichung()


delta_dead = pd.DataFrame(delta_dead).transpose()
delta_dead.columns = stations
delta_dead_arranged = pd.melt(delta_dead)
delta_dead_arranged.columns = ['Station', 'Schneehöhe']





fig,ax = plt.subplots(figsize = (16,9),dpi =80)


sns.boxplot(x='Station', y='Schneehöhe', data=delta_dead_arranged)
ax.axhline(linewidth=2, color='r',linestyle = 'dotted')

ax.set_xticklabels(stations,rotation=90)

ax.set_xlabel('Stationen')
ax.set_ylabel('Abweichung von der mittleren Schneehöhe')

# Save figures
path = str('figures/')
fname_pdf = str('Boxplot' + '.pdf')
fname_png = str('Boxplot' + '.png')    
fig.savefig(path + fname_pdf, format = 'pdf',dpi=1000,bbox_inches = 'tight') 
fig.savefig(path + fname_png, format = 'png',dpi=1000,bbox_inches = 'tight') 