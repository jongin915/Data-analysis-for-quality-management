# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for displaying DataFrames

import matplotlib.pyplot as plt
import seaborn as sns

# Import supplementary visualization code visuals.py from project root folder
import visuals as vs

# Load the Red Wines dataset
data = pd.read_csv("./Data/wine(Range).csv", sep=',')
# Display the first five records
display(data.head(n=5))

n_wines = data.shape[0]

# Number of wines with quality rating 1
data_y_1 = data.loc[(data['y'] == 1)]
n_y_1 = data_y_1.shape[0]

# Number of wines with quality rating 0
data_y_0 = data.loc[(data['y'] == 0)]
n_y_0 = data_y_0.shape[0]


# Percentage of wines with quality rating 1
y_1_ratio = n_y_1*100/n_wines

# Print the results
print("Total number of wine data: {}".format(n_wines))
print("Wines with y=1: {}".format(n_y_1))
print("Wines with y=0: {}".format(n_y_0))
print("Percentage of wines with y=1: {:.2f}%".format(y_1_ratio))

# Some more additional data analysis
display(np.round(data.describe(),2))
display(np.round(data_y_1.describe(),2))
display(np.round(data_y_0.describe(),2))

data.hist(bins=30, color='steelblue', edgecolor='black', linewidth=1.0,
              xlabelsize=8, ylabelsize=8, grid=False)
plt.tight_layout(rect=(0, 0, 2, 2))
plt.savefig("hist of total data.png", dpi=1000, bbox_inches='tight')

data_y_1.hist(bins=30, color='green', edgecolor='black', linewidth=1.0,
              xlabelsize=8, ylabelsize=8, grid=False)
plt.tight_layout(rect=(0, 0, 2, 2))
plt.savefig("hist of y_1.png", dpi=1000, bbox_inches='tight')

data_y_0.hist(bins=30, color='orange', edgecolor='black', linewidth=1.0,
              xlabelsize=8, ylabelsize=8, grid=False)
plt.tight_layout(rect=(0, 0, 2, 2))
plt.savefig("hist of y_0.png", dpi=1000, bbox_inches='tight')

var_list=list(data.head(0))
fig = plt.figure()
for i in range(1,6):
    ax = fig.add_subplot(1,9,2*i-1)
    data.boxplot(column=var_list[i-1],figsize=(2,2),fontsize=10,grid=0)
plt.savefig("boxplot1 of total data.png", dpi=1000, bbox_inches='tight')

fig = plt.figure()
for i in range(1,7):
    ax = fig.add_subplot(1,11,2*i-1)
    data.boxplot(column=var_list[i+4],figsize=(2,2),fontsize=6,grid=0)
plt.savefig("boxplot2 of total data.png", dpi=1000, bbox_inches='tight')

fig = plt.figure()
for i in range(1,6):
    ax = fig.add_subplot(1,9,2*i-1)
    data_y_1.boxplot(column=var_list[i-1],figsize=(2,2),fontsize=10,grid=0)
plt.savefig("boxplot1 of y_1.png", dpi=1000, bbox_inches='tight')

fig = plt.figure()
for i in range(1,7):
    ax = fig.add_subplot(1,11,2*i-1)
    data_y_1.boxplot(column=var_list[i+4],figsize=(2,2),fontsize=6,grid=0)
plt.savefig("boxplot2 of y_1.png", dpi=1000, bbox_inches='tight')

fig = plt.figure()
for i in range(1,6):
    ax = fig.add_subplot(1,9,2*i-1)
    data_y_0.boxplot(column=var_list[i-1],figsize=(2,2),fontsize=10,grid=0)
plt.savefig("boxplot1 of y_0.png", dpi=1000, bbox_inches='tight')


fig = plt.figure()
for i in range(1,7):
    ax = fig.add_subplot(1,11,2*i-1)
    data_y_0.boxplot(column=var_list[i+4],figsize=(2,2),fontsize=6,grid=0)
plt.savefig("boxplot2 of y_0.png", dpi=1000, bbox_inches='tight')

pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (30,30), diagonal = 'kde');
plt.savefig("corrgraph of total data.png", dpi=1000, bbox_inches='tight')

data_y_1_ = data_y_1.drop(columns=['y'])
pd.plotting.scatter_matrix(data_y_1_, alpha = 0.3, figsize = (30,30), diagonal = 'kde');
plt.savefig("corrgraph of y_1.png", dpi=1000, bbox_inches='tight')

data_y_0_ = data_y_0.drop(columns=['y'])
pd.plotting.scatter_matrix(data_y_0_, alpha = 0.3, figsize = (30,30), diagonal = 'kde');
plt.savefig("corrgraph of y_0.png", dpi=1000, bbox_inches='tight')

correlation = data.corr()
#display(correlation)
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
plt.savefig("corrtable of total data.png", dpi=1000, bbox_inches='tight')

correlation_y_1 = data_y_1_.corr()
#display(correlation)
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(correlation_y_1, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
plt.savefig("corrtable of y_1.png", dpi=1000, bbox_inches='tight')

correlation_y_0 = data_y_0_.corr()
#display(correlation)
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(correlation_y_0, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
plt.savefig("corrtable of y_0.png", dpi=1000, bbox_inches='tight')