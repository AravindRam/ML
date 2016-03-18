import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar

def plot_heatmap(df,graphNo):

    #Cross-tabulate Category and PdDistrict
    if(graphNo == 1):
        df_crosstab = pd.crosstab(df.PdDistrict,df.Category,margins=True)
    elif(graphNo == 2):
        df_crosstab = pd.crosstab(df.Category,df.Month,margins=True)
    elif(graphNo == 3):
        df_crosstab = pd.crosstab(df.PdDistrict,df.Year,margins=True)
    elif(graphNo == 4):
        df_crosstab = pd.crosstab(df.PdDistrict,df.Month,margins=True)
    del df_crosstab['All']
    df_crosstab = df_crosstab.ix[:-1]

    column_labels = list(df_crosstab.columns.values)
    row_labels = df_crosstab.index.values.tolist()

    if(graphNo == 2 or graphNo == 4):
        month_names=[]
        for month_number in column_labels:
            month_names.append(calendar.month_abbr[month_number])
        column_labels = month_names

    fig,ax = plt.subplots()
    #Specify color map for each visualization
    if(graphNo == 1):
        heatmap = ax.pcolor(df_crosstab,cmap=plt.cm.Blues)
    elif(graphNo == 2):
        heatmap = ax.pcolor(df_crosstab,cmap=plt.cm.RdPu)
    elif(graphNo == 3):
        heatmap = ax.pcolor(df_crosstab,cmap=plt.cm.PuBuGn)
    elif(graphNo == 4):
        heatmap = ax.pcolor(df_crosstab,cmap=plt.cm.YlOrRd)

    fig = plt.gcf()
    fig.set_size_inches(15,5)

    ax.set_frame_on(False)

    ax.set_yticks(np.arange(df_crosstab.shape[0])+0.5, minor=False)
    ax.set_xticks(np.arange(df_crosstab.shape[1])+0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels(column_labels, minor=False)
    ax.set_yticklabels(row_labels, minor=False)

    if(graphNo == 1):
        plt.xticks(rotation=90)

    ax.grid(False)

    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    plt.show()

df=pd.read_csv('train.csv')


df["Dates"].min()   #start date
df["Dates"].max()   #end date
df["Dates"] = pd.to_datetime(df["Dates"])   #convert col to datetime format
df["Year"],df["Month"] = df['Dates'].apply(lambda x: x.year), df['Dates'].apply(lambda x: x.month)

plot_heatmap(df,1)  #Visualization 1
plot_heatmap(df,2)  #Visualization 2
plot_heatmap(df,3)  #Visualization 3
plot_heatmap(df,4)  #Visualization 4


