# coding=utf-8

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import time
import matplotlib.ticker as ticker
import numpy as np
from decimal import Decimal

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline

###################################################################
###################################################################
###################################################################
###################################################################

class plotAnalysis():

###################################################################
###################################################################

    def __init__(self, plotsDir=""):

        sns.set_style("darkgrid")
        colors = ["windows blue", "amber", "greenish", "orange", "sky blue", "greyish", "salmon", "faded green", "lavender", "denim blue", "medium green"]
        # colors = ["windows blue", "aquamarine", "amber", "lightblue", "lavender"]
        custom_palette = sns.xkcd_palette(colors)
        sns.set_palette(custom_palette)
        self.outDir = plotsDir
        
        self.figWidth = 15
        self.figHeight = 8
        self.linewidth = 2

        return

###################################################################
###################################################################

    def plotBarGraph(self, x1, y1, title="", xlabel="", ylabel="", legendLabel1="", outputFileName="", xLabelSize=25, tilt=False, xTickRotation=0, top=20, bottom=False):
        
        fig, ax = plt.subplots(figsize=(self.figWidth, self.figHeight))
            
        ax.set_title(title, fontsize=xLabelSize)
        ax.set_xlabel(xlabel, fontsize=xLabelSize)
        ax.set_ylabel(ylabel, fontsize=xLabelSize)
        
        tempDF = pd.DataFrame({'Country':x1, 'Data':y1})
        tempDF.sort_values("Data", inplace=True, ascending=False)
        tempDF = tempDF.reset_index()
        if bottom==False:
            tempDF = tempDF[:top]
        else:
            tempDF = tempDF[-1*top:]

        ax.bar(tempDF['Country'].values, tempDF['Data'], width=1, label=legendLabel1)
        ax.legend(loc='upper right', prop={'size': xLabelSize-10}, shadow=True, frameon=True)
        ax.tick_params(axis='both', which='major', labelsize=xLabelSize-15)
        if tilt:
            fig.autofmt_xdate(rotation=xTickRotation)
            
        if len(outputFileName) > 0:
            plt.savefig(self.outDir+outputFileName)

        return

###################################################################
###################################################################

    def plotGraph(self, x1, y1, title="", xlabel="", ylabel="", legendLabel1="", outputFileName="", xLabelSize=25, tilt=False, xTickRotation=0, time=True, dateFormat='%Y-%m'):
        
        fig, ax = plt.subplots(figsize=(self.figWidth, self.figHeight))

        if (time==True):
            hfmt = matplotlib.dates.DateFormatter(dateFormat)
            ax.xaxis.set_major_formatter(hfmt)
        
        ax.set_title(title, fontsize=xLabelSize)
        ax.set_xlabel(xlabel, fontsize=xLabelSize)
        ax.set_ylabel(ylabel, fontsize=xLabelSize)
        
        if (time==True):
            ax.plot(x1,y1, label=legendLabel1, lw=self.linewidth)
        else:
            # ax.plot(x1,y1, label=legendLabel1, ls='None', marker='*')
            ax.scatter(x1, y1, label=legendLabel1)


        ax.legend(loc='upper right', prop={'size': xLabelSize-10}, shadow=True, frameon=True)
        ax.tick_params(axis='both', which='major', labelsize=xLabelSize)
        if tilt:
            fig.autofmt_xdate(rotation=xTickRotation)
            
        if len(outputFileName) > 0:
            plt.savefig(self.outDir+outputFileName)

        return

###################################################################
###################################################################

    def plotTargetGraph(self, x1, y1, title="", xlabel="", ylabel="", legendLabel1="", outputFileName="", xLabelSize=25, tilt=False, xTickRotation=0, time=True):
        
        fig, ax = plt.subplots(figsize=(self.figWidth, self.figHeight))
        
        ax.set_title(title, fontsize=xLabelSize)
        ax.set_xlabel(xlabel, fontsize=xLabelSize)
        ax.set_ylabel(ylabel, fontsize=xLabelSize)
        
        if (time==True):
            ax.plot(x1,y1, label=legendLabel1, lw=self.linewidth)
        else:
            ax.scatter(x1, y1, label=legendLabel1)

        ax.legend(loc='upper right', prop={'size': xLabelSize-10}, shadow=True, frameon=True)
        ax.tick_params(axis='both', which='major', labelsize=xLabelSize)
            
        if tilt:
            for tick in ax.get_xticklabels():
                tick.set_rotation(xTickRotation)

        if len(outputFileName) > 0:
            plt.savefig(self.outDir+outputFileName)

        return

###################################################################
###################################################################

    def plotResultGraph(self, x1, y1, title="", xlabel="", ylabel="", legendLabel=[""], outputFileName="", xLabelSize=25, tilt=False, xTickRotation=0):
        
        fig, ax = plt.subplots(figsize=(self.figWidth, self.figHeight))
        
        ax.set_title(title, fontsize=xLabelSize)
        ax.set_xlabel(xlabel, fontsize=xLabelSize)
        ax.set_ylabel(ylabel, fontsize=xLabelSize)

        for i in range(0,len(y1)):
            ax.scatter(x1, y1[i], label=legendLabel[i])

        ax.legend(loc='upper right', prop={'size': xLabelSize-10}, shadow=True, frameon=True)
        ax.tick_params(axis='both', which='major', labelsize=xLabelSize)
            
        if tilt:
            for tick in ax.get_xticklabels():
                tick.set_rotation(xTickRotation)

        if len(outputFileName) > 0:
            plt.savefig(self.outDir+outputFileName)

        return

###################################################################
###################################################################

    def plotParallelCoordinateGraph(self, inDF, title="", xlabel="", ylabel="", outputFileName="", xLabelSize=25, ranking=False):
        
        # fig, ax = plt.subplots(figsize=(self.figWidth, self.figHeight))
        
        # ax3.set_title(title, fontsize=xLabelSize)
        # ax3.set_xlabel(xlabel, fontsize=xLabelSize)
        # ax3.set_ylabel(ylabel, fontsize=xLabelSize)


        # Plot ranking
        plotDF = pd.DataFrame({'Country':inDF['Country']})
        plotDF = plotDF.set_index('Country')

        colNames = ['atmosphereCO2', 'GDP', 'populationTotal', 'populationUrban', 'landForest']

        # Create (X-1) sublots along x axis
        # fig, axes = plt.subplots(1, len(colNames)-1, sharey=False, figsize=(self.figWidth, self.figHeight))

        if ranking==True:
            for i in range(0,len(colNames)):
                # if colNames[i] not in ['CountryType', 'Country']:
                print(colNames[i])
                inDF.sort_values(colNames[i], inplace=True, ascending=False)
                inDF = inDF.reset_index()
                inDF['index'] = inDF.index.values
                inDF = inDF.set_index('Country')
                plotDF[colNames[i]] = inDF['index']
                inDF = inDF.drop(columns=['index']).copy()

        else:
            inDF = inDF.set_index('Country')
            for i in range(0,len(colNames)):
                # if colNames[i] not in ['CountryType', 'Country']:
                plotDF[colNames[i]] = inDF[colNames[i]]

        print(plotDF.head(5))

        # plotDF['Country'] = plotDF.index.values
        # parallel_coordinates(plotDF, class_column='Country', cols=['atmosphereCO2', 'GDP', 'populationTotal', 'populationUrban', 'landForest'], ax=ax)

        # tempDF = pd.DataFrame({'Country':plotDF.index.values, 'CO2':plotDF['atmosphereCO2']})
        # tempDF.sort_values("CO2", inplace=True, ascending=False)
        # # tempDF = tempDF.reset_index()
        # tempDF = tempDF[:5]
        # topCountries = list(tempDF.index.values)
        # # print(tempDF['Country'])
        # print(topCountries)
        # # if bottom==False:
        # #     tempDF = tempDF[:top]
        # # else:
        # #     tempDF = tempDF[-1*top:]

        # bad_df = plotDF.index.isin(topCountries)
        # plotDF = plotDF[~bad_df]
        # plotDF = plotDF.reset_index()
        # plotDF = plotDF.set_index('Country')

        countries = list(plotDF.index.values)

        print("plotDF['atmosphereCO2'].min(): " + str(plotDF['atmosphereCO2'].min()))
        print("plotDF['atmosphereCO2'].idxmin(): " + str(plotDF['atmosphereCO2'].idxmin()))

        data = [
            go.Parcoords(
                line = dict(color = plotDF['atmosphereCO2'],
                        # colorscale = 'Jet',
                        colorscale = 'Viridis',
                        showscale = True,
                        reversescale = True,
                        cmin = plotDF['atmosphereCO2'].idxmin(),
                        cmax = plotDF['atmosphereCO2'].idxmax()),
                dimensions = list([
                    dict(
                        range = [plotDF['atmosphereCO2'].min(),plotDF['atmosphereCO2'].max()],
                        # constraintrange = [100000,150000],
                        label = 'CO2', values = plotDF['atmosphereCO2']
                        ),
                    dict(
                        range = [plotDF['GDP'].min(),plotDF['GDP'].max()],
                        label = 'GDP', values = plotDF['GDP']
                        ),
                    dict(
                        range = [plotDF['populationTotal'].min(),plotDF['populationTotal'].max()],
                        label = 'populationTotal', values = plotDF['populationTotal']
                        ),
                    dict(
                        range = [plotDF['populationUrban'].min(),plotDF['populationUrban'].max()],
                        label = 'populationUrban', values = plotDF['populationUrban']
                        ),
                    dict(
                        range = [plotDF['landForest'].min(),plotDF['landForest'].max()],
                        label = 'landForest', values = plotDF['landForest']
                        )
                    ])
            )
        ]

        # py.iplot(data, filename = 'parcoords-advanced')

        layout = go.Layout(showlegend=True)
        fig = go.Figure(data=data, layout=layout)

        offline.plot(fig, filename = 'allCountries', image='png', image_width=1366,image_height=768, auto_open=True, image_filename='allCountries')

        ###########################################################
        ###########################################################

        # # Get min, max and range for each column
        # # Normalize the data for each column
        # min_max_range = {}
        # for i in range(0,len(colNames)):
        #     # if colNames[i] not in ['CountryType', 'Country']:
        #     min_max_range[colNames[i]] = [plotDF[colNames[i]].min(), plotDF[colNames[i]].max(), np.ptp(plotDF[colNames[i]])]
        #     plotDF[colNames[i]] = np.true_divide(plotDF[colNames[i]] - plotDF[colNames[i]].min(), np.ptp(plotDF[colNames[i]]))

        # x = [i for i, _ in enumerate(colNames)]
        # colours = ['#2e8ad8', '#cd3785', '#c64c00', '#889a00']

        # # Plot each row
        # for i, ax in enumerate(axes):
        #     for idx in countries:
        #         # if idx not in topCountries:
        #         #     country_category = plotDF.loc[idx]
        #         #     ax.plot(x, country_category, label='_nolegend_')    
        #         # country_category = plotDF.loc[idx]
        #         # if idx=='China':
        #         #     ax.plot(x, country_category)#, label=idx)
        #         # else:
        #         #     ax.plot(x, country_category, label='_nolegend_')#, colours[country_category])

        #         # ax.plot(x, country_category, label='_nolegend_')

        #         country_category = plotDF.loc[idx]
        #         ax.plot(x, country_category, label='_nolegend_')    
        #     ax.set_xlim([x[i], x[i+1]])

        # # axes[0].legend(loc='upper left', prop={'size': xLabelSize-10}, shadow=True, frameon=True)

        # # Set the tick positions and labels on y axis for each plot
        # # Tick positions based on normalised data
        # # Tick labels are based on original data
        # def set_ticks_for_axis(dim, ax, ticks):
        #     min_val, max_val, val_range = min_max_range[colNames[dim]]
        #     step = val_range / float(ticks-1)
        #     # tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
        #     tick_labels = [None] * ticks
        #     for i in range(0, ticks):
        #         if (min_val + step * ticks-1) > 1000:
        #             tick_labels[i] = '%.2E' % Decimal(str(round(min_val + step * i, 2)))
        #         else:
        #             tick_labels[i] = round(min_val + step * i, 2)

        #     norm_min = plotDF[colNames[dim]].min()
        #     norm_range = np.ptp(plotDF[colNames[dim]])
        #     norm_step = norm_range / float(ticks-1)
        #     ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
        #     ax.yaxis.set_ticks(ticks)
        #     ax.set_yticklabels(tick_labels)

        # for dim, ax in enumerate(axes):
        #     ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        #     set_ticks_for_axis(dim, ax, ticks=6)
        #     ax.set_xticklabels([colNames[dim]])

        # # # Move the final axis' ticks to the right-hand side
        # ax = plt.twinx(axes[-1])
        # dim = len(axes)
        # ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
        # set_ticks_for_axis(dim, ax, ticks=6)
        # # ax.set_xticklabels([colNames[-2], colNames[-1]], fontsize=xLabelSize-10)
        # ax.set_xticklabels([colNames[-2], colNames[-1]])
        # ax.grid(None)

        # # Remove space between subplots
        # plt.subplots_adjust(wspace=0)

        # # # Add legend to plot
        # # plt.legend(
        # #     [plt.Line2D((0,1),(0,0), color=colours[cat]) for cat in df['mpg'].cat.categories],
        # #     df['mpg'].cat.categories,
        # #     bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)

        # # plt.title(title, fontsize=xLabelSize)
        # ax.set_title(title, fontsize=xLabelSize)
        # # ax.set_xlabel(fontsize=xLabelSize)
        # # ax.set_ylabel(ylabel, fontsize=xLabelSize)

        ###########################################################
        ###########################################################

        if len(outputFileName) > 0:
            plt.savefig(self.outDir+outputFileName)

        return

###################################################################
###################################################################

    def plotPairsDF(self, df, title="", xlabel="", ylabel="", legendLabel1="", outputFileName="", xLabelSize=25):

        # fig, ax = plt.subplots(figsize=(self.figWidth, self.figHeight))

        # ax.set_title(title, fontsize=xLabelSize)
        # ax.set_xlabel(xlabel, fontsize=xLabelSize)
        # ax.set_ylabel(ylabel, fontsize=xLabelSize)
        
        # sns.pairplot(df)
        pd.plotting.scatter_matrix(df, marker='o',
                                 hist_kwds={'bins': 20}, s=60, alpha=.8)

        # ax.legend(loc='upper right', prop={'size': xLabelSize-10}, shadow=True, frameon=True)
        # ax.tick_params(axis='both', which='major', labelsize=xLabelSize)
            
        # if len(outputFileName) > 0:
        #     plt.savefig(self.outDir+outputFileName)

        return

###################################################################
###################################################################

    def plot_corr(self, df, size=10):
        '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

        Input:
            df: pandas DataFrame
            size: vertical and horizontal size of the plot'''

        corr = df.corr()
        fig, ax = plt.subplots(figsize=(size, size))
        ax.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns);
        plt.yticks(range(len(corr.columns)), corr.columns);

        return

###################################################################
###################################################################