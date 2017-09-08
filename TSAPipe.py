from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd
import numpy as np

def seas_decomp(data):
    for e in data.columns:
        print e
        try:
            result = seasonal_decompose(data[e], model= 'multiplicative', freq=1)
        except ValueError:
            print('Multiplicative seasonality is not appropriate for zero and negative values, using additive instead')
            result = seasonal_decompose(data[e], model='additive', freq=1)
        except TypeError:
            print('TypeError: ignoring:', e)
        result.plot()
        plt.show()
        station_check(data[e], e)
        autocorr(data[e], e)

def station_check(series, e):
        print('Augmented Dickey-Fuller Test')
        result = adfuller(series.values)
        if result[0] > result[4]['1%'] and result[0] < result[4]['5%']:
            print(e, 'is non-stationary with significance level of more the 1%')
        if result[0] > result[4]['5%'] and result[0] < result[4]['10%']:
            print(e, 'is non-stationary with significance level of more the 5%')
        if result[0] > result[4]['10%']:
            print(e, 'is non-stationary with significance level of more the 10%')
        print('')
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f'% (key, value))
        print('')

def autocorr(series, e):
    print('autocorrelation plot for:', e)
    #plt.figure(figsize=(18, 8))
    autocorrelation_plot(series)
    plt.show()

#def create_cleanl_series(data):
    # -----------------------------------------------------------------------------------------------------
    # alt - es wird der df direkt ausgewertet, kein Umweg mehr ueber series
    # -----------------------------------------------------------------------------------------------------
    #df_ser_cr = data.loc[:, ['cleanlinessraw']]     # cleanliness rawdata
    #df_ser_c = data.loc[:, ['cleanliness']]         # cleanliness fitted
    #df_ser_cc = data.loc[:, ['cleanlinesscorr']]    # fitted corrected cleanliness

    #df_ser_cr = df_ser_cr.set_index(pd.DatetimeIndex(data['datetime']),
    #                                drop=True)  # index has to be of type datetime, not type "index"
    #df_ser_c = df_ser_c.set_index(pd.DatetimeIndex(data['datetime']), drop=True)
    #df_ser_cc = df_ser_cc.set_index(pd.DatetimeIndex(data['datetime']), drop=True)

    #series_cr = df_ser_cr.iloc[:, 0]
    #series_c = df_ser_c.iloc[:, 0]
    #series_cc = df_ser_cc.iloc[:, 0]
    #series = [series_cr, series_c, series_cc]

    #df_cleanl = pd.concat([data['cleanlinessraw'], data['cleanliness'], data['cleanlinesscorr']], axis=1)
    #display(df_cleanl.describe())        # GIVE SUMMARY STATISTICS for 3 cleanliness series
    # -----------------------------------------------------------------------------------------------------
    #return series

def plot_cleanliness(plot_cleanl_data):

    # plot the cleanliness
    plt.figure(figsize=(15, 15))
    # series.plot(style='k.')
    for e in plot_cleanl_data:
        plot_cleanl_data[e].plot()
    plt.legend(loc='best')
    plt.show()

def plot_cleanl_window(plot_cleanl_data, timewindow):   # plots cleanliness for indicated time window
    plt.figure(figsize=(18, 8))

    for e in plot_cleanl_data:
        plot_cleanl_data[e][timewindow].plot()
    plt.legend(loc='best')
    plt.show()
    display(plot_cleanl_data)

def visualizeNaN(data):
    fig = plt.figure(figsize=(17, 2))
    ax = fig.add_subplot(1, 1, 1)

    data.isnull().sum().plot()
    data.isnull().sum().plot(style='k.')
    labels = data.columns.values

    ax.set_xticklabels(labels, rotation=90)
    ax.set_xticklabels(labels, minor=True)

    ax.locator_params(nbins=len(labels), axis='x')

    ax.grid(which='both')
    plt.show()

def vis_feat_distrib(data, subplts, xlimits):
    fig, axes = plt.subplots(subplts[0], subplts[1])
    axes = axes.flatten()
    fig.set_size_inches(18, 55)
    fig.suptitle('Features Distributions')

    for i, col in enumerate(data.columns):
        feature = data[col]
        sns.distplot(feature, label=col, ax=axes[i]).set(xlim=(xlimits[0], xlimits[1]), )
        axes[i].axvline(feature.mean(), linewidth=1)
        axes[i].axvline(feature.median(), linewidth=1, color='r')
