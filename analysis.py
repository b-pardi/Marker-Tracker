'''
Author: Brandon Pardi
Created 2/12/2024
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from exceptions import error_popup, warning_popup


def marker_euclidean_distance(p1x, p1y, p2x, p2y):
    """calculates the euclidean distance between 2 points in the 2D space

    Args:
        p1x (int): x coord of point 1
        p1y (int): y coord of point 1
        p2x (int): x coord of point 2
        p2y (int): y coord of point 2

    Returns:
        float: the euclidean distance between the two given points
    """    
    return np.sqrt((p2x - p1x)**2 + (p2y - p1y)**2)

def plot_data(x, y, plot_args):
    """util function to handle plotting and formatting of the plots

    Args:
        x (pd.Dataframe/Series, np.Array, list, etc): _description_
        y (pd.Dataframe/Series, np.Array, list, etc): _description_
        plot_args (dict): dictionary of plot customization arguments

    Returns:
        (plt.figure, plt.axes): the figure and axes objects created and modified in this function
    """    
    fig, ax = plt.subplots()

    # plot data, adding legend depending on plot args
    if plot_args['has_legend']:
        ax.plot(x, y, 'o', markersize=1, label=plot_args['data_label'])
        ax.legend()
    else:
        ax.plot(x, y, 'o', markersize=1)

    # set labels
    ax.set_xlabel(plot_args['x_label'])
    ax.set_ylabel(plot_args['y_label'])
    ax.set_title(plot_args['title'])
    
    return fig, ax

def analyze_marker_deltas():
    """plots euclidean distance between tracked fiducial marker data
    reads data from 'output/Tracking_Output.csv' which is created in the marker tracking process
    saves plot to the 'figures' folder
    """    
    print("Analyzing tracked marker distances...")
    df = pd.read_csv("output/Tracking_Output.csv") # open csv created/modified from marker tracking process
    print(df.head())

    # ensure only 2 markers were selected
    if df['Tracker'].unique().shape[0] != 2:
        msg = "Found more/less than 2 markers.\n\nPlease ensure exactly 2 markers are tracked"
        error_popup(msg)
        return

    # grab relevant data and put into np array
    m1_df = df[df['Tracker'] == 1]
    m2_df = df[df['Tracker'] == 2]
    time = df['Time(s)'].unique()
    m1_x = m1_df['x (px)'].values
    m1_y = m1_df['y (px)'].values
    m2_x = m2_df['x (px)'].values
    m2_y = m2_df['y (px)'].values

    # find euclidean distances of markers
    marker_deltas = []
    for i in range(len(time)):
        euclidean_dist = marker_euclidean_distance(m1_x[i], m1_y[i], m2_x[i], m2_y[i])
        marker_deltas.append(np.abs(euclidean_dist))
    
    print(np.mean(marker_deltas), len(m1_x), len(m2_x), len(marker_deltas), marker_deltas[0], marker_deltas[-1])

    plot_args = {
        'title': 'Marker Delta Tracking',
        'x_label': 'Time (s)',
        'y_label': 'Marker Deltas (px)',
        'data_label': '',
        'has_legend': False
    }

    fig, ax = plot_data(time, marker_deltas, plot_args)
    fig.savefig("figures/marker_deltas.png")
    print("Done")


def analyze_necking_point():
    """plots necking point data, x location of necking point against time
    reads from 'output/Necking_Point_Output.csv' which is created in the necking point tracking process
    saves plot in 'figures' folder
    """    
    print("Analyzing necking point...")
    df = pd.read_csv("output/Necking_Point_Output.csv") # open csv created/modified from marker tracking process
    print(df.head())

    time = df['Time(s)'].values
    necking_pt_x = df['x at necking point (px)'].values

    plot_args = {
        'title': 'Necking Point Detection',
        'x_label': 'Time (s)',
        'y_label': 'Horizontal coordinate of necking point (px)',
        'data_label': '',
        'has_legend': False
    }

    fig, ax = plot_data(time, necking_pt_x, plot_args)
    fig.savefig("figures/necking_point_location.png")

    print("Done")

if __name__=='__main__':
    analyze_marker_deltas()