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

def analyze_marker_deltas(df=None, will_save_figures=True):
    """plots euclidean distance between tracked fiducial marker data
    reads data from 'output/Tracking_Output.csv' which is created in the marker tracking process
    saves plot to the 'figures' folder
    """    
    print("Analyzing tracked marker distances...")
    if not isinstance(df, pd.DataFrame):
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
    marker_distances = []
    for i in range(len(time)):
        euclidean_dist = marker_euclidean_distance(m1_x[i], m1_y[i], m2_x[i], m2_y[i])
        marker_distances.append(np.abs(euclidean_dist))
    
    print(np.mean(marker_distances), len(m1_x), len(m2_x), len(marker_distances), marker_distances[0], marker_distances[-1])

    # longitudinal strain (deltaL / L0)
    L0 = marker_distances[0]
    longitudinal_strain = [(L-L0) / L0 for L in marker_distances]

    plot_args = {
        'title': r'Marker Delta Tracking',
        'x_label': 'Time (s)',
        'y_label': 'Marker Deltas (px)',
        'data_label': '',
        'has_legend': False
    }

    if will_save_figures:
        # plot marker distances
        fig, ax = plot_data(time, marker_distances, plot_args)
        fig.savefig("figures/marker_deltas.png")

        # plot longitudinal strain
        plot_args['title'] = r'Longitudinal strain - ${\epsilon}_{l}(t)$'
        plot_args['y_label'] = r'${\epsilon}_{l}$'
        longitudinal_strain_fig, longitudinal_strain_ax = plot_data(time, longitudinal_strain, plot_args)
        longitudinal_strain_fig.savefig("figures/longitudinal_strain.png")

    print("Done")

    return time, longitudinal_strain


def analyze_necking_point(df=None, will_save_figures=True):
    """plots necking point data, x location of necking point against time, as well as diameter at necking point
    reads from 'output/Necking_Point_Output.csv' which is created in the necking point tracking process
    saves plot in 'figures' folder
    """    
    print("Analyzing necking point...")
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv("output/Necking_Point_Output.csv") # open csv created/modified from marker tracking process
    print(df.head())

    time = df['Time(s)'].values
    necking_pt_x = df['x at necking point (px)'].values
    necking_pt_len = df['y necking distance (px)'].values

    # radial strain (deltaR / R0)
    R0 = necking_pt_len[0]
    radial_strain = [(R - R0) / R0 for R in necking_pt_len]

    plot_args = {
        'title': 'Necking Point Horizontal Location',
        'x_label': 'Time (s)',
        'y_label': 'Horizontal coordinate of necking point (px)',
        'data_label': '',
        'has_legend': False
    }

    if will_save_figures:
        # plot x location of necking point over time
        necking_pt_loc_fig, necking_pt_loc_ax = plot_data(time, necking_pt_x, plot_args)
        necking_pt_loc_fig.savefig("figures/necking_point_location.png")

        # plot diameter at necking point
        plot_args['title'] = r'Diameter of Hydrogel at Necking Point' 
        plot_args['y_label'] = 'Diameter (px)'
        necking_pt_len_fig, necking_pt_len_ax = plot_data(time, necking_pt_len, plot_args)
        necking_pt_len_fig.savefig("figures/diameter_at_necking_point.png")

        # plot radial strain
        plot_args['title'] = r'Radial strain - ${\epsilon}_{r}(t)$'
        plot_args['y_label'] = '${\epsilon}_{r}$'
        radial_strain_fig, radial_strain_ax = plot_data(time, radial_strain, plot_args)
        radial_strain_fig.savefig("figures/radial_strain.png")
    print("Done")

    return time, radial_strain

def poissons_ratio():
    marker_time, longitudinal_strain = analyze_marker_deltas()
    necking_time, radial_strain = analyze_necking_point()

    print("Finding Poisson's ratio...")

    # ensure tracking operations previously ran on the same data in the same time range
    if not (marker_time[0] == necking_time[0] and marker_time[-1] == necking_time[-1]):
        msg = "Error: Found discrepancies in marker deltas output and necking point output.\n"+\
        "please ensure that both marker tracking and necking point detection are run on the same experiment within the same time frame."
        error_popup(msg)

    # align values, as time values in one may have some missing from the other (from outlier removal)
    marker_df = pd.DataFrame({
        'time': marker_time,
        'long_strain': longitudinal_strain
    })
    necking_df = pd.DataFrame({
        'time': necking_time,
        'rad_strain': radial_strain
    })

    # calculate poisson's ratio, radial / longitudinal
    poissons_df = pd.merge(marker_df, necking_df, 'inner', 'time')
    poissons_df['v'] = np.where(poissons_df['long_strain'] != 0, poissons_df['rad_strain'] / poissons_df['long_strain'], 0)
    print(poissons_df)

    plot_args = {
        'title': r"Poisson's Ratio - $\mathit{v(t)}$",
        'x_label': 'Time (s)',
        'y_label': r"Poisson's Ratio $\mathit{v}$",
        'data_label': '',
        'has_legend': False
    }

    poisson_fig, poisson_ax = plot_data(poissons_df['time'], poissons_df['v'], plot_args)
    poisson_fig.savefig("figures/poissons_ratio.png")

    print("Done")

if __name__=='__main__':
    analyze_marker_deltas()