'''
Author: Brandon Pardi
Created 2/12/2024

performs various operations and visual analysis on tracked data
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

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

def rms_displacement(dx, dy):
    """calculate the root mean square distance of x, y points over time from displacements
    characterize the magnitude of fluctuations/movements of the marker

    Args:
        x (_type_): _description_
        y (_type_): _description_
    """    
    dx_sq = dx**2
    dy_sq = dy**2

    rms_displacement = np.sqrt(np.cumsum(dx_sq + dy_sq) / (np.arange(len(dx))+1))
    return rms_displacement

def plot_data(x, y, plot_args):
    """util function to handle plotting and formatting of the plots
    can accept 1 or multiple independent variable datasets

    Args:
        x (pd.Dataframe/Series, np.Array, list, etc): x points to plot
        y (pd.Dataframe/Series, np.Array, list, etc): y points to plot
        plot_args (dict): dictionary of plot customization arguments

    Returns:
        (plt.figure, plt.axes): the figure and axes objects created and modified in this function
    """ 
    fig, ax = plt.subplots()
    with open("plot_opts/plot_customizations.json", 'r') as plot_customs_file:
        plot_customs = json.load(plot_customs_file)

    # plotting if passed in 1 dataset
    if len(x) == len(y): # if passing in 1 dependent datset, lens will match
        color = plot_customs['colors'][f'scatter1']
        if plot_args['data_label'] != '':
            label = plot_args['data_label']
        else:
            label = None
        ax.plot(x, y, 'o', markersize=1, color=color, label=label)

    # plotting if passed in > 1 dataset
    else:
        for i, dataset in enumerate(y):
            if i < 5:  # only 5 colors spec'd in plot customizations
                color = plot_customs['colors'][f'scatter{i+1}']
            else:
                color = f'C{i}'

            if plot_args['data_label'] != '':
                label = plot_args['data_label'][i]
            else:
                label = None
            ax.plot(x, dataset, 'o', markersize=1, color=color, label=label)


    # adding legend depending on plot args
    if plot_args['has_legend']:
        ax.legend()
    
    # formatting the plot according to 'plot_customizations.json'
    font = plot_customs['font']
    plt.sca(ax)
    plt.legend(loc='best', fontsize=plot_customs['legend_text_size'], prop={'family': font}, framealpha=0.3)
    plt.xticks(fontsize=plot_customs['value_text_size'], fontfamily=font)
    plt.yticks(fontsize=plot_customs['value_text_size'], fontfamily=font) 
    plt.xlabel(plot_args['x_label'], fontsize=plot_customs['label_text_size'], fontfamily=font)
    plt.ylabel(plot_args['y_label'], fontsize=plot_customs['label_text_size'], fontfamily=font)
    plt.tick_params(axis='both', direction=plot_customs['tick_dir'])
    plt.title(plot_args['title'], fontsize=plot_customs['title_text_size'], fontfamily=font)

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
        print(len(time), len(marker_distances), len(np.abs(m2_x-m1_x)))
        plot_args['data_label'] = ['Euclidean Distances', 'Horizontal Differences']
        fig, ax = plot_data(time, [marker_distances, np.abs(m2_x-m1_x)], plot_args) # plot x distance as well for control/comparison
        fig.savefig("figures/marker_deltas.png")

        # plot longitudinal strain
        plot_args['title'] = r'Longitudinal strain - ${\epsilon}_{l}(t)$'
        plot_args['y_label'] = r'${\epsilon}_{l}$'
        plot_args['data_label'] = ''
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
        plot_args['y_label'] = r'${\epsilon}_{r}$'
        radial_strain_fig, radial_strain_ax = plot_data(time, radial_strain, plot_args)
        radial_strain_fig.savefig("figures/radial_strain.png")
    print("Done")

    return time, radial_strain

def poissons_ratio():
    marker_time, longitudinal_strain = analyze_marker_deltas()
    necking_time, radial_strain = analyze_necking_point()

    print("Finding Poisson's ratio...")

    # ensure tracking operations previously ran on the same data in the same time range
    if not (marker_time[0] == necking_time[0] and marker_time[-1] == necking_time[-1] and len(marker_time) == len(necking_time)):
        msg = "Warning: Found discrepancies in marker deltas output and necking point output.\n"+\
        "If this is due to outlier removal in one but not the other, proceed as normal.\n"+\
        "Otherwise please ensure that both marker tracking and necking point detection are run on the same experiment within the same time frame."
        warning_popup(msg)

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
    poissons_df['v'] = np.where(poissons_df['long_strain'] != 0, -1 * poissons_df['rad_strain'] / poissons_df['long_strain'], 0)
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

def single_marker_velocity(df=None, will_save_figures=True):
    print("Finding Marker Velocity...")
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv("output/Tracking_Output.csv") # open csv created/modified from marker tracking process
    print(df.head())

    # grab relevant values from df
    time = df['Time(s)'].values
    x = df['x (px)'].values
    y = df['y (px)'].values

    # get differences
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(time)

    # get velocities
    vel_x = dx / dt
    vel_y = dy / dt
    vel_mag = np.sqrt(vel_x**2 + vel_y**2) # magnitude of velocities

    # fourier transform
    vel_fft = np.abs(np.fft.fft(vel_mag)) # amplitude (pixels/Hz)
    freqs = np.fft.fftfreq(len(vel_mag), np.mean(dt)) # frequencies

    # plot
    plot_args = {
        'title': r'Cell Velocity',
        'x_label': 'Time (s)',
        'y_label': r'Magnitude of Cell Velocity $\mathit{\frac{pixels}{second}}$',
        'data_label': '',
        'has_legend': False
    }

    if will_save_figures:
        # plot marker velocity
        vel_fig, vel_ax = plot_data(time[:-1], vel_mag, plot_args)
        vel_fig.savefig("figures/marker_velocity.png")

        # plot fourier transform of marker distances
        plot_args['title'] = 'Marker Velocity FFT'
        plot_args['y_label'] = 'Pixels/Hz'
        plot_args['x_label'] = 'Hz'
        fft_fig, fft_ax = plot_data(freqs, vel_fft, plot_args)
        fft_fig.savefig("figures/marker_velocity_FFT.png")

    print("Done")
    return list(time[:-1]), list(vel_mag)


def single_marker_distance():
    print("Finding Marker RMS Distance...")
    
    df = pd.read_csv("output/Tracking_Output.csv") # open csv created/modified from marker tracking process
    print(df.head())
    
    time = df['Time(s)'].values
    x = df['x (px)'].values
    y = df['y (px)'].values

    rms_disps = rms_displacement(np.diff(x), np.diff(y))

    # plot
    plot_args = {
        'title': 'Cell RMS Displacement',
        'x_label': 'Time (s)',
        'y_label': 'RMS (px)',
        'data_label': '',
        'has_legend': False
    }
    # plot marker velocity
    vel_fig, vel_ax = plot_data(time[:-1], rms_disps, plot_args)
    vel_fig.savefig("figures/marker_RMS_displacement.png")

def single_marker_spread():
    print("Finding Marker Spread...")


if __name__=='__main__':
    analyze_marker_deltas()