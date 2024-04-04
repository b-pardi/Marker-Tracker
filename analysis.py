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
from enums import *


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

def get_time_labels(df):
    time_col = df.filter(like='Time').columns[0]
    time_label = r'Time, $\mathit{t}$ (s)'
    time_unit = TimeUnits.SECONDS.value
    if time_col.__contains__('min'):
        time_label = r'Time, $\mathit{t}$ (min)'
        time_unit = TimeUnits.MINUTES.value
    if time_col.__contains__('hr'):
        time_label = r'Time, $\mathit{t}$ (hr)'
        time_unit = TimeUnits.HOURS.value

    return time_col, time_label, time_unit

def check_tracker_data_lengths(df, n_trackers):
    tracker_shapes  = []
    col_start, col_end, n_sets = get_num_datasets(df)
    for dataset in range(col_start, col_end+1, 1):
        for tracker in range(n_trackers):
            cur_df = df[df[f'{dataset}-Tracker'] == tracker+1]
            tracker_shapes.append(cur_df.shape[0])
        min_len, max_len = min(tracker_shapes), max(tracker_shapes)
    if min_len != max_len:
        msg = "ERROR: Length of data entries for trackers are different.\n\n"+\
            "Please reattempt tracking"
        error_popup(msg)

def get_num_datasets(df):
    col_start = int(df.columns[2][0])
    col_end = int(df.columns[-1][0])
    n_sets = col_end - col_start + 1

    return col_start, col_end, n_sets

def check_num_trackers_with_num_datasets(n_trackers, num_tracker_datasets):
    data_multiplicity_type = 0
    if num_tracker_datasets > 1 and n_trackers > 1:
        msg = "ERROR: When using the append feature, only use one tracker per tracking operation.\n"+\
        "If using multiple trackers, only use one video tracking at a time for analysis\n"
        error_popup(msg)
        data_multiplicity_type = DataMultiplicity.BOTH
    elif num_tracker_datasets > 1 and n_trackers == 1:
        data_multiplicity_type = DataMultiplicity.VIDEOS
    elif n_trackers > 1 and num_tracker_datasets == 1:
        data_multiplicity_type = DataMultiplicity.TRACKERS
    elif n_trackers == 1 and num_tracker_datasets == 1:
        data_multiplicity_type = DataMultiplicity.SINGLE

    return data_multiplicity_type


def plot_scatter_data(x, y, plot_args, n_datasets, fig=None, ax=None):
    """util function to handle plotting and formatting of the plots
    can accept 1 or multiple independent variable datasets

    Args:
        x (pd.Dataframe/Series, np.Array, list, etc): x points to plot
        y (pd.Dataframe/Series, np.Array, list, etc): y points to plot
        plot_args (dict): dictionary of plot customization arguments

    Returns:
        (plt.figure, plt.axes): the figure and axes objects created and modified in this function
    """ 
    if fig is None and ax is None:
        fig, ax = plt.subplots()
    with open("plot_opts/plot_customizations.json", 'r') as plot_customs_file:
        plot_customs = json.load(plot_customs_file)

    print(len(x), len(y))
    for i in range(n_datasets):
        if i < 5:  # only 5 colors spec'd in plot customizations
            color = plot_customs['colors'][f'scatter{i+1}']
        else:
            color = f'C{i}'

        print(plot_args['data_label'])
        if plot_args['data_label'] is not None:
            if pd.isna(pd.Series(plot_args['data_label']).iloc[i]): # gross but effective way to check if nan or string
                label = f"data {i}"
            else:
                label = plot_args['data_label'][i]
        else:
            label = None
        ax.plot(x, y[i], 'o', markersize=1, color=color, label=label)      

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
    y_lower_bound = None if plot_customs['y_lower_bound'] == 'auto' else float(plot_customs['y_lower_bound'])
    y_upper_bound = None if plot_customs['y_upper_bound'] == 'auto' else float(plot_customs['y_upper_bound'])    
    plt.ylim(y_lower_bound, y_upper_bound)
    plt.title(plot_args['title'], fontsize=plot_customs['title_text_size'], fontfamily=font)
    plt.tight_layout()
    plt.figure(constrained_layout=True)

    return fig, ax

def plot_avgs_bar_data(n_ranges, x, y, plot_args, n_trackers=1):
    fig, ax = plt.subplots()
    with open("plot_opts/plot_customizations.json", 'r') as plot_customs_file:
        plot_customs = json.load(plot_customs_file)

    intervals = np.linspace(min(x), max(x), n_ranges+1) 
    averages = []

    for i in range(n_ranges):
        # find averages of each range interval
        range_start = intervals[i]
        range_end = intervals[i+1]
        range_idxs = (x >= range_start) & (x < range_end)
        averages.append([cur_y[range_idxs].mean() for cur_y in y])
    averages = np.array(averages)

    bar_width = 0.8 / n_trackers
    for i in range(n_trackers):
        if i < 5:  # only 5 colors spec'd in plot customizations
            color = plot_customs['colors'][f'scatter{i+1}']
        else:
            color = f'C{i}'

        if plot_args['data_label'] is not None:
            if plot_args['data_label'][i] == '':
                label = i
            else:
                label = plot_args['data_label'][i]
        else:
            label = None
        bar_pos = np.arange(n_ranges) - (0.4 - bar_width/2) + (i * bar_width)
        ax.bar(bar_pos, averages[:,i], width=bar_width, label=label, color=color)

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
    ax.set_xticks(np.arange(n_ranges))
    ax.set_xticklabels([f"{intervals[i]:.2f}-{intervals[i+1]:.2f}" for i in range(n_ranges)])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()

    return fig, ax

def get_marker_data(user_unit_conversion, df):
    print("Grabbing tracker data...")
    conversion_factor, conversion_units = user_unit_conversion

    n_trackers = df['Tracker'].unique().shape[0] # get number of trackers
    x_locations = []
    y_locations = []

    for tracker in range(n_trackers):
        cur_df = df[df['Tracker'] == tracker+1]
        x_locations.append(cur_df['x (px)'].values * conversion_factor)
        y_locations.append(cur_df['y (px)'].values * conversion_factor)

    return list(x_locations), list(y_locations)

def analyze_marker_deltas(user_unit_conversion, df=None, will_save_figures=True):
    """plots euclidean distance between tracked fiducial marker data
    reads data from 'output/Tracking_Output.csv' which is created in the marker tracking process
    saves plot to the 'figures' folder
    """    
    conversion_factor, conversion_units = user_unit_conversion

    print("Analyzing tracked marker distances...")
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv("output/Tracking_Output.csv") # open csv created/modified from marker tracking process
    print(df.head())

    # ensure only 2 markers were selected
    if df['1-Tracker'].unique().shape[0] != 2:
        msg = "Found more/less than 2 markers.\n\nPlease ensure exactly 2 markers are tracked"
        error_popup(msg)
        return
    if int(df.columns[-1][0]) != 1:
        msg = "Found more than 1 tracked video dataset.\nPlease ensure overwrite is selected instead of append for Poissons ratio related tracking"
        error_popup(msg)
        return
    
    time_col, time_label, _ = get_time_labels(df)
    # grab relevant data and put into np array
    m1_df = df[df['1-Tracker'] == 1]
    m2_df = df[df['1-Tracker'] == 2]
    time = df[time_col].unique()
    m1_x = m1_df['1-x (px)'].values * conversion_factor
    m1_y = m1_df['1-y (px)'].values * conversion_factor
    m2_x = m2_df['1-x (px)'].values * conversion_factor
    m2_y = m2_df['1-y (px)'].values * conversion_factor
    data_labels = []

    # find euclidean distances of markers
    marker_distances = []
    for i in range(len(time)):
        euclidean_dist = marker_euclidean_distance(m1_x[i], m1_y[i], m2_x[i], m2_y[i])
        marker_distances.append(np.abs(euclidean_dist))
        data_labels.append(df['1-data_label'].unique()[0])
    
    # longitudinal strain (deltaL / L0)
    L0 = marker_distances[0]
    longitudinal_strain = [(L-L0) / L0 for L in marker_distances]

    plot_args = {
        'title': r'Longitudinal Strain of Hydrogel',
        'x_label': time_label,
        'y_label': r'Longitudinal Strain, ${\epsilon}_{l}$',
        'data_label': data_labels,
        'has_legend': False
    }

    if will_save_figures:
        # plot longitudinal strain
        longitudinal_strain_fig, longitudinal_strain_ax = plot_scatter_data(time, [longitudinal_strain], plot_args, n_datasets=1)
        longitudinal_strain_fig.savefig("figures/longitudinal_strain.png")

        # plot marker distances
        plot_args['title'] = 'Marker Delta Tracking'
        plot_args['y_label'] = rf'Marker elongation, $\mathit{{\Delta {conversion_units}}}$'
        print(len(time), len(marker_distances), len(np.abs(m2_x-m1_x)))
        plot_args['data_label'] = ['Euclidean Distances', 'Horizontal Differences']
        fig, ax = plot_scatter_data(time, [marker_distances, np.abs(m2_x-m1_x)], plot_args, n_datasets=2) # plot x distance as well for control/comparison
        fig.savefig("figures/marker_deltas.png")

        # plot difference between euclidean and horizontal differences
        plot_args['title'] = 'Euclidean and Horizontal Differences'
        plot_args['y_label'] = rf'Marker elongation, $\mathit{{{conversion_units}}}$'
        plot_args['data_label'] = data_labels
        fig, ax = plot_scatter_data(time, [np.abs(marker_distances - np.abs(m2_x-m1_x))], plot_args, n_datasets=1) # plot x distance as well for control/comparison
        fig.savefig("figures/marker_euclidean_horizontal_differences.png")

    print("Done")

    return list(time), longitudinal_strain, plot_args


def analyze_necking_point(user_unit_conversion, df=None, will_save_figures=True):
    """plots necking point data, x location of necking point against time, as well as diameter at necking point
    reads from 'output/Necking_Point_Output.csv' which is created in the necking point tracking process
    saves plot in 'figures' folder
    """    
    conversion_factor, conversion_units = user_unit_conversion

    print("Analyzing necking point...")
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv("output/Necking_Point_Output.csv") # open csv created/modified from marker tracking process
    print(df.head())

    if int(df.columns[-1][0]) != 1:
        msg = "Found more than 1 tracked video dataset.\nPlease ensure overwrite is selected instead of append for Poissons ratio related tracking"
        error_popup(msg)
        return

    time_col, time_label, _ = get_time_labels(df)
    time = df[time_col].values
    data_labels = []
    necking_pt_x = df['1-x at necking point (px)'].values * conversion_factor
    necking_pt_len = df['1-y necking distance (px)'].values * conversion_factor
    data_labels.append(df['1-data_label'].unique()[0])

    # radial strain (deltaR / R0)
    R0 = necking_pt_len[0]
    radial_strain = [(R - R0) / R0 for R in necking_pt_len]

    plot_args = {
        'title': 'Radial Strain of Hydrogel',
        'x_label': time_label,
        'y_label': r'Radial strain, ${\epsilon}_{r}$',
        'data_label': data_labels,
        'has_legend': False
    }

    if will_save_figures:
        # plot radial strain
        radial_strain_fig, radial_strain_ax = plot_scatter_data(time, [radial_strain], plot_args, n_datasets=1)
        radial_strain_fig.savefig("figures/radial_strain.png")

        # plot x location of necking point over time
        plot_args['title'] = 'Necking Point Horizontal Location'
        plot_args['y_label'] = f'Horizontal location of necking point ({conversion_units})'
        necking_pt_loc_fig, necking_pt_loc_ax = plot_scatter_data(time, [necking_pt_x], plot_args, n_datasets=1)
        necking_pt_loc_fig.savefig("figures/necking_point_location.png")

        # plot diameter at necking point
        plot_args['title'] = r'Diameter of Hydrogel at Necking Point' 
        plot_args['y_label'] = f'Diameter ({conversion_units})'
        necking_pt_len_fig, necking_pt_len_ax = plot_scatter_data(time, [necking_pt_len], plot_args, n_datasets=1)
        necking_pt_len_fig.savefig("figures/diameter_at_necking_point.png")

    print("Done")

    return list(time), radial_strain, plot_args

def poissons_ratio(user_unit_conversion):
    conversion_factor, conversion_units = user_unit_conversion
    marker_time, longitudinal_strain, plot_args = analyze_marker_deltas(user_unit_conversion)
    necking_time, radial_strain, _ = analyze_necking_point(user_unit_conversion)
    data_label = plot_args['data_label']
    print("LABEL", plot_args['data_label'])
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
    time_col, time_label, _ = get_time_labels(pd.read_csv("output/Necking_Point_Output.csv"))
    
    poissons_df['d(long_strain)/dt'] = poissons_df['long_strain'].diff() / poissons_df['time']
    poissons_df['d(rad_strain)/dt'] = poissons_df['rad_strain'].diff() / poissons_df['time']
    poissons_df['d(v)/dt'] = poissons_df['v'].diff() / poissons_df['time']
    
    print(poissons_df)
    poissons_df.to_csv("output/poissons_ratio.csv")

    plot_args['title'] = r"Poisson's Ratio - $\mathit{\nu(t)}$"
    plot_args['y_label'] = r"Poisson's ratio, $\mathit{\nu}$"

    # plot poissons ratio against time
    poisson_fig, poisson_ax = plot_scatter_data(poissons_df['time'], [poissons_df['v']], plot_args, n_datasets=1)
    poisson_fig.savefig("figures/poissons_ratio.png")

    # plot derivatives
    plot_args['title'] = r"Derivative of Poissons Ratio - $\dot{\nu} (t)$"
    plot_args['y_label'] = r"$\dot{\nu}(t)$"
    poisson_prime_fig, poisson_prime_ax = plot_scatter_data(poissons_df['time'], [poissons_df['d(v)/dt']], plot_args, n_datasets=1)
    poisson_prime_fig.savefig("figures/poissons_ratio_prime.png")

    plot_args['title'] = r"Derivative of Longitudinal Strain - $\dot{\epsilon}_l (t)$"
    plot_args['y_label'] = r"$\dot{\epsilon}_l (t)$"
    long_strain_prime_fig, long_strain_prime_ax = plot_scatter_data(poissons_df['time'], [poissons_df['d(long_strain)/dt']], plot_args, n_datasets=1)
    long_strain_prime_fig.savefig("figures/long_strain_prime.png")

    plot_args['title'] = r"Derivative of Radial Strain - $\dot{\epsilon}_r (t)$"
    plot_args['y_label'] = r"$\dot{\epsilon}_r (t)$"
    rad_strain_prime_fig, rad_strain_prime_ax = plot_scatter_data(poissons_df['time'], [poissons_df['d(rad_strain)/dt']], plot_args, n_datasets=1)
    rad_strain_prime_fig.savefig("figures/rad_strain_prime.png")

    print("Done")

def marker_velocity(user_unit_conversion, df=None, will_save_figures=True, chosen_video_data=None):
    print("Finding Marker Velocity...")

    conversion_factor, conversion_units = user_unit_conversion
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv("output/Tracking_Output.csv") # open csv created/modified from marker tracking process
    print(df.head())

    time_col, time_label, time_unit = get_time_labels(df)
    _, _, num_tracker_datasets = get_num_datasets(df) # get num datasets
    if chosen_video_data is None:
        n_trackers = df['1-Tracker'].unique().shape[0] # get number of trackers
    else:
        n_trackers = df[f'{chosen_video_data}-Tracker'].unique().shape[0] # get number of trackers
    n_plots = 0
    times = []
    data_labels = []
    tracker_velocities = []
    tracker_amplitudes = []
    tracker_frequencies = []

    # determine if there are multiple trackers and 1 video, or multiple videos and 1 tracker
    data_multiplicity_type = check_num_trackers_with_num_datasets(n_trackers, num_tracker_datasets)
    if data_multiplicity_type == DataMultiplicity.TRACKERS or data_multiplicity_type == DataMultiplicity.SINGLE:
        # account for mismatch lengths of tracker information (if 1 tracker fell off before the other)
        check_tracker_data_lengths(df, n_trackers)
        n_plots = n_trackers
        num_sets = n_trackers
    elif data_multiplicity_type == DataMultiplicity.VIDEOS:
        n_plots = num_tracker_datasets
        num_sets = num_tracker_datasets

    # grab relevant values from df
    vel_df = pd.DataFrame({time_col: df[time_col].unique()[:-1]})
    for n in range(num_sets):
        if data_multiplicity_type == DataMultiplicity.SINGLE:
            if chosen_video_data is None:
                chosen_video_data = 1
            print("TEST SINGLE AND CHOSEN VIDEO DATA")
            cur_df = df[df[f'{chosen_video_data}-Tracker'] == n+1]
            x = cur_df[f'{chosen_video_data}-x (px)'].values * conversion_factor
            y = cur_df[f'{chosen_video_data}-y (px)'].values * conversion_factor
            data_labels.append(cur_df[f'{chosen_video_data}-data_label'].unique()[0])
            time = cur_df[time_col].values
        elif data_multiplicity_type == DataMultiplicity.TRACKERS:
            cur_df = df[df['1-Tracker'] == n+1]
            x = cur_df['1-x (px)'].values * conversion_factor
            y = cur_df['1-y (px)'].values * conversion_factor
            time = cur_df[time_col].values
            data_labels.append(cur_df['1-data_label'].unique()[0])
        elif data_multiplicity_type == DataMultiplicity.VIDEOS:
            x = df[f'{n+1}-x (px)'].values * conversion_factor
            y = df[f'{n+1}-y (px)'].values * conversion_factor
            time = df[time_col].values
            data_labels.append(df[f'{n+1}-data_label'].unique()[0])

        # get differences
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(time)
        print(dx.shape, dy.shape, dt.shape)

        # get velocities
        vel_x = dx / dt
        vel_y = dy / dt
        vel_mag = np.sqrt(vel_x**2 + vel_y**2) # magnitude of velocities
        times.append(time[:-1])
        tracker_velocities.append(vel_mag)
        print(vel_x.shape)

        vel_df[f'x_velocity_tracker{n+1}'] = vel_x
        vel_df[f'y_velocity_tracker{n+1}'] = vel_y
        vel_df[f'magnitude_velocity_tracker{n+1}'] = vel_mag

        # fourier transform
        vel_fft = np.abs(np.fft.fft(vel_mag)) # amplitude (pixels/Hz)
        freqs = np.fft.fftfreq(len(vel_mag), np.mean(dt)) # frequencies
        tracker_amplitudes.append(vel_fft)
        tracker_frequencies.append(freqs)

    # save intermediate calculations
    vel_df.to_csv("output/marker_velocity.csv", index=False)

    # plot
    plot_args = {
        'title': r'Cell Velocity',
        'x_label': time_label,
        'y_label': rf'Average magnitude of cell velocity, $|\mathit{{v}}|$ $(\frac{{{conversion_units}}}{{{time_unit}}})$',
        'data_label': data_labels,
        'has_legend': True,

    }
    if will_save_figures:
        # plot marker velocity
        vel_fig, vel_ax = plot_scatter_data(times[0], tracker_velocities, plot_args, n_plots)
        vel_fig.savefig("figures/marker_velocity.png")

        # plot fourier transform of marker distances
        plot_args['title'] = 'Marker Velocity FFT'
        plot_args['y_label'] = f'({conversion_units}/Hz)'
        plot_args['x_label'] = 'Hz'
        fft_fig, fft_ax = plt.subplots()
        for i in range(n_plots):
            plot_scatter_data(tracker_frequencies[i], [tracker_amplitudes[i]], plot_args, 1, fft_fig, fft_ax)
        fft_fig.savefig("figures/marker_velocity_FFT.png")

    print("Done")
    return times, tracker_velocities, plot_args, n_plots

def velocity_boxplot(conditions, user_unit_conversion):
    print(f"Piecing together boxplot of cell velocities for each of the given conditions: {conditions}")
    df = pd.read_csv("output/Tracking_Output.csv")
    time_col, time_label, time_unit = get_time_labels(df)
    conversion_factor, conversion_units = user_unit_conversion
    time = df[time_col].values

    condition_avg_velocities = {} # group average velocities by condition
    for condition in conditions: # separate output file datasets by condition spec'd by user found in datalabels
        cur_condition_avg_velocities = []
        for col in df.columns: # find dataset index of current condition
            if str(df[col].unique()[0]).__contains__(condition):
                cur_dataset = col[0]
                cur_df = df.filter(like=cur_dataset)

                # grab data from cur df
                x = cur_df[f'{cur_dataset}-x (px)'].values * conversion_factor
                y = cur_df[f'{cur_dataset}-y (px)'].values * conversion_factor

                # get velocities
                dx = np.diff(x)
                dy = np.diff(y)
                dt = np.diff(time)
                vel_x = dx / dt
                vel_y = dy / dt
                vel_mag = np.sqrt(vel_x**2 + vel_y**2) # magnitude of velocities
                cur_condition_avg_velocities.append(np.mean(vel_mag))

        condition_avg_velocities[condition] = cur_condition_avg_velocities
    print(condition_avg_velocities)
    data_lists = [condition_avg_velocities[condition] for condition in conditions]

    # plotting
    plt.boxplot(data_lists)
    plt.xticks([1,2], list(condition_avg_velocities.keys()))
    plt.title("Average Cell Velocities Across Multiple Conditions")
    plt.ylabel(rf'Magnitude of cell velocity, $|\mathit{{v}}|$ $(\frac{{{conversion_units}}}{{{time_unit}}})$')
    plt.xlabel(time_label)
    plt.savefig("figures/marker_velocity_boxplot")


def marker_distance(user_unit_conversion):
    print("Finding Marker RMS Distance...")
    conversion_factor, conversion_units = user_unit_conversion
    df = pd.read_csv("output/Tracking_Output.csv") # open csv created/modified from marker tracking process
    print(df.head())

    time_col, time_label, _ = get_time_labels(df)
    _, _, num_tracker_datasets = get_num_datasets(df) # get num datasets
    n_trackers = df['1-Tracker'].unique().shape[0] # get number of trackers
    data_labels = []
    rms_disps = []
    time = df[time_col].unique()
    print(n_trackers, num_tracker_datasets)
    n_plots = 0

    # determine if there are multiple trackers and 1 video, or multiple videos and 1 tracker
    data_multiplicity_type = check_num_trackers_with_num_datasets(n_trackers, num_tracker_datasets)

    if data_multiplicity_type == DataMultiplicity.TRACKERS or data_multiplicity_type == DataMultiplicity.SINGLE:
        # account for mismatch lengths of tracker information (if 1 tracker fell off before the other)
        check_tracker_data_lengths(df, n_trackers)
        n_plots = n_trackers
        label = 'Tracker'
        num_sets = n_trackers
        for tracker in range(n_trackers):
            cur_df = df[df['1-Tracker'] == tracker+1]
            x = cur_df['1-x (px)'].values * conversion_factor
            y = cur_df['1-y (px)'].values * conversion_factor

            data_labels.append(cur_df['1-data_label'].unique()[0])
            rms_disps.append(rms_displacement(np.diff(x), np.diff(y)))

    elif data_multiplicity_type == DataMultiplicity.VIDEOS:
        n_plots = num_tracker_datasets
        label = 'Video'
        num_sets = num_tracker_datasets
        for dataset in range(num_tracker_datasets):
            x = df[f'{dataset+1}-x (px)'].values * conversion_factor
            y = df[f'{dataset+1}-y (px)'].values * conversion_factor
            data_labels.append(df[f'{dataset+1}-data_label'].unique()[0])
            rms_disps.append(rms_displacement(np.diff(x), np.diff(y)))

    plot_args = {
        'title': 'Root mean square Displacement',
        'x_label': time_label,
        'y_label': r'Root mean square Displacement, $\mathit{D_{rms}}$' + rf' ({conversion_units})    ',
        'data_label': data_labels,
        'has_legend': True
    }
    # plot marker velocity
    disp_fig, disp_ax = plot_scatter_data(time[:-1], rms_disps, plot_args, n_plots)
    disp_fig.savefig("figures/marker_RMS_displacement.png")

def single_marker_spread(user_unit_conversion, df=None, will_save_figures=True, chosen_video_data=None):
    print("Finding Marker Spread...")
    conversion_factor, conversion_units = user_unit_conversion

    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv("output/Surface_Area_Output.csv") # open csv created/modified from marker tracking process
    print(df.head())

    _, _, num_tracker_datasets = get_num_datasets(df)
    time_col, time_label, time_unit = get_time_labels(df)
    data_multiplicity_type = check_num_trackers_with_num_datasets(1, num_tracker_datasets)

    surface_areas = []
    data_labels = []
    time = df[time_col].values
    for i in range(num_tracker_datasets):
        if data_multiplicity_type == DataMultiplicity.SINGLE:
            if chosen_video_data is None:
                chosen_video_data = 1
            surface_area = df[f'{chosen_video_data}-cell surface area (px^2)'].values * conversion_factor
            data_labels.append(df[f'{chosen_video_data}-data_label'].unique()[0])
        else:
            surface_area = df[f'{i+1}-cell surface area (px^2)'].values * conversion_factor
            data_labels.append(df[f'{i+1}-data_label'].unique()[0])

        surface_areas.append(surface_area)

    # plot
    plot_args = {
        'title': r'Cell Surface Area Spread',
        'x_label': time_label,
        'y_label': rf'Projected cell area, $\mathit{{A}}$ ' + rf'({conversion_units})$^{{2}}$',
        'data_label': data_labels,
        'has_legend': True,
    }

    if will_save_figures:
        # plot marker velocity
        area_fig, area_ax = plot_scatter_data(time, surface_areas, plot_args, num_tracker_datasets)
        area_fig.savefig("figures/marker_surface_area.png", dpi=400)

    return time, surface_area, plot_args, num_tracker_datasets

if __name__=='__main__':
    analyze_marker_deltas()