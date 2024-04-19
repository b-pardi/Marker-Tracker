'''
Author: Brandon Pardi
Created 2/12/2024

performs various operations and visual analysis on tracked data
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import re

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

def get_time_labels(df, col_num=1):
    ''' util function to return the appropriate strings for:
        - time column in dataframe
        - plot x axis label
        - specific time unit
    '''
    time_col = f'{col_num}-' + (df.filter(like='Time').columns[0]).split('-')[1] # accounts for any kind of units in time col
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
    col_start = int(df.columns[0][0])
    col_end = int(df.columns[-1][0])
    n_sets = col_end - col_start + 1

    return col_start, col_end, n_sets

def get_relevant_columns(df, col_num):
    relevant_cols = [col for col in df.columns if col.startswith(f"{col_num}-")]
    return df[relevant_cols]

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


def plot_scatter_data(x, y, plot_args, n_datasets, fig=None, ax=None, output_fig_name=None):
    """
    Handles the plotting and formatting of scatter plot data, supporting both single and multiple datasets. 
    The function customizes the plot appearance based on provided arguments and saves the plot if specified.

    Details:
        - This function reads plot customization settings from a JSON file ('plot_customizations.json') to apply color schemes, fonts, and other stylistic elements.
        - The function can manage up to five datasets with predefined colors, and uses default matplotlib colors for additional datasets.
        - Data labels are managed dynamically; if a label is not provided or is NaN for a dataset, a default label is assigned.
        - Legend placement is automatically adjusted based on the number of datasets to enhance plot readability.
        - Axis labels, ticks, and title are set according to `plot_args`.
        - Optionally, the plot can be saved in a specified format with a high resolution, and customization details are pulled from the JSON file.

    Note:
        - Expects a list of datasets, if using only 1 it should be wrapped in a list.
        - Ensure that `plot_customs` JSON file contains all required customization parameters.
        - If `output_fig_name` is specified, ensure the 'figures' directory exists or handle the potential `FileNotFoundError`.

    Args:
        x (pd.DataFrame/Series, np.array, list, etc.): Independent variable data points to plot.
        y (pd.DataFrame/Series, np.array, list, etc.): Dependent variable data points to plot. Each dataset in `y` is plotted against `x`.
        plot_args (dict): Dictionary containing customization options such as labels, titles, and legend configuration.
        n_datasets (int): Number of datasets to plot, typically the length of `y` if it contains multiple sets.
        fig (matplotlib.figure.Figure, optional): Figure object to plot on, if None a new figure is created.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on, if None a new axes is created on the figure.
        output_fig_name (str, optional): Filename to save the figure; if not provided, the figure is not saved.

    Returns:
        tuple: A tuple containing the matplotlib figure and axes objects (`fig`, `ax`) used or created.
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8,5))
        n_prev_plots = 0 # if adding to existing plot, continue color index
    else:
        n_prev_plots = len(ax.lines)

    with open("plot_opts/plot_customizations.json", 'r') as plot_customs_file:
        plot_customs = json.load(plot_customs_file)

    print(len(x), len(y))
    for i in range(n_datasets):
        if i+n_prev_plots < 5:  # only 5 colors spec'd in plot customizations
            color = plot_customs['colors'][f'scatter{i+1+n_prev_plots}']
        else:
            color = f'C{i+n_prev_plots}'

        if plot_args['data_label'] is not None:
            if pd.isna(pd.Series(plot_args['data_label']).iloc[i]): # gross but effective way to check if nan or string
                label = f"data {i}"
            else:
                label = plot_args['data_label'][i]
        else:
            label = None
        ax.plot(x[i], y[i], 'o', markersize=1, color=color, label=label)      

    font = plot_customs['font']

    # adding legend depending on plot args
    if plot_args['has_legend']:
        if n_datasets <= 3:
            legend = ax.legend(loc='best', fontsize=plot_customs['legend_text_size'], prop={'family': font}, framealpha=0.3)
        else: # put legend outside plot if more than 3 datasets for readability
            legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=plot_customs['legend_text_size'], prop={'family': font}, framealpha=0.3)
    else:
        legend = None
    # formatting the plot according to 'plot_customizations.json'
    plt.sca(ax)
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

    if output_fig_name is not None:
        plt.savefig(f"figures/{output_fig_name}.{plot_customs['fig_format']}", bbox_extra_artists=(legend,), dpi=plot_customs['fig_dpi'])
        plt.close()
    return fig, ax

def plot_avgs_bar_data(n_ranges, x, y, plot_args, n_trackers=1, output_fig_name=None):
    '''DEPRECATED'''
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

    if output_fig_name is not None:
        plt.savefig(f"figures/{output_fig_name}.{plot_customs['fig_format']}", dpi=plot_customs['fig_dpi'])

    return fig, ax

def plot_fft(freqs, mags, N, n_datasets, plot_args):  
    """
    Plots the Fast Fourier Transform (FFT) of given signal data. This function visualizes frequency domains of multiple datasets,
    highlighting the dominant frequency for each dataset.

    Details:
        - The FFT often produces a mirror image around the Nyquist frequency, so this function only considers the first half of the frequency data.
        - The magnitude of the FFT is taken as the absolute value, since FFT outputs are generally complex numbers.
        - For each dataset, the dominant frequency (frequency with the highest amplitude) is identified and highlighted in the legend.
        - This function is configured to save the plot automatically to a specified file.

    Note:
        - Ensure that all frequency and magnitude arrays are of the same length and correspond to each other in the `freqs` and `mags` lists.
        - The plot is saved to 'figures/marker_velocity_FFT.png'; ensure the directory exists or handle potential errors for file writing.

    Args:
        freqs (list of np.array): Frequencies for each dataset; expects each list element to be an array of frequencies.
        mags (list of np.array): Magnitudes corresponding to the frequencies for each dataset.
        N (int): Total number of data points in the FFT. Used to limit the data to the first half (real component).
        n_datasets (int): Number of datasets to plot.
        plot_args (dict): Dictionary containing plot options such as data labels.    
    """  
    fft_fig, fft_ax = plt.subplots()
    for n in range(n_datasets):
        # fft produces mirror image so take first half of data and its abs value for amplitdue
        cur_freqs = (freqs[n])[:N // 2]
        cur_mags = np.abs(mags[n])[:N // 2]
        max_magnitude_idx = np.argmax(cur_mags)
        dom_freq = cur_freqs[max_magnitude_idx]
        fft_ax.plot(cur_freqs, cur_mags, color=f'C{n}', label=f"Dominant frequency for {plot_args['data_label'][n]} {dom_freq:.2f} Hz")

    fft_ax.legend()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude normalized")
    plt.title("Fourier Transform of Selected Signal Data")
    plt.grid(True)
    fft_fig.tight_layout()
    fft_fig.savefig("figures/marker_velocity_FFT.png")

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

def analyze_marker_deltas(user_unit_conversion, df=None, will_save_figures=True, chosen_video_data=None):
    """
    Analyzes and plots the Euclidean distances between two tracked markers to evaluate marker deltas and 
    calculates longitudinal strain from these distances. Optionally saves the plots to a specified directory.

    Details:
        - The function expects the DataFrame to have data from exactly two markers for correct processing. If more or fewer are found, an error is raised.
        - It checks for the presence of more than one dataset. If found, an error is raised assuming only single-session tracking data should be analyzed.
        - Euclidean distance between the two markers is computed for each time point and converted using the provided conversion factor.
        - Longitudinal strain is calculated as the change in distance relative to the initial distance.
        - The function generates several plots:
            1. Longitudinal strain over time.
            2. Euclidean distances and horizontal differences between markers.
            3. Difference between Euclidean and purely horizontal distances for further analysis.

    Note:
        - If saving plots, ensure that the 'figures' directory exists or handle potential FileNotFoundError issues.

    Args:
        user_unit_conversion (tuple): Tuple containing a conversion factor and unit name to convert pixel distances to a desired unit.
        df (pd.DataFrame, optional): DataFrame containing marker tracking data. If not provided, data is read from 'output/Tracking_Output.csv'.
        will_save_figures (bool, optional): If True, saves generated plots to the 'figures' directory. Defaults to True.

    Returns:
        tuple: Returns a tuple containing time points, longitudinal strains, and plot arguments used for plotting.
    """
    print("Analyzing tracked marker distances...")
    conversion_factor, conversion_units = user_unit_conversion

    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv("output/Tracking_Output.csv") # open csv created/modified from marker tracking process
    print(df.head())
    _, _, n_datasets = get_num_datasets(df)

    # initialize lists to store processed data
    times = []
    marker_distances = []
    horizontal_distances = []
    longitudinal_strains = []
    elongation_differences = []
    data_labels = []

    if chosen_video_data: # if called by outlier removal tool
        for_range = (chosen_video_data, chosen_video_data+1)
    else:
        for_range = (1, n_datasets+1)

    for i in range(for_range[0], for_range[1]):
        # ensure only 2 markers were selected
        if df[f'{i}-Tracker'].dropna().unique().shape[0] != 2:
            msg = "Found more/less than 2 markers.\n\nPlease ensure exactly 2 markers are tracked in each dataset"
            error_popup(msg)
            return
    
        cur_df = get_relevant_columns(df, i).dropna()
        time_col, time_label, _ = get_time_labels(df, i)

        # grab relevant data and put into np array
        m1_df = cur_df[cur_df[f'{i}-Tracker'] == 1]
        m2_df = cur_df[cur_df[f'{i}-Tracker'] == 2]
        time = cur_df[time_col].unique()
        m1_x = m1_df[f'{i}-x (px)'].values * conversion_factor
        m1_y = m1_df[f'{i}-y (px)'].values * conversion_factor
        m2_x = m2_df[f'{i}-x (px)'].values * conversion_factor
        m2_y = m2_df[f'{i}-y (px)'].values * conversion_factor
        
        # find euclidean distances of markers
        marker_dist = []
        print(time, len(time), time_col)
        for j in range(len(time)):
            euclidean_dist = marker_euclidean_distance(m1_x[j], m1_y[j], m2_x[j], m2_y[j])
            marker_dist.append(np.abs(euclidean_dist))
        
        # longitudinal strain (deltaL / L0)
        L0 = marker_dist[0]
        longitudinal_strain = [(L-L0) / L0 for L in marker_dist]

        # append each datasets calculations for plotting
        print(df[f'{i}-data_label'].unique()[0])
        data_labels.append(df[f'{i}-data_label'].unique()[0])
        times.append(time)
        marker_distances.append(marker_dist)
        horizontal_distances.append(np.abs(m2_x-m1_x))
        longitudinal_strains.append(longitudinal_strain)
        elongation_differences.append(np.abs(marker_dist - np.abs(m2_x-m1_x)))

    plot_args = {
        'title': r'Longitudinal Strain of Hydrogel',
        'x_label': time_label,
        'y_label': r'Longitudinal Strain, ${\epsilon}_{l}$',
        'data_label': data_labels,
        'has_legend': True
    }

    if will_save_figures:
        # plot longitudinal strain
        longitudinal_strain_fig, longitudinal_strain_ax = plot_scatter_data(times, longitudinal_strains, plot_args, n_datasets, output_fig_name='longitudinal_strain')

        # plot marker distances
        plot_args['title'] = 'Marker Delta Tracking'
        plot_args['y_label'] = rf'Marker elongation, $\mathit{{\Delta {conversion_units}}}$'
        plot_args['data_label'] = ['Euclidean Distances ' + label for label in data_labels]
        fig, ax = plot_scatter_data(times, marker_distances, plot_args, n_datasets)
        plot_args['data_label'] = ['Horizontal Distances ' + label for label in data_labels]
        plot_scatter_data(times, horizontal_distances, plot_args, n_datasets, fig=fig, ax=ax, output_fig_name='marker_deltas') # plot x distance as well for control/comparison

        # plot difference between euclidean and horizontal differences
        plot_args['title'] = 'Euclidean and Horizontal Differences'
        plot_args['y_label'] = rf'Marker elongation differences, $\mathit{{{conversion_units}}}$'
        plot_args['data_label'] = data_labels
        fig, ax = plot_scatter_data(times, elongation_differences, plot_args, n_datasets, output_fig_name='marker_euclidean_horizontal_differences') # plot x distance as well for control/comparison

    print("Done")

    return times, longitudinal_strains, plot_args, n_datasets


def analyze_necking_point(user_unit_conversion, df=None, will_save_figures=True, chosen_video_data=None):
    """
    Analyzes and plots data related to the necking point of materials, such as the horizontal location and diameter over time,
    as well as calculating radial strain. The function reads from a specified CSV file and optionally saves plots to a directory.

    Details:
        - The function reads necking point data from 'output/Necking_Point_Output.csv' which should have been created during the necking point tracking process.
        - It expects data from a single tracking session; if multiple datasets are found, it raises an error.
        - The function plots three key metrics:
            1. Radial strain of the material over time.
            2. Horizontal location of the necking point over time.
            3. Diameter at the necking point over time.
        - Each of these metrics is calculated and then plotted, using the unit of measure specified by the user.

    Note:
        - Ensure the 'output/Necking_Point_Output.csv' file is formatted correctly with necessary columns for x-location, y-length, and data labels.
        - If saving plots, ensure that the 'figures' directory exists or handle potential FileNotFoundError issues.

    Args:
        user_unit_conversion (tuple): Tuple containing a conversion factor and unit name to convert pixel measurements to a desired unit.
        df (pd.DataFrame, optional): DataFrame containing necking point tracking data. If not provided, data is read from 'output/Necking_Point_Output.csv'.
        will_save_figures (bool, optional): If True, saves generated plots to the 'figures' directory. Defaults to True.

    Returns:
        tuple: Returns a tuple containing time points, radial strains, and plot arguments used for plotting.
    """
    conversion_factor, conversion_units = user_unit_conversion

    print("Analyzing necking point...")
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv("output/Necking_Point_Output.csv") # open csv created/modified from marker tracking process
    print(df.head())

    _, _, n_datasets = get_num_datasets(df)
    data_labels = []
    radial_strains = []
    times = []
    necking_pt_x_locs = []
    necking_pt_lens = []

    if chosen_video_data: # if called by outlier removal tool
        for_range = (chosen_video_data, chosen_video_data+1)
    else:
        for_range = (1, n_datasets+1)

    for i in range(for_range[0], for_range[1]):
        cur_df = get_relevant_columns(df, i).dropna()
        time_col, time_label, _ = get_time_labels(df, i)
        print(time_col)
        time = cur_df[time_col].values
        necking_pt_x = cur_df[f'{i}-x at necking point (px)'].values * conversion_factor
        necking_pt_len = cur_df[f'{i}-y necking distance (px)'].values * conversion_factor
        data_labels.append(cur_df[f'{i}-data_label'].unique()[0])

        # radial strain (deltaR / R0)
        R0 = necking_pt_len[0]
        radial_strains.append([(R - R0) / R0 for R in necking_pt_len])
        
        times.append(time)
        necking_pt_x_locs.append(necking_pt_x)
        necking_pt_lens.append(necking_pt_len)

    plot_args = {
        'title': 'Radial Strain of Hydrogel',
        'x_label': time_label,
        'y_label': r'Radial strain, ${\epsilon}_{r}$',
        'data_label': data_labels,
        'has_legend': True
    }

    if will_save_figures:
        # plot radial strain
        radial_strain_fig, radial_strain_ax = plot_scatter_data(times, radial_strains, plot_args, n_datasets, output_fig_name='radial_strain')

        # plot x location of necking point over time
        plot_args['title'] = 'Necking Point Horizontal Location'
        plot_args['y_label'] = rf'Horizontal location of necking point, $\mathit{{x_{{np}}}}$ ({conversion_units})'
        necking_pt_loc_fig, necking_pt_loc_ax = plot_scatter_data(times, necking_pt_x_locs, plot_args, n_datasets, output_fig_name='necking_point_location')

        # plot diameter at necking point
        plot_args['title'] = r'Diameter of Hydrogel at Necking Point' 
        plot_args['y_label'] = f'Diameter, $\mathit{{D_{{np}}}}$ ({conversion_units})'
        necking_pt_len_fig, necking_pt_len_ax = plot_scatter_data(times, necking_pt_lens, plot_args, n_datasets, output_fig_name='diameter_at_necking_point')

    print("Done")

    return times, radial_strains, plot_args, n_datasets

def poissons_ratio(user_unit_conversion):
    """
    Calculates Poisson's ratio for a material by analyzing longitudinal and radial strains, and additionally computes
    the derivatives of these strains over time. The function integrates data from previous tracking and necking operations,
    ensuring time alignment, and generates plots for both Poisson's ratio and its derivative.

    Details:
        - The function first retrieves longitudinal and radial strain data by calling `analyze_marker_deltas` and `analyze_necking_point`.
        - It verifies that the data from both functions cover the same time period and handles potential discrepancies.
        - A DataFrame is created to align and merge the strains, and Poisson's ratio is calculated as the negative ratio of radial to longitudinal strain.
        - The time derivatives of Poisson's ratio, longitudinal strain, and radial strain are also calculated and plotted.
        - Results are saved to 'output/poissons_ratio.csv', and plots are saved in the 'figures' folder.

    Note:
        - Ensure that both the `analyze_marker_deltas` and `analyze_necking_point` functions are run prior to this with compatible data sets.
        - Ensure proper handling of directories for saving outputs and that the necessary CSV files are formatted correctly.

    Args:
        user_unit_conversion (tuple): Tuple containing a conversion factor and unit name to convert pixel measurements to a desired unit.

    Returns:
        None: The function directly outputs CSV files and generates plots, but does not return any variables.
    """
    conversion_factor, conversion_units = user_unit_conversion
    marker_times, longitudinal_strains, plot_args, _ = analyze_marker_deltas(user_unit_conversion)
    necking_times, radial_strains, _, _ = analyze_necking_point(user_unit_conversion)
    
    print("LABEL", plot_args['data_label'])
    print("Finding Poisson's ratio...")

    # ensure there are the same number of marker and necking datasets
    if not (len(marker_times) == len(necking_times) and len(longitudinal_strains) == len(radial_strains)):
        msg = "Error: Found inequality of marker deltas and necking point datasets.\n\n"+\
        "Please ensure each video used has tracking data for markers and necking point"
        print("Aborted")
        error_popup(msg)
        return
    else:
        n_datasets = len(marker_times)
    
    poissons_df = pd.DataFrame()
    times = []
    poissons_ratios = []
    long_strain_primes = []
    rad_strain_primes = []
    poisson_primes = []

    for i in range(n_datasets):
        # ensure tracking operations previously ran on the same data in the same time range
        if not (marker_times[i][0] == necking_times[i][0] and marker_times[i][-1] == necking_times[i][-1] and len(marker_times[i]) == len(necking_times[i])):
            msg = "Warning: Found discrepancies in marker deltas output and necking point output.\n"+\
            "If this is due to outlier removal in one but not the other, proceed as normal.\n"+\
            "Otherwise please ensure that both marker tracking and necking point detection are run on the same experiment within the same time frame."
            warning_popup(msg)

        # align values, as time values in one may have some missing from the other (from outlier removal)
        marker_df = pd.DataFrame({
            f'{i+1}-time': marker_times[i],
            f'{i+1}-long_strain': longitudinal_strains[i]
        })
        necking_df = pd.DataFrame({
            f'{i+1}-time': necking_times[i],
            f'{i+1}-rad_strain': radial_strains[i]
        })

        time_unit = re.findall(r'\((.*?)\)', plot_args['x_label'])[0] # grab time unit from x axis label for use in y axis labels here

        # calculate poisson's ratio, radial / longitudinal
        cur_poissons_df = pd.merge(marker_df, necking_df, 'inner', f'{i+1}-time')
        times.append(cur_poissons_df[f'{i+1}-time'])
        print(cur_poissons_df)
        # calculates poissons ratio while also avoiding divide by 0 errors
        cur_poissons_df[f'{i+1}-v'] = np.where(cur_poissons_df[f'{i+1}-long_strain'] != 0, -1 * cur_poissons_df[f'{i+1}-rad_strain'] / cur_poissons_df[f'{i+1}-long_strain'], 0)
        poissons_ratios.append(cur_poissons_df[f'{i+1}-v'])
        
        cur_poissons_df[f'{i+1}-d(long_strain)/dt'] = cur_poissons_df[f'{i+1}-long_strain'].diff() / cur_poissons_df[f'{i+1}-time']
        long_strain_primes.append(cur_poissons_df[f'{i+1}-d(long_strain)/dt'])
        cur_poissons_df[f'{i+1}-d(rad_strain)/dt'] = cur_poissons_df[f'{i+1}-rad_strain'].diff() / cur_poissons_df[f'{i+1}-time']
        rad_strain_primes.append(cur_poissons_df[f'{i+1}-d(rad_strain)/dt'])
        cur_poissons_df[f'{i+1}-d(v)/dt'] = cur_poissons_df[f'{i+1}-v'].diff() / cur_poissons_df[f'{i+1}-time']
        poisson_primes.append(cur_poissons_df[f'{i+1}-d(v)/dt'])

        poissons_df = pd.concat([poissons_df, cur_poissons_df], axis=1)

    print(poissons_df)
    poissons_df.to_csv("output/poissons_ratio.csv")

    plot_args['title'] = r"Poisson's Ratio - $\mathit{\nu(t)}$"
    plot_args['y_label'] = r"Poisson's ratio, $\mathit{\nu}$"

    # plot poissons ratio against time
    poisson_fig, poisson_ax = plot_scatter_data(times, poissons_ratios, plot_args, n_datasets, output_fig_name='poissons_ratio')

    # plot rates
    plot_args['title'] = r"Rate of Poissons Ratio - $\dot{\nu} (t)$"
    plot_args['y_label'] = rf"Poisson's ratio rate, $\dot{{\nu}}(t)$ $(\frac{{1}}{{{time_unit}}})$"
    poisson_prime_fig, poisson_prime_ax = plot_scatter_data(times, poisson_primes, plot_args, n_datasets, output_fig_name='poissons_ratio_prime')

    plot_args['title'] = r"Rate of Longitudinal Strain - $\dot{\epsilon}_l (t)$"
    plot_args['y_label'] = rf"Longitudinal strain rate, $\dot{{\epsilon}}_l (t) $ $(\frac{{1}}{{{time_unit}}})$"
    long_strain_prime_fig, long_strain_prime_ax = plot_scatter_data(times, long_strain_primes, plot_args, n_datasets, output_fig_name='long_strain_prime')

    plot_args['title'] = r"Rate of Radial Strain - $\dot{\epsilon}_r (t)$"
    plot_args['y_label'] = rf"Radial strain rate $\mathit{{\dot{{\epsilon}}_r (t)}}$ $(\frac{{1}}{{{time_unit}}})$"
    rad_strain_prime_fig, rad_strain_prime_ax = plot_scatter_data(times, rad_strain_primes, plot_args, n_datasets, output_fig_name='rad_strain_prime')

    print("Done")

def marker_velocity(user_unit_conversion, df=None, will_save_figures=True, chosen_video_data=None):
    """
    Calculates and plots the velocities of tracked markers based on their positional data over time. Optionally performs
    Fourier transform to analyze the frequency components of these velocities. The function supports multiple datasets
    or multiple trackers within a single dataset and handles discrepancies in data lengths.

    Details:
        - Retrieves marker positional data from 'output/Tracking_Output.csv' unless a DataFrame is directly provided.
        - Determines whether the data pertains to multiple trackers or multiple video datasets and adjusts processing accordingly.
        - Calculates instantaneous velocities in both x and y directions and their magnitudes.
        - Optionally performs a Fourier transform on the velocity data to analyze frequency components and plots these analyses.
        - Saves processed velocity data to 'output/marker_velocity.csv' and plots to the 'figures' folder if enabled.

    Note:
        - Ensure the input DataFrame or CSV file is formatted correctly with necessary columns for x and y positions.
        - Ensure the directory for saving outputs exists if saving plots or data.
        - The function handles data from either multiple trackers in a single dataset or multiple datasets but expects a consistent format across datasets.

    Args:
        user_unit_conversion (tuple): Contains the conversion factor and units to convert pixel measurements to a desired unit.
        df (pd.DataFrame, optional): DataFrame containing tracking data. If not provided, data is read from 'output/Tracking_Output.csv'.
        will_save_figures (bool, optional): If True, saves generated plots to the 'figures' directory. Defaults to True.
        chosen_video_data (int, optional): Specifies a particular dataset to analyze if multiple datasets are present.

    Returns:
        tuple: Contains time arrays, velocity arrays for each tracker or dataset, plot arguments, and the number of plots or datasets processed.
    """
    
    print("Finding Marker Velocity...")

    conversion_factor, conversion_units = user_unit_conversion
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv("output/Tracking_Output.csv") # open csv created/modified from marker tracking process
    print(df.head())

    if chosen_video_data is None:
        chosen_video_data = 1
    
    time_col, time_label, time_unit = get_time_labels(df, chosen_video_data)
    _, _, num_tracker_datasets = get_num_datasets(df) # get num datasets
    if chosen_video_data is None:
        tracker_cols = [col for col in df.columns if col.__contains__('Tracker')]
        n_trackers = df[tracker_cols[0]].unique().shape[0] # get number of trackers
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
        n_plots = n_trackers
        num_sets = n_trackers
    elif data_multiplicity_type == DataMultiplicity.VIDEOS:
        n_plots = num_tracker_datasets
        num_sets = num_tracker_datasets

    # grab relevant values from df
    print('asdfasdf\n', df[time_col])
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

        vel_df[f'{n+1}-x_velocity_tracker'] = vel_x
        vel_df[f'{n+1}-y_velocity_tracker'] = vel_y
        vel_df[f'{n+1}-magnitude_velocity_tracker'] = vel_mag

        # fourier transform
        N = len(vel_mag)
        vel_fft = np.fft.fft(vel_mag) # amplitude (pixels/Hz)
        freqs = np.fft.fftfreq(len(vel_mag), np.mean(dt)) # frequencies
        tracker_amplitudes.append(vel_fft)
        tracker_frequencies.append(freqs)

    # save intermediate calculations
    vel_df.to_csv("output/marker_velocity.csv", index=False)

    # plot
    plot_args = {
        'title': r'Cell Velocity',
        'x_label': time_label,
        'y_label': rf'Magnitude of instantaneous cell velocity, $|\mathit{{v}}|$ $(\frac{{{conversion_units}}}{{{time_unit}}})$',
        'data_label': data_labels,
        'has_legend': True,

    }
    if will_save_figures:
        # plot marker velocity
        vel_fig, vel_ax = plot_scatter_data(times, tracker_velocities, plot_args, n_plots, output_fig_name='marker_velocity')

        # plot fourier transform of marker distances
        plot_args['title'] = 'Marker Velocity FFT'
        plot_args['y_label'] = f'({conversion_units}/Hz)'
        plot_args['x_label'] = 'Hz'
        plot_fft(tracker_frequencies, tracker_amplitudes, N, n_plots, plot_args)
        
    print("Done")
    return times, tracker_velocities, plot_args, n_plots

def velocity_boxplot(conditions, user_unit_conversion):
    """
    Generates a boxplot of cell velocities for specified conditions. This function reads cell velocity data,
    computes averages, and displays a boxplot to compare velocities across different experimental conditions.

    Details:
        - The function reads velocity data from 'output/Tracking_Output.csv'.
        - It filters data based on conditions specified by the user and calculates average velocities for each condition.
        - A boxplot is generated showing the distribution of average velocities across conditions, with additional scatter points
          representing individual dataset averages.
        - The plot is saved to the 'figures' directory.

    Note:
        - Ensure that the CSV file contains the necessary data columns formatted correctly.
        - The conditions list should match the data labels in the dataset to correctly filter the data.
        - Ensure the 'figures' directory exists to prevent a FileNotFoundError when saving the plot.

    Args:
        conditions (list): A list of strings representing the conditions under which the data was collected, used to filter datasets.
        user_unit_conversion (tuple): Contains the conversion factor and units to convert pixel measurements to a desired unit.

    Returns:
        None: The function directly outputs a boxplot to a file but does not return any variables.
    """

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

        if not cur_condition_avg_velocities:
            msg = f"WARNING: Found no data for condition: {condition}.\nPlease ensure the conditions were entered correctly"
            warning_popup(msg)
        condition_avg_velocities[condition] = cur_condition_avg_velocities
    print(condition_avg_velocities)
    
    # err checking make sure all specified conditions contained data

    data_lists = [condition_avg_velocities[condition] for condition in conditions]
    
    fig, ax = plt.subplots()
    boxplot = ax.boxplot(data_lists, patch_artist=True)
    for patch in boxplot['boxes']: # customize boxes of plot
        patch.set_facecolor('lightblue')  
        patch.set_alpha(0.4) 
        patch.set_linewidth(1)

    # The edge color of the boxes is set after the boxplot is created
    for box in boxplot['boxes']:
        box.set_edgecolor('black')
        box.set_alpha(1)

    # plotting scatter points on boxplot
    for i, condition in enumerate(conditions, start=1):  # count from 1 to match boxplot positions
        mean_velocities = condition_avg_velocities[condition] # grab avg velocity previously calculated
        x_jitter = np.random.normal(i, 0.025, size=len(mean_velocities)) # some x jitter so points don't overlap
        ax.scatter(x_jitter, mean_velocities, color='darkblue', zorder=3, marker='D', s=20, alpha=0.8) # plot mean value in the correct box with some x jitter

    with open("plot_opts/plot_customizations.json", 'r') as plot_customs_file:
        plot_customs = json.load(plot_customs_file)
    font = plot_customs['font']

    tick_pos = list(range(1, len(condition_avg_velocities) + 1))
    plt.xticks(tick_pos, list(condition_avg_velocities.keys()), fontsize=plot_customs['value_text_size'], fontfamily=font)
    plt.yticks(fontsize=plot_customs['value_text_size'], fontfamily=font) 
    plt.title("Average Cell Velocities Across Multiple Conditions")
    plt.ylabel(rf'Average Magnitude of cell velocity, $|\mathit{{v}}|$ $(\frac{{{conversion_units}}}{{{time_unit}}})$')
    plt.xlabel(time_label)
    plt.tick_params(axis='both', direction=plot_customs['tick_dir'])
    plt.tight_layout()
    
    plt.savefig(f"figures/marker_velocity_boxplot.{plot_customs['fig_format']}", dpi=plot_customs['fig_dpi'])


def marker_distance(user_unit_conversion):
    """
    Calculates and plots the root mean square displacement of markers over time. This function reads tracking data,
    processes it to compute the RMS displacement for each tracker or dataset, and plots the results.

    Details:
        - The function reads tracking data from 'output/Tracking_Output.csv'.
        - It identifies whether data are from multiple trackers or multiple video datasets and calculates RMS displacement accordingly.
        - Displacement calculations are based on changes in x and y positions over time, adjusted for a conversion factor to a specified unit.
        - The plot is generated to visually represent the displacement over time for each dataset or tracker.

    Note:
        - Ensure the input CSV file is properly formatted with necessary columns for x and y positions.
        - RMS calculations are sensitive to noise in the data, which can affect the results.
        - Ensure directories for saving outputs exist if saving plots or data.

    Args:
        user_unit_conversion (tuple): Contains the conversion factor and units to convert pixel measurements to a desired unit.

    Returns:
        None: The function directly outputs a plot but does not return any variables.
    """

    print("Finding Marker RMS Distance...")
    conversion_factor, conversion_units = user_unit_conversion
    df = pd.read_csv("output/Tracking_Output.csv") # open csv created/modified from marker tracking process
    print(df.head())
    
    time_col, time_label, time_unit = get_time_labels(df)
    _, _, num_tracker_datasets = get_num_datasets(df) # get num datasets
    n_trackers = df['1-Tracker'].unique().shape[0] # get number of trackers
    data_labels = []
    rms_disps = []
    times = []
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
            times.append(time[:-1])

    elif data_multiplicity_type == DataMultiplicity.VIDEOS:
        n_plots = num_tracker_datasets
        label = 'Video'
        num_sets = num_tracker_datasets
        for dataset in range(num_tracker_datasets):
            x = df[f'{dataset+1}-x (px)'].values * conversion_factor
            y = df[f'{dataset+1}-y (px)'].values * conversion_factor
            time_col, _, _ = get_time_labels(df, dataset+1)
            time = df[time_col].unique()

            data_labels.append(df[f'{dataset+1}-data_label'].unique()[0])
            rms_disps.append(rms_displacement(np.diff(x), np.diff(y)))
            times.append(time[:-1])

    plot_args = {
        'title': 'Root mean square Displacement',
        'x_label': time_label,
        'y_label': r'Root mean square Displacement, $\mathit{D_{rms}}$' + rf' ({conversion_units})    ',
        'data_label': data_labels,
        'has_legend': True
    }
    # plot marker velocity
    disp_fig, disp_ax = plot_scatter_data(times, rms_disps, plot_args, n_plots, output_fig_name='marker_RMS_displacement')

def single_marker_spread(user_unit_conversion, df=None, will_save_figures=True, chosen_video_data=None):
    """
    Analyzes and visualizes changes in tracked surface areas over time. This function processes the previously tracked surface
    area data to compute metrics such as area growth rate and plots these changes over time to facilitate analysis.

    Details:
        - The function reads surface area tracking data from 'output/Tracking_Output.csv', which should contain surface area data
          across multiple frames or conditions.
        - It calculates the percentage change in surface area relative to the initial area and other relevant metrics.
        - Results are visualized in a plot that displays changes in surface area over time, highlighting any significant growth or contraction.
        - Optionally, the function can compare surface area changes across different experimental conditions if such data is available.

    Note:
        - Ensure that the input CSV file is formatted correctly with necessary columns for surface areas and possibly conditions.
        - Accurate initial area measurements are crucial for meaningful percentage change calculations.
        - Ensure directories for saving outputs exist if saving plots or data.

    Args:
        user_unit_conversion (tuple): Contains the conversion factor and units to convert pixel measurements to a desired unit.

    Returns:
        None: The function directly outputs a plot but does not return any variables.
    """
    print("Finding Marker Spread...")
    conversion_factor, conversion_units = user_unit_conversion

    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv("output/Surface_Area_Output.csv") # open csv created/modified from marker tracking process
    print(df.head())

    if chosen_video_data is None:
        chosen_video_data = 1
    
    time_col, time_label, time_unit = get_time_labels(df, chosen_video_data)
    _, _, num_tracker_datasets = get_num_datasets(df)
    data_multiplicity_type = check_num_trackers_with_num_datasets(1, num_tracker_datasets)

    surface_areas = []
    data_labels = []
    times = []
    for i in range(num_tracker_datasets):
        time = df[time_col].values
        if data_multiplicity_type == DataMultiplicity.SINGLE:
            if chosen_video_data is None:
                chosen_video_data = 1
            surface_area = df[f'{chosen_video_data}-cell surface area (px^2)'].values * conversion_factor
            data_labels.append(df[f'{chosen_video_data}-data_label'].unique()[0])
        else:
            surface_area = df[f'{i+1}-cell surface area (px^2)'].values * conversion_factor
            data_labels.append(df[f'{i+1}-data_label'].unique()[0])

        times.append(time)
        surface_areas.append(surface_area)

    # plot
    plot_args = {
        'title': r'Cell Surface Area Spread',
        'x_label': time_label,
        'y_label': rf'Projected cell area, $\mathit{{A}}$ ' + rf'({conversion_units})$^{{2}}$',
        'data_label': data_labels,
        'has_legend': True
    }

    if will_save_figures:
        # plot marker velocity
        area_fig, area_ax = plot_scatter_data(times, surface_areas, plot_args, num_tracker_datasets, output_fig_name='marker_surface_area')

    return times, surface_areas, plot_args, num_tracker_datasets

if __name__=='__main__':
    analyze_marker_deltas()