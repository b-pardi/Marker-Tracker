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
import os
from collections import defaultdict

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

def rms_distance(x, y):
    """
    calculate the root mean square distance of x, y points over time
    characterize the magnitude of fluctuations/movements of the marker

    Args:
        dx (_type_): change in x positions
        dy (_type_): change in y positions
    """    
    dx_sq = (np.diff(x))**2
    dy_sq = (np.diff(y))**2

    rms_dist = np.sqrt(np.cumsum(dx_sq + dy_sq) / (np.arange( len(dx_sq) ) + 1 ))
    return rms_dist

def rms_displacement(x, y):
    """calculate the root mean square distance of x, y points over time
    characterize the magnitude of fluctuations/movements of the marker

    Args:
        dx (_type_): change in x positions
        dy (_type_): change in y positions
    """  
    r = np.sqrt(x**2 + y**2)
    rms_disp = np.sqrt( np.mean( np.diff(r) ) )
    return rms_disp

def get_time_labels(df, col_num=1):
    ''' util function to return the appropriate strings for:
        - time column in dataframe
        - plot x axis label
        - specific time unit

    Expects dataframe in the formatting of the tracking methods output.
    '''
    time_col = f'{col_num}-' + (df.filter(like='Time').columns[0]).split('-', 1)[1] # accounts for any kind of units in time col
    if 'min' in time_col:
        time_label = r'Time, $\mathit{t}$ (min)'
        time_unit = TimeUnits.MINUTES.value
    elif 'hr' in time_col:
        time_label = r'Time, $\mathit{t}$ (hr)'
        time_unit = TimeUnits.HOURS.value
    else:
        time_label = r'Time, $\mathit{t}$ (s)'
        time_unit = TimeUnits.SECONDS.value

    return time_col, time_label, time_unit

def check_tracker_data_lengths(df, n_trackers):
    """
    Verifies the consistency of data lengths across multiple trackers within the datasets contained in a DataFrame.
    This function ensures that all trackers have recorded data for the same number of entries.

    Details:
        - Iterates through each dataset and tracker, collecting the number of data entries (rows) for each.
        - Checks if the minimum and maximum data lengths across all trackers are equal.
        - If discrepancies are found in data lengths, an error message is raised, indicating a potential issue in the tracking or data collection process.

    Note:
        - Expects dataframe in the formatting of the tracking methods output.

    Args:
        df (pd.DataFrame): DataFrame containing tracker data, with trackers and datasets identified by columns.
        n_trackers (int): The number of trackers used in the data collection, expected to be consistent across all datasets.

    Returns:
        None: The function does not return a value but raises an exception if discrepancies in data lengths are found.
    """
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
        raise Exception(msg)

def get_num_datasets(df):
    """
    Calculates the number of datasets represented within a DataFrame based on the naming convention of its columns.
    Expects dataframe in the formatting of the tracking methods output.

    Args:
        df (pd.DataFrame): DataFrame containing data with columns named according to dataset identifiers.

    Returns:
        tuple: Returns the starting identifier, ending identifier, and total number of datasets (int, int, int).
    """
    col_start = int(df.columns[0][0])
    col_end = int(df.columns[-1][0])
    n_sets = col_end - col_start + 1

    return col_start, col_end, n_sets

def get_relevant_columns(df, col_num):
    """
    Extracts columns relevant to a specific dataset number from a DataFrame, based on the naming convention of its columns.

    Details:
        - Filters columns that begin with a specific dataset number followed by a hyphen, indicating dataset-specific data.

    Args:
        df (pd.DataFrame): DataFrame from which to filter columns.
        col_num (int): The dataset number used to identify relevant columns.

    Returns:
        pd.DataFrame: A DataFrame containing only the columns relevant to the specified dataset number.
    """
    relevant_cols = [col for col in df.columns if col.startswith(f"{col_num}-")]
    return df[relevant_cols]

def check_num_trackers_with_num_datasets(n_trackers, num_tracker_datasets):
    """
    Determines the multiplicity of data sources (trackers and datasets) and provides a classification based on the analysis suitability.

    Details:
        - Checks if there are multiple trackers and multiple datasets, which might complicate analysis unless handled correctly.
        - Identifies four possible scenarios: multiple videos (datasets), multiple trackers, both, or single tracker and single dataset.

    Args:
        n_trackers (int): The number of trackers used in the data collection.
        num_tracker_datasets (int): The number of distinct video datasets involved.

    Returns:
        enum: An enum value indicating the type of data multiplicity (SINGLE, TRACKERS, VIDEOS, BOTH).
    """
    data_multiplicity_type = 0
    if num_tracker_datasets > 1 and n_trackers > 1:
        msg = "ERROR: When using the append feature, only use one tracker per tracking operation.\n"+\
        "If using multiple trackers, only use one video tracking at a time for analysis\n"
        error_popup(msg)
        raise Exception(msg)
    
    elif num_tracker_datasets > 1 and n_trackers == 1:
        data_multiplicity_type = DataMultiplicity.VIDEOS
    elif n_trackers > 1 and num_tracker_datasets == 1:
        data_multiplicity_type = DataMultiplicity.TRACKERS
    elif n_trackers == 1 and num_tracker_datasets == 1:
        data_multiplicity_type = DataMultiplicity.SINGLE

    return data_multiplicity_type

def find_interest_column_and_type(df):
    """
    Identifies the column of interest for analysis and determines the corresponding type of analysis based on predefined
    column names. This function is designed to automate the selection of data columns from various datasets for subsequent
    analysis and visualization.

    Details:
        - The function searches through the DataFrame columns to match against a predefined list of interest columns that
          correspond to different types of analyses, such as velocity, displacement, or surface area.
        - Each column of interest is associated with a specific type of analysis, which is determined during the process.
        - This utility supports multiple analysis types and ensures that the data is correctly identified for accurate processing.

    Note:
        - Expects dataframe in the formatting of the tracking methods output.
        - If no matching column is found, the function raises a ValueError, so handle exceptions appropriately where this function is called.

    Args:
        df (pd.DataFrame): The DataFrame from which to identify the column of interest for further analysis.

    Returns:
        tuple: A tuple containing the name of the column of interest (str) and the corresponding type of analysis (enum), if found.
    """
    interest_cols = [
        'v', # poissons ratio df
        'velocity', # velocity df
        'displacement', # displacement df
        'distance', # distance df
        'surface_area' # surface spread
    ]

    analysis_types = [
        AnalysisType.POISSONS_RATIO_CSV,
        AnalysisType.VELOCITY,
        AnalysisType.DISTANCE,
        AnalysisType.DISPLACEMENT,
        AnalysisType.SURFACE_AREA
    ]

    # get column names of df, excluding the dataset index
    df_cols = [col.split('-', 1)[1] for col in df.columns]
    df_unique_cols = set(df_cols)
    print(df_unique_cols, interest_cols)
    y_col, analysis_type = None, None
    for df_col in df_unique_cols:
        for i, interest_col in enumerate(interest_cols):
            if interest_col == df_col:
                print("found ", interest_col)
                y_col = interest_col
                analysis_type = analysis_types[i]
                break

    if not y_col:
        raise ValueError("No column in any output df's matched expected boxplot y columns")

    return y_col, analysis_type

def get_plot_args(analysis_type, **kwargs):
    '''
    Simple util function to take in the analysis type and plot_arg values and return the appropriate dictionary
    This was just cleaner than defining these in every function
    '''

    time_label = kwargs.get('time_label', '')
    data_labels = kwargs.get('data_labels', '')
    conversion_units = kwargs.get('conversion_units', '')
    time_unit = kwargs.get('time_unit', '')
    locator_type = kwargs.get('locator_type', LocatorType.BBOX).value.capitalize()
    
    plot_args = {}

    if analysis_type == AnalysisType.MARKER_DELTAS:
        plot_args = {
            'title': r'Longitudinal Strain of Hydrogel',
            'x_label': time_label,
            'y_label': r'Longitudinal Strain, ${\epsilon}_{l}$',
            'data_label': data_labels,
            'has_legend': True
        }
    if analysis_type == AnalysisType.NECKING_POINT:
        plot_args = {
            'title': 'Radial Strain of Hydrogel',
            'x_label': time_label,
            'y_label': r'Radial strain, ${\epsilon}_{r}$',
            'data_label': data_labels,
            'has_legend': True
        }
    if analysis_type == AnalysisType.POISSONS_RATIO or analysis_type == AnalysisType.POISSONS_RATIO_CSV:
        plot_args = {
            'title': r"Poisson's Ratio - $\mathit{\nu(t)}$",
            'x_label': time_label,
            'y_label': r"Poisson's ratio, $\mathit{\nu}$",
            'data_label': data_labels,
            'has_legend': True
        }
    if analysis_type == AnalysisType.VELOCITY:
        plot_args = {
            'title': f'{locator_type} Velocity',
            'x_label': time_label,
            'y_label': rf'Magnitude of instantaneous cell velocity, $|\mathit{{v}}|$ $(\frac{{{conversion_units}}}{{{time_unit}}})$',
            'data_label': data_labels,
            'has_legend': True,
        }
    if analysis_type == AnalysisType.DISPLACEMENT:
        plot_args = {
            'title': f'{locator_type} Displacement',
            'x_label': time_label,
            'y_label': r'Displacement, $\mathit{\Delta\vec{r}}$' + rf' ({conversion_units})    ',
            'data_label': data_labels,
            'has_legend': True
        }
    if analysis_type == AnalysisType.DISTANCE:
        plot_args = {
            'title': f'{locator_type} Distance',
            'x_label': time_label,
            'y_label': r'Distance, $\mathit{D}$' + rf' ({conversion_units})    ',
            'data_label': data_labels,
            'has_legend': True
        }
    if analysis_type == AnalysisType.SURFACE_AREA:
        plot_args = {
            'title': r'Cell Surface Area Spread',
            'x_label': time_label,
            'y_label': rf'Projected cell area, $\mathit{{A}}$ ' + rf'({conversion_units})$^{{2}}$',
            'data_label': data_labels,
            'has_legend': True
        }

    return plot_args

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
        fig, ax = plt.subplots(figsize=(8,6))
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
    '''DEPRECATED'''
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
    '''DEPRECATED'''
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

def analyze_marker_deltas(conversion_factor, conversion_units, df=None, will_save_figures=True, chosen_video_data=None):
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

    if chosen_video_data: # if called by outlier removal or data selector tool
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
        time = m1_df[time_col]
        m1_x = m1_df[f'{i}-x (px)'].values * conversion_factor
        m1_y = m1_df[f'{i}-y (px)'].values * conversion_factor
        m2_x = m2_df[f'{i}-x (px)'].values * conversion_factor
        m2_y = m2_df[f'{i}-y (px)'].values * conversion_factor
        print(len(time), len(m1_x), len(m2_x), len(m2_x), len(m2_y))
        
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

    plot_args = get_plot_args(AnalysisType.MARKER_DELTAS, time_label=time_label, data_labels=data_labels)


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


def analyze_necking_point(conversion_factor, conversion_units, df=None, will_save_figures=True, chosen_video_data=None):
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
        conversion_factor (float): The factor to convert pixel measurements to a desired unit.
        conversion_units (str): The units resulting from the conversion for labeling and output.
        df (pd.DataFrame, optional): DataFrame containing necking point tracking data. If not provided, data is read from 'output/Necking_Point_Output.csv'.
        will_save_figures (bool, optional): If True, saves generated plots to the 'figures' directory. Defaults to True.

    Returns:
        tuple: Returns a tuple containing time points, radial strains, and plot arguments used for plotting.
    """

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

    plot_args = get_plot_args(AnalysisType.NECKING_POINT, time_label=time_label, data_labels=data_labels)

    if will_save_figures:
        # plot radial strain
        radial_strain_fig, radial_strain_ax = plot_scatter_data(times, radial_strains, plot_args, n_datasets, output_fig_name='radial_strain')

        # plot x location of necking point over time
        plot_args['title'] = 'Necking Point Horizontal Location'
        plot_args['y_label'] = rf'Horizontal location of necking point, $\mathit{{x_{{np}}}}$ ({conversion_units})'
        necking_pt_loc_fig, necking_pt_loc_ax = plot_scatter_data(times, necking_pt_x_locs, plot_args, n_datasets, output_fig_name='necking_point_location')

        # plot diameter at necking point
        plot_args['title'] = r'Diameter of Hydrogel at Necking Point' 
        plot_args['y_label'] = rf'Diameter, $\mathit{{D_{{NP}}}}$ ({conversion_units})'
        necking_pt_len_fig, necking_pt_len_ax = plot_scatter_data(times, necking_pt_lens, plot_args, n_datasets, output_fig_name='diameter_at_necking_point')

    print("Done")

    return times, radial_strains, plot_args, n_datasets

def poissons_ratio(conversion_factor, conversion_units):
    """
    Calculates Poisson's ratio for a material by analyzing longitudinal and radial strains, and additionally computes
    the derivatives of these strains over time. The function integrates data from previous tracking and necking operations,
    ensuring time alignment, and generates plots for both Poisson's ratio and its derivative.

    Details:
        - The function first retrieves longitudinal and radial strain data by calling `analyze_marker_deltas` and `analyze_necking_point`.
        - It verifies that the data from both functions cover the same time period and handles potential discrepancies.
        - A DataFrame is created to align and merge the strains, and Poisson's ratio is calculated as the negative ratio of radial to longitudinal strain.
        - The time derivatives of Poisson's ratio, longitudinal strain, and radial strain are also calculated and plotted.
        - Results are saved to 'output/poissons_ratio_calc.csv', and plots are saved in the 'figures' folder.

    Note:
        - Ensure that both the `analyze_marker_deltas` and `analyze_necking_point` functions are run prior to this with compatible data sets.
        - Ensure proper handling of directories for saving outputs and that the necessary CSV files are formatted correctly.

    Args:
        conversion_factor (float): The factor to convert pixel measurements to a desired unit.
        conversion_units (str): The units resulting from the conversion for labeling and output.
        will_calculate (Bool): determines if will calculate from recorded data, or if read from csv (False when removing points in OutlierRemoval)
        
    Returns:
        None: The function directly outputs CSV files and generates plots, but does not return any variables.
    """
    marker_times, longitudinal_strains, plot_args, _ = analyze_marker_deltas(conversion_factor, conversion_units)
    necking_times, radial_strains, _, _ = analyze_necking_point(conversion_factor, conversion_units)
    
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

    temp_df = pd.read_csv("output/Necking_Point_Output.csv")
    time_col, time_label, _ = get_time_labels(temp_df)
    time_col = time_col[2:]

    for i in range(n_datasets):
        # align values, as time values in one may have some missing from the other (from outlier removal)
        marker_df = pd.DataFrame({
            f'{i+1}-{time_col}': marker_times[i],
            f'{i+1}-long_strain': longitudinal_strains[i]
        })
        necking_df = pd.DataFrame({
            f'{i+1}-{time_col}': necking_times[i],
            f'{i+1}-rad_strain': radial_strains[i]
        })

        time_unit = re.findall(r'\((.*?)\)', plot_args['x_label'])[0] # grab time unit from x axis label for use in y axis labels here

        # calculate poisson's ratio, radial / longitudinal
        cur_poissons_df = pd.merge(marker_df, necking_df, 'inner', f'{i+1}-{time_col}')
        times.append(cur_poissons_df[f'{i+1}-{time_col}'])
        cur_poissons_df[f'{i+1}-data_label'] = plot_args['data_label'][i]
        print(cur_poissons_df)
        # calculates poissons ratio while also avoiding divide by 0 errors
        cur_poissons_df[f'{i+1}-v'] = np.where(cur_poissons_df[f'{i+1}-long_strain'] != 0, -1 * cur_poissons_df[f'{i+1}-rad_strain'] / cur_poissons_df[f'{i+1}-long_strain'], 0)
        poissons_ratios.append(cur_poissons_df[f'{i+1}-v'])
        
        cur_poissons_df[f'{i+1}-d(long_strain)/dt'] = cur_poissons_df[f'{i+1}-long_strain'].diff() / cur_poissons_df[f'{i+1}-{time_col}']
        long_strain_primes.append(cur_poissons_df[f'{i+1}-d(long_strain)/dt'])
        cur_poissons_df[f'{i+1}-d(rad_strain)/dt'] = cur_poissons_df[f'{i+1}-rad_strain'].diff() / cur_poissons_df[f'{i+1}-{time_col}']
        rad_strain_primes.append(cur_poissons_df[f'{i+1}-d(rad_strain)/dt'])
        cur_poissons_df[f'{i+1}-d(v)/dt'] = cur_poissons_df[f'{i+1}-v'].diff() / cur_poissons_df[f'{i+1}-{time_col}']
        poisson_primes.append(cur_poissons_df[f'{i+1}-d(v)/dt'])

        poissons_df = pd.concat([poissons_df, cur_poissons_df], axis=1)

    print(poissons_df)
    poissons_df.to_csv("output/poissons_ratio.csv")

    plot_args['title'] = r"Poisson's Ratio - $\mathit{\nu(t)}$"
    plot_args['y_label'] = r"Poisson's ratio, $\mathit{\nu}$"

    # plot poissons ratio against time
    poisson_fig, poisson_ax = plot_scatter_data(times, poissons_ratios, plot_args, n_datasets, output_fig_name='poissons_ratio_calc')

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

def poissons_ratio_csv(df=None, will_save_figures=True, chosen_video_data=None):
    """
    Performs similar function to poissons_ratio(), but reads from the csv that that function creates
    This is for two reasons:
        - Outlier removal tool needs the points from the poissons ratio csv file directly as opposed to removing points and recalculating everytime
            - This is because removing a point in the poissons plot will not directly correspond to that point being removed in the marker deltas and necking point files
        - It is faster to not have to run the calculations everytime
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv('output/poissons_ratio.csv')
        
    n_datasets = int(df.columns[-1].split('-', 1)[0])
    if not chosen_video_data:
        for_range = (1, n_datasets+1)
    else:
        for_range = (chosen_video_data, chosen_video_data+1)

    times, poissons_ratios, data_labels = [], [], []
    for i in range(for_range[0], for_range[1]):
        cur_df = get_relevant_columns(df, i).dropna()
        time_col, time_label, _ = get_time_labels(df, i)
        time = cur_df[time_col].values
        data_labels.append(cur_df[f'{i}-data_label'].unique()[0])
        ratios = cur_df[f'{i}-v'].values

        times.append(time)
        poissons_ratios.append(ratios)
        print(len(time), time, '\n', len(ratios), ratios)

    plot_args = get_plot_args(AnalysisType.POISSONS_RATIO, time_label=time_label, data_labels=data_labels)

    if will_save_figures:
        poisson_fig, poisson_ax = plot_scatter_data(times, poissons_ratios, plot_args, n_datasets, output_fig_name='poissons_ratio_csv')

    print("Done")
    return times, poissons_ratios, plot_args, n_datasets


def boxplot_time_ranges(df, times, labels, conversion_units):
    """
    Groups data by specified time ranges, calculates the average of the data for each label within these ranges, 
    and generates boxplots to compare these averages across the different time ranges.

    Details:
        - The function first determines which data points fall within each specified time range.
        - It then computes the average of these data points for each label within the time ranges.
        - A boxplot is generated for each time range, displaying the distribution of these averages, allowing for
          comparison across different periods.
        - This function is designed to handle any data that can be segmented into time ranges and is flexible to
          various types of analyses depending on the data column provided.

    Note:
        - Expects dataframe in the formatting of the tracking methods output.
        - Ensure directories for saving outputs exist if saving plots to avoid FileNotFoundError.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be plotted, which includes at least a time column and data columns.
        time_ranges (list of tuples): A list where each tuple represents a start and end time defining a range.
        conversion_units (str): The units for the data being plotted, used for labeling the y-axis.

    Returns:
        None: This function does not return any values but outputs a boxplot to a file.
    """
    print(df, times, labels)

    # find column of interest (the y column that will be averaged across time ranges)
    interest_column, analysis_type = find_interest_column_and_type(df)
    df_label_columns = [col for col in df.columns if 'data_label' in col]
    time_col, time_label, time_unit = get_time_labels(df)
    time_ranges_averages = defaultdict(list)
    time_range_labels = [f"{float(t0)}-{float(tf)}" for t0, tf in times] # turn list of tuples into list of strings for boxplot labels
    data_lists = []

    for t0,tf in times: # iterate through tuple time ranges
        t0, tf = float(t0), float(tf)
        cur_range_interest_values = [] # list of all measured values within cur_time range to be averaged for a box
        for label in labels: # iterate through all labels selected for analysis
            for col in df_label_columns: # find dataset corresponding to current label
                if df[col].str.contains(label).any():
                    dataset_num = int(col.split('-', 1)[0])
                    time_col, _, _ = get_time_labels(df, dataset_num)
                    cur_range_df = df[(df[time_col] >= t0) & (df[time_col] <= tf)]
                    cur_data = cur_range_df[f'{dataset_num}-{interest_column}']
                    cur_range_interest_values.append(cur_data)
            avg = np.mean(cur_range_interest_values)
            time_ranges_averages[f'{t0}-{tf}'].append(avg)
        data_lists.append(time_ranges_averages[f'{t0}-{tf}'])
    print(time_ranges_averages)

    fig, ax = plt.subplots()
    boxplot = ax.boxplot(data_lists, patch_artist=True)

    for patch in boxplot['boxes']:
        patch.set_facecolor('white')

    for median in boxplot['medians']:
        median.set_color('black') 

    for i, time_range in enumerate(time_range_labels, start=1):  # count from 1 to match boxplot positions
        means = time_ranges_averages[time_range] # grab avg velocity previously calculated
        print( means, time_range_labels, time_ranges_averages)
        x_jitter = np.random.normal(i, 0.025, size=len(means)) # some x jitter so points don't overlap
        ax.scatter(x_jitter, means, color='darkblue', zorder=3, marker='D', s=20, alpha=0.8) # plot mean value in the correct box with some x jitter

    plot_args = get_plot_args(analysis_type, time_label=time_label, conversion_units=conversion_units, time_unit=time_unit)

    with open("plot_opts/plot_customizations.json", 'r') as plot_customs_file:
        plot_customs = json.load(plot_customs_file)
    font = plot_customs['font']

    tick_pos = list(range(1, len(time_range_labels) + 1))
    plt.xticks(tick_pos, time_range_labels, fontsize=plot_customs['value_text_size'], fontfamily=font)
    plt.yticks(fontsize=plot_customs['value_text_size'], fontfamily=font) 
    plt.title(plot_args['title'], fontsize=plot_customs['title_text_size'], fontfamily=font)
    plt.xlabel(plot_args['x_label'], fontsize=plot_customs['label_text_size'], fontfamily=font)
    plt.ylabel(plot_args['y_label'], fontsize=plot_customs['label_text_size'], fontfamily=font)
    plt.tick_params(axis='both', direction=plot_customs['tick_dir'])
    plt.tight_layout()
    plt.savefig(f"figures/{interest_column}_time_ranges_boxplot.{plot_customs['fig_format']}", dpi=plot_customs['fig_dpi'])

    print("Done")


def boxplot_conditions(df, conditions_dict, conversion_units):
    """
    Aggregates data by user-defined conditions and generates boxplots for each condition. This function
    calculates the average values of specified data columns associated with each condition and presents 
    them in a comparative visualization.

    Details:
        - This function identifies relevant data columns based on user-specified conditions and data labels in the DataFrame.
            - Data labels correspond to label identifiers also spec'd by user when recording data.
        - It computes the average of the values for these columns, grouping them according to the conditions provided.
        - A boxplot is then generated to display the distribution of these averages across the specified conditions.
        - The function supports customization of the plot's appearance and saves the plot to a specified directory.

    Note:
        - find_interest_column_and_type() is used to determine which analysis type is selected and find the appropriate y column of interest.
        - Ensure that the 'figures' directory exists to prevent a FileNotFoundError when saving the plot.
        - Expects dataframe in the formatting of the tracking methods output.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be plotted.
        conditions_dict (dict): A dictionary where keys are condition names and values are lists of data labels under each condition.
        conversion_units (str): The units for the data being plotted, used for labeling the y-axis.

    Returns:
        None: This function does not return any values but outputs a boxplot to a file.
    """

    print(df, conditions_dict)

    # find column of interest
    # this is the y value column name that will be averaged and plotted
    interest_column, analysis_type = find_interest_column_and_type(df)
    df_label_columns = [col for col in df.columns if 'data_label' in col]
    condition_averages = defaultdict(list)
    conditions = list(conditions_dict.keys())
    _, time_label, time_unit = get_time_labels(df)

    for condition, label_list in conditions_dict.items(): # iterate through each condition
        print(condition, label_list)
        for label in label_list: # iterate through the labels associated with that condition
            print(label)
            for col in df_label_columns: # iterate through data label cols to find dataset index
                if df[col].str.contains(label).any(): # when we find the datalabel column for the cur label
                    dataset_num = int(col.split('-', 1)[0])
                    avg = np.mean(df[f'{dataset_num}-{interest_column}'])
                    condition_averages[condition].append(avg)
                    break

    print(condition_averages)

    data_lists = [condition_averages[condition] for condition in list(conditions_dict.keys())] # boxplot takes in a list of lists, each inner list is a list of points for the box
    fig, ax = plt.subplots()
    boxplot = ax.boxplot(data_lists, patch_artist=True)

    for patch in boxplot['boxes']:
        patch.set_facecolor('white')

    for median in boxplot['medians']:
        median.set_color('black') 

    for i, condition in enumerate(conditions, start=1):  # count from 1 to match boxplot positions
        means = condition_averages[condition] # grab avg previously calculated
        print(means, conditions, condition_averages)
        x_jitter = np.random.normal(i, 0.025, size=len(means)) # some x jitter so points don't overlap
        ax.scatter(x_jitter, means, color='darkblue', zorder=3, marker='D', s=20, alpha=0.8) # plot mean value in the correct box with some x jitter

    plot_args = get_plot_args(analysis_type, time_label=time_label, conversion_units=conversion_units, time_unit=time_unit)

    with open("plot_opts/plot_customizations.json", 'r') as plot_customs_file:
        plot_customs = json.load(plot_customs_file)
    font = plot_customs['font']

    tick_pos = list(range(1, len(condition_averages) + 1))
    plt.xticks(tick_pos, list(condition_averages.keys()), fontsize=plot_customs['value_text_size'], fontfamily=font)
    plt.yticks(fontsize=plot_customs['value_text_size'], fontfamily=font) 
    plt.title(plot_args['title'], fontsize=plot_customs['title_text_size'], fontfamily=font)
    plt.xlabel(plot_args['x_label'], fontsize=plot_customs['label_text_size'], fontfamily=font)
    plt.ylabel(plot_args['y_label'], fontsize=plot_customs['label_text_size'], fontfamily=font)
    plt.tick_params(axis='both', direction=plot_customs['tick_dir'])
    plt.tight_layout()
    
    plt.savefig(f"figures/{interest_column}_conditions_boxplot.{plot_customs['fig_format']}", dpi=plot_customs['fig_dpi'])

    print("Done")


def marker_movement_analysis(analysis_type, conversion_factor, conversion_units, output_df_path, output_y_col_name, **kwargs):
    """
    Analyzes marker movement data to calculate distance, displacement, velocity, or surface area changes over time.
    This function supports multiple types of analysis and is configurable to handle different types of location data.
    It also supports visualization and saving of results based on user configuration.

    Details:
        - The function reads tracking data from specified input CSV files depending on the locator type.
        - It performs the specified analysis type: distance, displacement, velocity, or changes in surface area.
        - Supports analysis of data from multiple trackers within a single video, multiple videos, or specified subsets of data.
        - Results are saved to a specified output path and visualized if configured.
    
    Note:
        - Ensure the input data files are properly formatted with the necessary columns for x and y locations or areas.
            - This should not be an issue if recorded csv data files are left unmodified
        - Parameters for the analysis such as `conversion_factor` and `conversion_units` are user spec'd.
        - The function uses additional optional parameters from the kwargs when called from the data selector or outlier removal tool.
        - Expects dataframe in the formatting of the tracking methods output.

    Args:
        analysis_type (enum): The type of analysis to perform (distance, displacement, velocity, or surface area).
        conversion_factor (float): The factor to convert pixel measurements to a desired unit.
        conversion_units (str): The units resulting from the conversion for labeling and output.
        output_df_path (str): The path to save the resulting DataFrame as a CSV file.
        output_y_col_name (str): The column name for the y-axis data in the output DataFrame.
        kwargs:
            df (pd.DataFrame, optional): DataFrame containing the tracking data. If not provided, data is read from the specified CSV.
            will_save_figures (bool, optional): If True, saves generated plots to files. Defaults to True.
            chosen_video_data (int, optional): Specifies a particular dataset to analyze if multiple datasets are present.
            locator_type (enum, optional): Specifies whether the data comes from bounding boxes or centroids.

    Returns:
        tuple: Contains time arrays, data arrays for each analyzed dataset, plot arguments, and the number of plots or datasets processed.
    """
    # unpack kwargs
    df = kwargs.get('df', None)
    will_save_figures = kwargs.get('will_save_figures', True)
    chosen_video_data = kwargs.get('chosen_video_data', None)
    locator_type = kwargs.get('locator_type', LocatorType.BBOX)

    # determine if we are finding just marker distance or centroid distance
    if locator_type == LocatorType.BBOX:
        input_df_path = "output/Tracking_Output.csv"
        x_loc_column = "-x (px)"
        y_loc_column = "-y (px)"
    elif locator_type == LocatorType.CENTROID:
        input_df_path = "output/Surface_Area_Output.csv"
        x_loc_column = "-x centroid location"
        y_loc_column = "-y centroid location"
        y_area_column = "-cell surface area (px^2)"
    else:
        raise Exception("Unkonwn locator type")

    print(f"Finding Marker something maybe...")
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(input_df_path) # open csv created/modified from marker tracking process
    print(df.head())

    if locator_type == LocatorType.BBOX:
        n_trackers = df['1-Tracker'].dropna().unique().shape[0] # get number of trackers
    else:
        n_trackers = 1

    time_col, time_label, time_unit = get_time_labels(df)
    _, _, num_datasets = get_num_datasets(df) # get num datasets
    
    data_labels = []
    analyzed_data = []
    times = []

    analysis_df = pd.DataFrame()
    for_range = [0,0]

    # determine if multiple videos, or multiple trackers, or single of both (outlier removal)
    n_plots = 1
    if chosen_video_data is None:
        data_multiplicity_type = check_num_trackers_with_num_datasets(n_trackers, num_datasets)
    else:
        data_multiplicity_type = DataMultiplicity.SINGLE
    
    if data_multiplicity_type == DataMultiplicity.VIDEOS:
        for_range = [1, num_datasets+1] # output tracking csv's are 1 indexed
        n_plots = num_datasets
        print("VIDEOS")
    elif data_multiplicity_type == DataMultiplicity.TRACKERS:
        for_range = [1, n_trackers+1]
        n_plots = n_trackers
        print("TRACKERS")
    elif data_multiplicity_type == DataMultiplicity.SINGLE:
        if chosen_video_data == None:
            chosen_video_data = 1
        for_range = [chosen_video_data, chosen_video_data+1]
        print("SINGLE")

    print(for_range)
    for data_idx in range(for_range[0], for_range[1]):
        try: # handle error that may occur of a column was removed manually by user
            if data_multiplicity_type == DataMultiplicity.TRACKERS:
                cur_df = df[df['1-Tracker'] == data_idx]
            else:
                cur_df = get_relevant_columns(df, data_idx)
        except:
            print(f"Dataset index: {data_idx} not found, skipping...")
            continue

        time_col, _, _ = get_time_labels(df, data_idx)
        time = cur_df[time_col].values
        time = time[~np.isnan(time)]
        x = cur_df[f'{data_idx}{x_loc_column}'].values * conversion_factor
        x = x[~np.isnan(x)]
        y = cur_df[f'{data_idx}{y_loc_column}'].values * conversion_factor
        y = y[~np.isnan(y)]

        data_label = cur_df[f'{data_idx}-data_label'].dropna().unique()[0]
        data_labels.append(data_label)
        
        if analysis_type == AnalysisType.DISTANCE:
            incr_dist = np.sqrt( np.diff(x)**2 + np.diff(y)**2 ) # magnitude of x and y differences
            cum_distance = np.cumsum(incr_dist)
            interest_data = np.insert(cum_distance, 0, 0.0)

        if analysis_type == AnalysisType.DISPLACEMENT:
            disps = np.sqrt( (x - x[0])**2 + (y - y[0])**2 )[1:]
            interest_data = np.insert(disps, 0, 0.0)

        if analysis_type == AnalysisType.VELOCITY:
            vel_x = np.diff(x) / np.diff(time)
            vel_y = np.diff(y) / np.diff(time)
            interest_data = np.sqrt(vel_x**2 + vel_y**2) # magnitude of velocities
            interest_data = np.insert(interest_data, 0, 0.0)

        if analysis_type == AnalysisType.SURFACE_AREA:
            interest_data = df[f'{data_idx}{y_area_column}'].values * conversion_factor
            interest_data = interest_data[~np.isnan(interest_data)]

        analyzed_data.append(interest_data)
        times.append(time)

        print("time len", len(time), "; y len", len(interest_data))
        cur_analysis_df = pd.DataFrame({
            time_col: time,
            f'{data_idx}-{output_y_col_name}': interest_data,
            f'{data_idx}-locator_type': locator_type.value,
            f'{data_idx}-units': conversion_units,
            f'{data_idx}-data_label': data_label
        })
        analysis_df = pd.concat([analysis_df, cur_analysis_df], axis=1)

    if(analysis_df != AnalysisType.SURFACE_AREA):
        analysis_df.to_csv(output_df_path, index=False)

    # plot
    plot_args = get_plot_args(analysis_type, time_label=time_label, data_labels=data_labels, conversion_units=conversion_units, time_unit=time_unit, locator_type=locator_type)

    if will_save_figures:
        fig_name = locator_type.value + ' ' + os.path.splitext(os.path.basename(output_df_path))[0]
        fig, ax = plot_scatter_data(times, analyzed_data, plot_args, n_plots, output_fig_name=fig_name)

    print("Done")
    return times, analyzed_data, plot_args, n_plots


#if __name__=='__main__':
#    analyze_marker_deltas()