
''' DEPRECATED FUNCTIONS '''


def marker_distance(user_unit_conversion, df=None, will_save_figures=True, chosen_video_data=None, locator_type=LocatorType.BBOX):
    """
    DEPRECATED - opted for all encompassing marker_movement_analysis

    CURRENTLY AWAITING DETAILS ON REIMPLEMENTING RMS, so currently just plots regular distance and displacement
    magnitude of difference of 2D points

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
    # determine if we are finding just marker distance or centroid distance
    if locator_type == LocatorType.BBOX:
        df_path = "output/Tracking_Output.csv"
        x_loc_column = "-x (px)"
        y_loc_column = "-y (px)"
    elif locator_type == LocatorType.CENTROID:
        df_path = "output/Surface_Area_Output.csv"
        x_loc_column = "-x centroid location"
        y_loc_column = "-y centroid location"

    print("Finding Marker Displacement and Distance...")
    conversion_factor, conversion_units = user_unit_conversion
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df_path) # open csv created/modified from marker tracking process
    print(df.head())
    
    time_col, time_label, time_unit = get_time_labels(df)
    _, _, num_tracker_datasets = get_num_datasets(df) # get num datasets
    n_trackers = df['1-Tracker'].dropna().unique().shape[0] # get number of trackers
    data_labels = []
    #rms_disps = []
    displacements = []
    distances = []
    times = []
    print(n_trackers, num_tracker_datasets)
    n_plots = 0
    rms_df = pd.DataFrame()

    n_datasets = int(df.columns[-1].split('-', 1)[0])
    # determine if there are multiple trackers and 1 video, or multiple videos and 1 tracker
    if chosen_video_data is None:
        chosen_video_data = 1
        data_multiplicity_type = check_num_trackers_with_num_datasets(n_trackers, num_tracker_datasets)
    else: # if called from data selector or outlier removal will do one dataset at a time
        data_multiplicity_type = DataMultiplicity.SINGLE

    if data_multiplicity_type == DataMultiplicity.TRACKERS or data_multiplicity_type == DataMultiplicity.SINGLE:
        # account for mismatch lengths of tracker information (if 1 tracker fell off before the other)
        #check_tracker_data_lengths(df, n_trackers)
        n_plots = n_trackers

        for tracker in range(n_trackers):
            print("CHOSEN ", chosen_video_data)
            time_col, _, _ = get_time_labels(df, chosen_video_data)
            cur_df = df[df[f'{chosen_video_data}-Tracker'] == tracker+1]
            time = cur_df[time_col].unique()
            time = time[~np.isnan(time)]
            x = cur_df[f'{chosen_video_data}{x_loc_column}'].values * conversion_factor
            y = cur_df[f'{chosen_video_data}{y_loc_column}'].values * conversion_factor
            data_label = cur_df[f'{chosen_video_data}-data_label'].dropna().unique()[0]
            data_labels.append(data_label)
            '''rms = rms_displacement(x[~np.isnan(x)], y[~np.isnan(y)])
            rms_disps.append(rms)'''

            disp = np.sqrt( (x - x[0])**2 + (y - y[0])**2 )[1:]
            dist = np.sqrt( np.diff(x)**2 + np.diff(y)**2 ) # magnitude of x and y differences
            times.append(time[:-1])
            displacements.append(disp)
            distances.append(dist)

            print(len( time[:-1]),len(disp),len(dist),len(data_label),)
            cur_rms_df = pd.DataFrame({
                time_col: time[:-1],
                f'{chosen_video_data}-displacement': disp,
                f'{chosen_video_data}-distance': dist,
                f'{chosen_video_data}-rms_displacement': np.nan,
                f'{chosen_video_data}-data_label': data_label
            })
            rms_df = pd.concat([rms_df, cur_rms_df], axis=1)

    elif data_multiplicity_type == DataMultiplicity.VIDEOS:
        n_plots = num_tracker_datasets
        for dataset in range(num_tracker_datasets):
            cur_df = get_relevant_columns(df, dataset+1)
            x = cur_df[f'{dataset+1}{x_loc_column}'].values * conversion_factor
            y = cur_df[f'{dataset+1}{y_loc_column}'].values * conversion_factor
            time_col, _, _ = get_time_labels(df, dataset+1)
            time = cur_df[time_col].values
            time = time[~np.isnan(time)]
            data_label = cur_df[f'{dataset+1}-data_label'].dropna().unique()[0]
            data_labels.append(data_label)
            '''rms = rms_displacement(x[~np.isnan(x)], y[~np.isnan(y)])
            rms_disps.append(rms)'''
            disp = np.sqrt( (x - x[0])**2 + (y - y[0])**2 )
            dist = np.sqrt( np.diff(x)**2 + np.diff(y)**2 ) # magnitude of x and y differences
            times.append(time[:-1])
            displacements.append(disp)
            distances.append(dist)

            cur_rms_df = pd.DataFrame({
                time_col: time[:-1],
                f'{dataset+1}-displacement': disp,
                f'{dataset+1}-distance': dist,
                f'{dataset+1}-rms_displacement': np.nan,
                f'{dataset+1}-data_label': data_label
            })
            rms_df = pd.concat([rms_df, cur_rms_df], axis=1)

    rms_df.to_csv("output/rms_displacement.csv", index=False)

    
    # plot marker velocity
    disp_plot_args = get_plot_args(AnalysisType.MARKER_DISPLACEMENT, time_label, data_labels, conversion_units, time_unit)
    dist_plot_args = get_plot_args(AnalysisType.MARKER_DISTANCE, time_label, data_labels, conversion_units, time_unit)

    if will_save_figures:
        disp_fig, disp_ax = plot_scatter_data(times, displacements, disp_plot_args, n_plots, output_fig_name='marker_displacement')
        dist_fig, dist_ax = plot_scatter_data(times, distances, dist_plot_args, n_plots, output_fig_name='marker_distance')

    print("Done")
    return times, displacements, disp_plot_args, n_plots

def single_marker_spread(user_unit_conversion, df=None, will_save_figures=True, chosen_video_data=None):
    """
    DEPRECATED - opted for all encompassing marker_movement_analysis
    Analyzes and visualizes changes in tracked surface areas over time. This function processes the previously tracked surface
    area data to compute metrics such as area growth rate and plots these changes over time to facilitate analysis.

    Details:
        - The function reads surface area tracking data from 'output/Surface_Are_Output.csv', which should contain surface area data
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
    plot_args = get_plot_args(AnalysisType.SURFACE_AREA, time_label, data_labels, conversion_units)

    if will_save_figures:
        # plot marker velocity
        area_fig, area_ax = plot_scatter_data(times, surface_areas, plot_args, num_tracker_datasets, output_fig_name='marker_surface_area')

    print("Done")
    return times, surface_areas, plot_args, num_tracker_datasets

def marker_velocity(user_unit_conversion, df=None, will_save_figures=True, chosen_video_data=None):
    """ 
    DEPRECATED - opted for all encompassing marker_movement_analysis
    
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
    
    _, _, num_tracker_datasets = get_num_datasets(df) # get num datasets
    if chosen_video_data is None:
        chosen_video_data = 1
        tracker_cols = [col for col in df.columns if col.__contains__('Tracker')]
        n_trackers = df[tracker_cols[0]].dropna().unique().shape[0] # get number of trackers
        data_multiplicity_type = check_num_trackers_with_num_datasets(n_trackers, num_tracker_datasets)
        print("DATA MULT", data_multiplicity_type)
    else: # if called from data selector or outlier removal will do one dataset at a time
        data_multiplicity_type = DataMultiplicity.SINGLE
        n_trackers = df[f'{chosen_video_data}-Tracker'].dropna().unique().shape[0] # get number of trackers
    n_plots = 0
    times = []
    data_labels = []
    tracker_velocities = []
    tracker_amplitudes = []
    tracker_frequencies = []
    time_col, time_label, time_unit = get_time_labels(df, chosen_video_data)

    # determine if there are multiple trackers and 1 video, or multiple videos and 1 tracker
    if data_multiplicity_type == DataMultiplicity.TRACKERS or data_multiplicity_type == DataMultiplicity.SINGLE:
        # account for mismatch lengths of tracker information (if 1 tracker fell off before the other)
        n_plots = n_trackers
        num_sets = n_trackers
    elif data_multiplicity_type == DataMultiplicity.VIDEOS:
        n_plots = num_tracker_datasets
        num_sets = num_tracker_datasets

    # grab relevant values from df
    vel_df = pd.DataFrame()
    for n in range(num_sets):
        print("TIMECOL", time_col)
        if data_multiplicity_type == DataMultiplicity.SINGLE:
            print("TEST SINGLE AND CHOSEN VIDEO DATA")
            time_col, _, _ = get_time_labels(df, chosen_video_data)
            cur_df = df[df[f'{chosen_video_data}-Tracker'] == n+1]
            x = cur_df[f'{chosen_video_data}-x (px)'].values * conversion_factor
            y = cur_df[f'{chosen_video_data}-y (px)'].values * conversion_factor
            data_label = df[f'{n+1}-data_label'].dropna().unique()[0]
            data_labels.append(data_label)
            time = cur_df[time_col].values
        elif data_multiplicity_type == DataMultiplicity.TRACKERS:
            cur_df = df[df['1-Tracker'] == n+1]
            x = cur_df['1-x (px)'].values * conversion_factor
            y = cur_df['1-y (px)'].values * conversion_factor
            time = cur_df[time_col].values
            data_label = df[f'{n+1}-data_label'].dropna().unique()[0]
            data_labels.append(data_label)
        elif data_multiplicity_type == DataMultiplicity.VIDEOS:
            time_col, _, _ = get_time_labels(df, n+1)
            x = df[f'{n+1}-x (px)'].values * conversion_factor
            y = df[f'{n+1}-y (px)'].values * conversion_factor
            time = df[time_col].values
            data_label = df[f'{n+1}-data_label'].dropna().unique()[0]
            data_labels.append(data_label)

        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        time = time[~np.isnan(time)]
        print("LENGTHSG", len(time[:-1]),time[:-1], '\n', len(x),x)

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

        cur_vel_df = pd.DataFrame({
            time_col: time[:-1],
            f'{n+1}-x_velocity_tracker': vel_x,
            f'{n+1}-y_velocity_tracker': vel_y,
            f'{n+1}-magnitude_velocity_tracker': vel_mag,
            f'{n+1}-data_label': data_label
        })
        vel_df = pd.concat([vel_df, cur_vel_df], axis=1)

        # fourier transform
        N = len(vel_mag)
        vel_fft = np.fft.fft(vel_mag) # amplitude (pixels/Hz)
        freqs = np.fft.fftfreq(len(vel_mag), np.mean(dt)) # frequencies
        tracker_amplitudes.append(vel_fft)
        tracker_frequencies.append(freqs)

    # save intermediate calculations
    vel_df.to_csv("output/marker_velocity.csv", index=False)

    # plot
    plot_args = get_plot_args(AnalysisType.MARKER_VELOCITY, time_label, data_labels, conversion_units, time_unit)
    
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
    """ DEPRECATED OPTING FOR BOXPLOTTER CLASS
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

