# Marker-Tracker

### Computer vision scripts intended to track fiducial markers from videos of viscoelastic hydrogels as they expand over time (updated 2/16/24)

# How to Use
## Getting Started
There are two methods for downloading the package.
### For developers or those who may want to contribute in some fashion
- Clone the repository into a directory of your choosing
	- In a terminal, create or navigate to the folder you want to download the code to
	- Run the command `git clone https://github.com/b-pardi/Marker-Tracker.git`
	- You should now have all the required files
### For those without programming experience that simply want to use the software
- Download the zip file
	- In the webpage for this repository, click the button that says 'code' (1)
	- At the bottom of the drop down, click the 'Download ZIP' button (2)
	- Extract the zip file to a folder of your choice
### Once downloaded, the remaining steps apply to both cases
- In a terminal, make sure you are in the parent directory of the code
	- Your current path should look like: `C:\path\to\directory\Marker-Tracker` with the key component being that the directory ends with 'Marker-Tracker'
	- If not, use the `cd` command to navigate to the directory
		- If you cloned the repository, you only need to move one directory: `cd Marker-Tracker`
		- If you extracted the ZIP file, use your file explorer to find the code you extracted and enter the Marker-Tracker folder
		- Then copy the folder path
		- In a terminal, type: `cd "<paste your path here>"` ensuring you have the quotes
	- **Note for Spyder**: you can open a terminal by going to `menu > View > Panes > Terminal`
- Setup virtual environment (optional but reccommended)
	- Create the virtual environment: `python -m venv .venv`
	- Activate it
		- Windows: `\.venv\Scripts\activate`
		- Mac/Linux: `source .venv/bin/activate`
	- You can also use conda if preferred
- Install package dependencies
	- Install with pip using the requirements file: `pip install -r requirements.txt`
	- If you get an error with this command in Spyder,
		- Have python 3.10.x installed on your computer (not via spyder, from https://www.python.org/downloads/)
		- In a command prompt, (not anaconda terminal) type 'where python' on windows, or in a mac terminal type 'which python'
		- Copy and paste the full path that it prints out
			- On windows it should look something like: 'C:\<some path stuff>\Python\Python310\python.exe'
			- In spider, go to tools > preferences > python interpreter
			- Select 'Use the following Python interpreter:'
			- Paste in the path you copied earlier from the terminal
			- Click apply and ok, and restart spyder for changes to take effect
- Run main.py to get started
	- **NOTE** main.py is the ONLY python file that should ever be executed. The other scripts are dependend on main.py for UI inputs.
## Main UI Instructions

- First run 'install_packages.py'

  - If you get an error with this script using Spyder,
    - First ensure you have python 3.10.x installed (not just through Spyder)
    - In a command prompt, (not anaconda terminal) type 'where python' on windows, or in a mac terminal type 'which python'
    - Copy and paste the full path that it prints out
      - On windows it should look something like: 'C:\<some path stuff>\Python\Python310\python.exe'
      - In spider, go to tools > preferences > python interpreter
      - Select 'Use the following Python interpreter:'
      - Paste in the path you copied earlier from the terminal
      - Click apply and ok, and restart spyder for changes to take effect

- Run 'main.py' upon completion of above
- Click button to select video file
- **NOTE** This software was designed on Windows, for Windows. The video tracking and data analysis should perform the same across operating systems, but the UI main have some bugs displaying. Several adjustments have been made to behave better on Mac and Linux systems, but issues may persist.

### Crop/Compression Tool

- Once video selected, user may click on button to open this tool
  - In new window button to crop video appears
    - Opens first frame of video and user can select a roi (region of interest)
    - Clicking enter confirms this selection and ESC cancels
    - Original and new dimensions displayed below button
  - User may also click the checkbox to compress video
    - If checked, user may adjust quality and scale factors for reducing video bitrate and resolution
    - If a video is already compressed well it may increase size because why not
  - Clicking save video will save a copy of the selected video with the selected cropping/compression
    - Prepends 'CROP-COMP' to beginning of original video name and changes extension to mkv
    - Saves in the same folder as the original video
  - Progress bar will indicate progression of reduced video encoding/saving
  - After saving, the new video file name and path will be updated in the main UI, so no need to select the new video, it is automatically selected

### Frame Selection Tool

- If needed, select the frames of the video you want tracked by clicking the 'Select start/end frames' button
  - This will open a window to scroll through the video
  - Can use either arrow keys or the slider at the bottom of the window
    - when using arrow keys, holding shift will skip 10 frames at a time
    - holding ctrl+shift will skip 100 at a time
  - Scroll to where you want the tracking to begin and click 'Confirm start frame
  - Repeat for tracking end frame
  - close window and frame selections are saved

### Frame Preprocessor

- Once video selected, user may choose and fine-tune various preprocessing functions to apply to the video
	- Allows users to fine-tune sharpness, contrast, and brightness sliders. Users can preview the effects of these adjustments in real-time.
- Select a checkbox to enable that process. Boxes left checked when the window is closed will be saved and applied when you use the video.

### Timelapse Accountability

- Indicate if the video you are using is a timelapse
  - This is important for later analysis, as the frames per second (FPS) will not directly correlate to the original time of the video if it is a timelapse
  - If the timelapse box is checked, 2 more prompts appear
    - Firstly specify the units of time of the original video, these units exactly as you type them will be displayed on any plot that has time as a variable (basically all plots)
    - Then specify the frame interval in units specified above
      - This is how often a picture was taken to make the timelapse
      - i.e. if you have a 24 hour timelapse and a picture was taken every 30 minutes, you could specify the desired time units as minutes and put 30 for the frame interval, or use hours and put 0.5 for the interval (every half hour)

### Output Data File Mode Indication

- Indicate if the tracking you are about to run will overwrite or append to existing tracking data
  - This will allow you to analyze tracked markers from multiple different videos in the same plot
  - **Do Not** Use append mode for the first tracking operation of your set of videos, only for subsequent videos
  - **Note** Use only 1 marker per video tracked if using the append feature
    - You can either use multiple markers within 1 video for analysis, or multiple videos with 1 marker each
    - The software will let you record multiple videos each with multiple markers, but will not be able to handle analysis
    - If you need to record data for multiple videos each with multiple markers, you can rerun the tracking operation on the same video and selecting a different object to track, essentially 1 marker per tracking operation, but multiple tracking operations are done on the same video.

### Frame Recording Interval

- Sometimes videos are unnecessarily long, and not much happens between frames
- User has the option to record every 'x' frames, specified here
- The video will still play and track as usual, but instead of playing and tracking every frame, it will skip the number of frames specified here, exponentially speeding up process
- **Note** not recommended for tracking very dynamic objects in videos

### Data Labels

- Enter an optional label that will be associated with the data you are about to record.
  - This label will appear in the output file and in the plot legend
  - If no label indicated, will just show the default 'data i' where i is in the dataset index
  - **Note** If you plan on utilizing any tools beyond just basic plotting, such as the data selector, outlier removal, or boxplot tools, use data labels, they are used to reference your recordings in all of these features

### Beginning Tracking

- Select either marker tracking or necking point detection
- Specify parameters (detailed below)
- Check to use multithreaded tracking if desired
  - Over 3 times faster than regular single thread, but unstable
- Click begin tracking, each option's process is described below
  - **Note** if there was a mistake or error in an appended tracking operation, the 3 buttons on the bottom left allow you to remove the most recently appended recorded data from each of the 3 tracking operations separately

## Individual Tracking Operation Details

### Marker Tracking

- For hydrogels, click on 2 markers on the hydrogel to track
- For cell tracking, click one or more cells to track
  - **Note** if using append feature to track cells across multiple datasets, use only one marker per video
- Tunable parameters include:
  - Size of bounding box (bbox) for tracker (default 100 pixels)
    - bbox should be slightly larger than the object being tracked
    - should capture discernable areas of contrast
  - Choice of tracking algorithm
    - KCF (default): fast and efficient, more stable tracking rigid objects (non deforming)
    - CSRT: slower than KCF and jittery on rigid objects, but can track deformable objects
- After submitting, you'll see window of first frame of the video you selected
- In this window click on the markers you want to track
  - Right click to undo an erroneous selection
- Hit enter to confirm selections and proceed with tracking, or ESC to cancel
  - First window will close, but reopen momentarily, visualizing the tracking process
  - Hit ESC to cancel this process once it begins
- Outputs of the tracking will be saved in 'output/Tracking_Output.csv'

### Necking Point Detection

- Tunable parameters include:
  - % of video width to exclude outter edges of
    - Sometimes lighting effects make the tube at the edges of the video appear narrower than they actually are, effecting the necking point detection
    - This allows the ability to exclude x% of the video from detection, essentially cropping it
    - Full video will still be displayed, but the vertical blue lines will indicate area of consideration
  - Threshold pixel intensity value for frame binarization
    - This is a pixel brightness value from 0-255, where any brightness value below this becomes black (0), any any above becomes white (255)
    - This frame binarization is not displayed, but the edge detection uses these binarized images behind the scenes
- After submitting, just sit back and watch the magic
  - Green lines are detected edges, top and bottom edges are what are looked at vertical distance of necking point
  - blue lines are visualizations of the vertical distances to ensure they are touching the top and bottom green edges
  - red line moving around is tracking the minimum distance
- **Midpoint method**
  - Alternatively, you can track the necking point with the midpoint method which will simply keep track of the vertical distance at the midpoint between 2 tracked markers
    - Includes same threshold parameter fom necking point above, as well as tracker bbox size described in Marker Tracking section
- Outputs are saved to 'output/Necking_Point_Detection.csv' and there will be a column to indicate if it was from minimum distance or midpoint method

### Surface Area Tracking

- Tunable parameters include:
  - Size of bounding box (bbox) for tracker (default 100 pixels)
    - bbox should be slightly larger than the object being tracked
    - should capture discernable areas of contrast
  - Distance from marker threshold (default 150 pixels)
    - When running surface area tracking, a CSRT marker tracker is used
    - This marker identifies which cell should have its surface area (contours) tracked
    - This parameter is the maximum distance from the marker to look for contours
    - i.e. when the marker is placed and this paramter is at the default 150, the tracking will only consider contours less than or equal to 150 pixels away from the center of the marker
    - This is so the algorithm knows which contours to pay attention to
- When beginning tracking, the first frame will popup to place a marker to select a cell
  - This is the same marker selection as the marker tracker uses
- Hit enter after marker is placed to begin tracking
- The green box will follow the cell and the blue lines indicate the contours of the cell, recording the surface area within the blue boundaries, as well as a red dot to track the centroid
- Outputs are saved to 'output/Surface_Area_Output.csv'

## Multithreading Capabilities

- Every tracking function includes a single threaded and multi threaded version
- Clicking the check box to use multi threading will activate the multithreading version
- This version is less stable but has been shown to more than triple the speed that these algorithms run in most cases
- Here is a summary of the run time performance of each tracking method. The numbers represent cumulative run time of the indicated tracking algorithm, on the left is single thread and on the right is multithread. Each measure was taken from the same video and the same section of the video. All units are in seconds
  marker tracker (2 threads) (cells)
  10.94 -> 2.87
  marker tracker (2 threads) (hydrogels)
  34.58 -> 9.65
  necking point (2 threads):
  46.89 -> 12.96
  necking midpt method (3 threads):
  56.62 -> 19.78
  surface area (4 threads):
  9.96 -> 6.85

# After Tracking Videos, Data Cleaning and Customized Visualization

### Outlier Removal

- Sometimes necking point or less commonly marker tracker jump around certain points and can skew data
- There is a 'Remove outliers' button which opens a new window prompting choice between available tracking output files
- Choose an analysis operation output to view and select points from, and upon selecting you will see the data labels you have tracked to choose which dataset to remove points from
- There are several analysis options that can be viewed here
- It is here that outlier points can be identified and removed
- Simply click on points that are outliers to be removed, the plot will updated with the removed points gone after each click
- selections can be saved or undone via buttons at the top, as nothing is saved until the 'Confirm Removal' button is clicked
- Removed points become nan values in the datasheet

### Unit Conversion

- Users have the ability to convert units from pixels to whatever units they are working in, as long as they know the conversion ratio
- Enter the conversion factor to convert units of pixels to your desired unit of length (or area if using surface area tracking)
  - Common PPI (pixels per inch values) are 72, 96, or 300. This information can typically be found in the recording device/spec sheet of the device, or in any software that was used to process this previously like Fiji.
- Additionally, specify the units that result from this conversion
  - Akin to the timelapse units, these will be displayed exactly as typed out in all plots where this unit is the dependent variable
  - Pro tip: to type µ in any text field on windows, hold the alt key and press 230 (ALT+230)
  - mac users try Option+M I cannot confirm this I don't have a mac so good luck

### Y Axis limits

- Users have the option to set y axis limits to keep consistency across multiple plots
- Simply type the number at which you want to y axis to start and stop (in units that you spec'd above) and click set limits
- Once these are set you will not need to alter them upon closing and reopening the software, they are saved in 'plot_opts/plot_customizations.json' for future use
- They will remain unaltered until 'Set limits' is clicked again

### Data Selector Tool

- To precisely narrow down the range of data you want to save your plots as, this it the tool
- Select analysis option available, and upon doing so the previously tracked data labels will appear for you to select
  - You can select as many as you want
- By clicking generate plot the analysis will be performed as if you had clicked the original buttons (described below)
- Key difference is that this plot has a span selector to zoom in and essentially crop irrelevant data
- Click reset zoom to reset the x bounds back to their original
- Once desired zoom is attained, click save plot to save the plot with the zoom you have currently applied
- Data selector will also save that mean and standard deviation of points within selected range
  - Saves those stats of each label as well as those stats across all labels in time range
  - Saves to 'output/data*selector/data_selector_stats*{analysis type selected}.csv'

### Box Plotter Tool

- Box plotter allows for grouping of data to generate box plots
- **Note** for whichever analysis option you would like to box plot (i.e. marker velocity) make sure you have already clicked that analysis button in the main ui to run the computation first and generate the standard scatter plots
  - boxplot uses output csv's from the analysis functions
- Groupings of data for boxes are either by condition or time ranges
  - By conditions:
    - First enter the name of the condition. This condition name can be whatever you like, it will become the axis label for the box you are about to select data labels for in the plot
    - Then select data labels you would like to be averaged under this condition
      - Average of the y values of the selected analysis option will become points in the box plot, and the condition they are grouped under will be the box
    - Click add condition to add this grouping of labels, it will be added to the initially empty box below, and you can select and remove a condition if needed
    - Repeat this process for all desired condition/label groupings
    - Click 'Go' and a box plot will be generated
  - By time ranges:
    - First select all the data labels you would like to consider for analysis
      - Contrary to 'By conditions', this method will average y values in the selected time ranges across data from ALL selected labels, so the labels you choose here should remain consistent across all time range entries
      - Each time range entered will become the label used for that box in the plot
    - Then enter a start and end time pair, ensuring units you enter them in are the same that you chose when recording data (default seconds)
      - For your convenience, there is a label above the entry boxes that tells you what time unit you selected previously
    - Click add time range, this will put the pair t0 and tf into the initially empty box below
      - A remove selected entry button is available if needed
    - Repeat for all desired time ranges
    - Click 'Go' and a box plot will be generated

### Plot Customizations

- All plotting functions read customization options from 'plot_opts/plot_customizations.json'
- The values associated with each key are modifiable to the user, and will save on exiting of the software so they do not need to be adjusted constantly

# Currently 8 analysis buttons are available for tracking visualizations

## Poisson's Ratio Related Analysis (Hydrogels)

### Marker deltas:

- 'marker_deltas' plots the distance that 2 tracked markers are moving away from eachother, showing the horizontal (x) distance, as well as the euclidean distance for comparison
  - euclidean distance defined as: np.sqrt((p2x - p1x)**2 + (p2y - p1y)**2)
  - p1 and p2 are 2 points that the distance between is being calculated
- 'longitudinal_strain' plots the longitudinal strain ((L-L0) / L0) where L is the distance between the markers at time t and L0 is the initial distance between the markers
  - This is the plot for Marker deltas that shows in the outlier removal tool
- 'marker_euclidean_horizontal_differences' plots the discrepancies between euclidean and horizontal distances

### Necking point:

- 'necking_point_location' plots as a control the horizontal marker location of the necking point (minimum diameter)
- 'diameter_at_necking_point' plots the length of the vertical red marker from the tracking operation
- 'radial_strain' plots the radial strain ((R-R0) / R0) where R is the diameter of the necking point and R0 is the initial diameter
  - This is the plot for necking point that shows in the outlier removal tool

### Poisson's ratio (plots all of the above, in addition to):

- 'poissons_ratio' plots the poissons ratio using above calculations (-rad_strain / long_strain)
- 'poissons_ratio_prime' derivative of poissons_ratio
- 'long_strain_prime' derivative of long_strain
- 'rad_strain_prime' derivative of rad_strain
- **NOTE** Distinguishing between calculate and csv buttons
  - poissons ratio calculate or from csv both plot the poissons ratio, however calculate runs all the calculations using the original data in necking point and marker output csv files,
    and from csv will use the output csv file produced by the calculate button
  - The difference is so poisson's ratio can benefit from the Outlier removal and Data selector tools. So when selecting data or removing outliers, it will look in and modify that csv file
    - This is because removing points at indices selecting in the poissons ratio does not necessarily correlate to removing them from the predecessor data files, necking point and marker output, due to the various calculations done
  - **Important** Poisson's ratio calculate will need to be run first no matter what, and then when plotting after removing outliers or using data selector, the from_csv button will generate and save the plot with the modified data

## Cell Related Analysis

**Important Note for the following analysis methods**

- You have the option to user either the marker or centroid to determine the position of the cell
  - This choice will herein be referred to as 'locator choice/type'
- Above the analysis buttons there are radio buttons to indicate which you'd like to use
- If you would like to use Centroid locator type, track your video using the 'Surface area' tracker
- If you would like to use the traditional marker bounding box (bbox), track your video with marker tracker
- The locator choice will be reported in the output csv files of the analysis, as well as the figure name and title
  - Figures are named as: "{locator type} {analysis type}.{extension}"
- **Note** Surface area analysis button will require the centroid radio button to be selected as surface area data is only possible to record in this one

### Marker velocity:

- '{locator_type}\_velocity' plots the magnitude of the differences of x and y locations over time
  - we get this the following way:
    - find dx, dy, dt (the differences between each x, y, and time value)
    - get velocity of x and y separately: dx / dt and dy / dt
    - magnitude of velocity: np.sqrt(vel_x**2 + vel_y**2)

### Marker Distance

- RMS metrics are work in progress, for now we are just plotting the difference between points for distance, and the difference of points from there initial (displacement)
  - '{locator_type}\_distance' plots the magnitude of differences of x and y coordinate points overtime

### Marker Displacement

- RMS metrics are work in progress, for now we are just plotting the difference of points from there initial location
  - '{locator_type}\_displacement' plots the magnitude of the differences of x-x0 and y-y0 over time

### Marker spread

- 'centroid_surface_area' plots the surface area of the tracked contours over time

### Plot Customizations

- In the plot opts folder there are 2 json files, in order to customize your plot you may edit any variable in the 'plot_customizations.json' file, while the 'default_opts.json' is purely for reference

# FAQ

- my CSRT trackers aren't tracking well, how can I fix this?
  - change the size of the bounding box, it should encompass as much of the object as possible without getting too much background. Play with the bbox size to find a happy medium
  - Try selecting a different area of the object to track
- my KCF trackers aren't tracking as well as they should, why?
  - If the trackers in your video are moving really slowly, increase the frame record interval so they move faster
  - Try using CSRT trackers instead
