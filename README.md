# Marker-Tracker
### Computer vision scripts intended to track fiducial markers from videos of viscoelastic hydrogels as they expand over time (updated 2/16/24)

# How to Use

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

- If needed, select the frames of the video you want tracked by clicking the 'Select start/end frames' button
    - This will open a window to scroll through the video
    - Can use either arrow keys (shift + arrow keys to move 10 frames at a time) or the slider at the bottom of the window
    - Scroll to where you want the tracking to begin and click 'Confirm start frame
    - Repeat for tracking end frame
    - close window and frame selections are saved

- Indicate if the video you are using is a timelapse
    - This is important for later analysis, as the frames per second (FPS) will not directly correlate to the original time of the video if it is a timelapse
    - If the timelapse box is checked, 2 more prompts appear
        - Firstly specify the units of time of the original video, these units exactly as you type them will be displayed on any plot that has time as a variable (basically all plots)
        - Then specify the frame interval in units specified above
            - This is how often a picture was taken to make the timelapse
            - i.e. if you have a 24 hour timelapse and a picture was taken every 30 minutes, you could specify the desired time units as minutes and put 30 for the frame interval, or use hours and put 0.5 for the interval (every half hour) 

- Indicate if the tracking you are about to run will overwrite or append to existing tracking data
    - This will allow you to analyze tracked markers from multiple different videos in the same plot
    - **Do Not** Use append mode for the first tracking operation of your set of videos, only for subsequent videos
    - **Note** Use only 1 marker per video tracked if using the append feature
        - You can either use multiple markers within 1 video for analysis, or multiple videos with 1 marker each
        - The software will let you record multiple videos each with multiple markers, but will not be able to handle analysis
        - If you need to record data for multiple videos each with multiple markers, you can rerun the tracking operation on the same video and selecting a different object to track, essentially 1 marker per tracking operation, but multiple tracking operations are done on the same video.

- Select either marker tracking or necking point detection
- Specify parameters (detailed below)
- Click submit, each option's process is described below

## Individual Tracking Operation Details
### Marker Tracking
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
- Outputs are saved to 'output/Necking_Point_Detection.csv'

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
- When submitting, the first frame will popup to place a marker to select a cell
    - This is the same marker selection as the marker tracker uses
- Hit enter after marker is placed to begin tracking
- The green box will follow the cell and the blue lines indicate the contours of the cell, recording the surface area within the blue boundaries
- Outputs are saved to 'output/Surface_Area_Output.csv'

## After tracking videos, further analysis
### Outlier Removal
- Sometimes necking point or less commonly marker tracker jump around certain points and can skew data
- There is a 'Remove outliers' button which opens a new window prompting choice between available tracking output files
- Marker tracker currently work in progress, will not perform correctly
- Necking point detection will show the radial strain plot (same one generated by Necking point analysis button in main ui)
- It is here that outlier points can be identified and removed
- Simply click on points that are outliers to be removed, the plot will updated with the removed points gone after each click
- selections can be saved or undone via buttons at the top, as nothing is saved until the 'Confirm Removal' button is clicked

### Unit Conversion
- Users have the ability to convert units from pixels to whatever units they are working in, as long as they know the conversion ratio
- Enter the conversion factor to convert units of pixels to your desired unit of length (or area if using surface area tracking)
    - Common PPI (pixels per inch values) are 72, 96, or 300. This information can typically be found in the recording device/spec sheet of the device, or in any software that was used to process this previously like Fiji.
- Additionally, specify the units that result from this conversion
    - Akin to the timelapse units, these will be displayed exactly as typed out in all plots where this unit is the dependent variable
    - Pro tip: to type µ in any text field on windows, hold the alt key and press 230 (ALT+230)
    - mac users try Option+M I cannot confirm this I don't have a mac so good luck

### Data Visualization
- Currently 6 analysis buttons are available for tracking visualizations

- Marker deltas:
    - 'marker_deltas' plots the distance that 2 tracked markers are moving away from eachother, showing the horizontal (x) distance, as well as the euclidean distance for comparison
    - 'longitudinal_strain' plots the longitudinal strain ((L-L0) / L0) where L is the distance between the markers at time t and L0 is the initial distance between the markers
    - 'marker_euclidean_horizontal_differences' plots the discrepancies between euclidean and horizontal distances

- Necking point:
    - 'necking_point_location' plots as a control the horizontal marker location of the necking point (minimum diameter)
    - 'diameter_at_necking_point' plots the length of the vertical red marker from the tracking operation
    - 'radial_strain' plots the radial strain ((R-R0) / R0) where R is the diameter of the necking point and R0 is the initial diameter

- Poisson's ratio (plots all of the above, in addition to):
    - 'poissons_ratio' plots the poissons ratio using above calculations (-rad_strain / long_strain)
    - 'poissons_ratio_prime' derivative of poissons_ratio
    - 'long_strain_prime' derivative of long_strain
    - 'rad_strain_prime' derivative of rad_strain

- Marker velocity:
    - 'marker_velocity' plots the magnitude of the differences of x and y locations over time
    - 'average_marker_velocity' plots of bar graph of the average velocity of each marker within a user specified number of ranges
    - 'marker_velocity_FFT' plots the Fast Fourier Transform of the marker velocities

- Marker disance
    - 'marker_RMS_displacement' plots the root mean squared (RMS) displacement travelled by the marker over time

- Marker spread
    - 'marker_surface_area' plots the surface area of the tracked contours over time


### Plot Customizations
- In the plot opts folder there are 2 json files, in order to customize your plot you may edit any variable in the 'plot_customizations.json' file, while the 'default_opts.json' is purely for reference