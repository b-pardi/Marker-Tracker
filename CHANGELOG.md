## ISSUES

## TODO

FIX

- more mac compatible?
- scale frames more and in frame selector
- shrink ui
  - buttons width
  - open smaller height
- kcf markers not tracking well when moving really slow
  - try kalman filter
  - try optical flow
    - klt algorithm

ISAAC

- FramePreprocessor: Add internal live preview of frame changes

Housekeeping

Hydrogel

Multi purpose

- option to sharpen (kernel) and boost contrast (clahe)
  - aggression levels

Cell Mechanics

- see about surface tracking on the stained videos (awaiting videos)
  - (siammask/deepsort)

# Changelog

5/29 - ISAAC

- Incorporated frame preprocessor to track_area and associated marker select

5/28 - ISAAC

- In main: Frame preprocessor
  - added internal-frame preview and basic preprocessing slider options
- In tracker: added the preprocessing functions
  - enhance contrast
  - sharpen/sharpen
  - brighten (gamma)

5/23

- mac friendly stuff
  - fixed scrollbar not working on mac
  - scaled frame to even smaller for frame selection to it's all visible
  - put all stylings into a function, and OS specific styles are dictated
- added styles to things that were originally default (labels, checkboxes)
- added buttons for klt optical flow
- began implementation of optical flow
  - new function called in main (single thread only rn)
  - currently ignoring user inputted markers
    - need to do something about that probably
- boiler plate code for isaac to do frame preprocessing tool

5/22

- increased speed of single thread for tracking all frames
- added FAQ to the readme
- added various WIP frame enhancement functions that will be fully flushed out at a later date
  - sharpness
  - cotnrast (kernel)
  - gamma

5/21

- removed old err msg that prevented poissons ratio calculations being different sizes across different dataset
- fixed bug in dataselector where selecting range with 2 different size datasets broke it

5/15

- changed locator distance analysis from incremental distance to total distance at each time point
- fixed bug where data selector could not handle if 2 selected datasets were different lenghts (broke in the averaging section) due to np deprecating arr.flatten() 's ability to squeeze ndarrays with diff lens
  - used np.concatenate instead of arr.flatten()
- fixed multithreading bug with frame_tracker_midpt_finder_thread()
  - removed passing of scale frame in one area but didn't remove it function defn
- fixed bug in marker deltas where one particular video recorded multiple time values and broke marker deltas
  - changed how time values are attained, using time col of one of the marker dataframes instead of .unique() of current dataframe
- fixed bug with multithreading where changing frame record_interval meant if left the tracking loop early
  - literally just left out cap.set
  - putting it in slowed it down like crazy WHY GOD WHY
  - at least made it faster for tracking all frames

5/9

- adapted outlier removal tool to work with new refactor
  - since args are not all the same vars for all functions now, new dictionary self.function_args_map made and called right before function call to pass in \*\*args to analysis function appropriately
  - added centroid or marker options to outlier removal
    - this dictates the df input file for marker movement functions
    - added error checking to ensure centroid not selected if using poissons ratio analysis
    - added error checking to ensure centroid IS selected for surface area
- adapted data selector to new refactor
- fixed bug where data selector was not finding mean and std dev of labels individually
- fixed how data selector was calculating global mean and std dev
- fixed bug where plot args would only look for poissons ratio enum not also poissons ratio CSV
- data selector refactored
  - mainly just adapted interest column grabber and added updated output file names

5/8

- commented out rms displacement from marker distance awaiting meeting to decide future of rms functionality
- replaced it for now with regular displacement and distance
- fixed bug where contours were not being drawn on frame in multithreaded version

- beginning major refactor to put distance, displacement, velocity, and surface area into the same analysis function, and accounting for centroid vs marker locations

- in marker movement refactor branch:
  - added button for displacement, separating displacement and distance
  - removed rms distance and rms displacement opting for the regular distance and displacement
  - changed all cell related analysis buttons (displacement, distance, velocity, and area) to call a TrackingUI class function 'call_analysis'
  - this function calls marker movement analysis with the appropriate args depending on the enum given in the button function call
  - adjusted enums to not include the word marker and also split displacement into displacement and distance
  - added enum for locator type, so user can use the centroid as the marker from surface area tracking
  - error check to make sure user has centroid selected when using surface area cuz marker tracking will not give area or centroid
  - added kwargs to marker movement analysis in prep for receiving the args like df and chosen video data from tool function calls
  - determined df input path and x,y location column names based on locator type choice (marker or centroid)
  - data multiplicity determines for range, instead of handling different data multiplicities differently in the for loop, this waaaaayy cleaner
  - grab time, x, and y values all the same way across analysis types, each analysis type just has a few unique lines for the specific operations it needs.
  - depecreated velocity, displacement, and area functions opting for this new refactor
  - tl:dr combined 3 functions into 1 and it's way cleaner than any of the old ones were
- switched get_plot_args to use kwargs
- plot titles now include marker or centroid (if relevant)
- poissons ratio function calls now also handled from main ui class call_analysis function

5/7

- added formulas of how data is analyzed to readme
- fixed multithread bug where if queue item is none it breaks in some threads because some threads tried to unpack iterable of queue item before checking if it's none
- added centroid tracking ability to surface area tracker
  - draws red dot on screen where centroid is determined to be
  - made this modification to single and multithread versions
  - output file mostly unchanged, just changed column titles for x and y cell locations to centroid locations instead
- surface area multithread now shows the scaled original frames instead of processed ones
  - slightly less efficient, but more important to be able to see tracked things being drawn
- both box plot functions have changed formatting to a white face color and black median line
- fixed bug in time ranges box plot where mean points were not showing

5/5

- fixed bug where surface area wasn't checking for previously used data labels correctly
- box plotter conditions ability now working
- implemented box plotter time ranges
- full explanation of box plotter:
  - new class/window for boxplot
  - akin to other tools, prompt for analysis (3 cell ones only) and datalabels
  - prompt for grouping boxes by conditions or by time ranges
    - by conditions,
      - move from main ui to new window
      - prompt for user to enter a condition name, then select all labels that belong in that condition, then a button to add the label/condition combo to a list displayed in the ui, condition dict where key is condition and value is list of applicable data labels
      - modify to generalize the current cell velocity boxplot function
      - average velocity of each label across whole video becomes a point in the box, each condition becomes a box
    - by time points,
      - prompt user to select all data labels they want analyzed
      - two labelled entries for a t0 and tf of time range, with a button to pop up a new pair of labelled entries for user to keep iteratively entering time ranges they want analyzed
      - each time range becomes a box, the average value of the maker thing in that time range becomes a point
        - each cell will have n_ranges points in the whole graph, one average value in each box
  - in short, tool lets you box plot where boxes/group/averages can be determined by a condition where user groups data labels together, or time ranges where user specifies all labels to consider for a given time range
- data selector now finds average and std dev range selections
  - saves output to 'output/data_selector/'
  - records avg and stddev of each label as well as the global avg and std dev (across all labels)
- added buttons to boxplotter tool to remove an entered condition or time range
- added crop/compression tool
  - once video selected, user may click on button to open this tool
  - in new window button to crop video appears
    - opens first frame of video and user can select a roi (region of interest)
    - hitting enter confirms this selection and ESC cancels
    - original and new dimensions displayed below button
  - checkbox to compress video
    - if checked user may adjust quality and scale factors for reducing video bitrate and resolution
    - if a video is already compressed well it may increase size because why not
    - added openh264 binaries to folder to use h264 compression as it is best available open source compression algo
  - click save video will save a copy of the selected video with the selected cropping/compression
    - prepends 'CROP-COMP' to beginning of original video name and changes extension to mkv
  - when saved it'll update the main ui video path label to the new file and update TrackingUI.video_path to the new one as well
  - progress bar for video saving also available

5/3

- fixed bug when trackers fail to update software crashes due to integration of multithreading and tkinter pop up messages
  - made a queue for messages that is then read after threads are exited and rejoined
  - queue displayed after
- added option to do multithreading in ui
  - this means if multithread breaks for whatever reason single thread option still available
- necking point multithread option available
  - added thread for edge detection and saving necking point data
- necking point midpt version multithread option available
  - THREE THREADS HELL YEA
  - initial frame grabbing thread
  - marker update thread modified version of marker tracker thread specifically for finding midpoint and feeding another queue for third thread
    - feeds the next queue with a tuple with frame num, scaled frame, scale factor, and the located midpoint between the 2 markers
  - frame midpt edge detection thread grabs from above queue and performs rest of necking midpoint's functionality
    - detects edges and finds length of edge at midpoint and records data to dictionary
- added cProfile stats to all tracking functions
- surface area tracking multithread support
  - HUH 4 THREADS NOW???
  - same frame grabbing thread as others
  - reused necking_midpt finder tracker updater threader
    - modified to return whole bbox along with the midpt
    - made midpt a useless return arg ( \_ ) in this context, and made the bbox return useless in the necking midpt thread
  - frame_preprocessing thread does all the median blurring, and improved binarization methods, adding to the next queue the same things it pulled from the queue from necking_midpt tracker thread but giving it processed frame instead of original scaled frame
  - contours thread finds the contours and chooses the appropriate (nearest) one, then saves the data and does all the screen display things

5/1-5/2

- added cProfile tools to print runtime info to terminal
- marker tracking multithreading
  - track markers function still initializes trackers, but splits frame fetching and frame processing into 2 separate functions that then become threads
  - capture thread takes relevant frame information as well as initialized frame queue object
    - frame queue has max size of 10 to prevent memory leaks
  - processing thread reads frame queue, updates trackers, and records data
  - both frames check for ESC key to set the stop event, but only frame capture thread stops when stop event triggered
    - if frame processing also stopped with frames still in queue (if frame capture grabs frames faster than frame process can handle) then threads cannot rejoin
    - so when stop event happens, frames stop being grabbed, but queue will still be emptied by frame processing before termination

4/30

- fixed small box plotter bug so go button is always at bottom
- added a bit further instruction to labels in box plotter ui
- adjusted df output of marker velocity to include a time col for each dataset
- marker displacement now has output of data analyzed
- function to find the appropriate y column for boxplots given df passed in
  - so one function can handle condition boxplots for each analysis type
- adjusted how each analysis function gets plot args
  - function to hold the various plot args definitions
  - enum for analysis type to determine which one
- boxplotter kinda working, will fix next day and describe more

4/29

- refactored tracking loops to call record data function when saving dicts to csv
  - changed column renaming for appending to prettier way that functionalizes better
- implemented necking_point version 2 (midpoint method)
  - this does not replace the old method, rather serves as an alternate option
  - ui button for this method right next to original necking point button
  - prompts the same binarization threshold entry as necking point v1, and the marker bbox entry akin to marker tracking
  - function for tracking is amalgamation of marker tracking and necking point
    - inits trackers, enters tracking loop
    - processes frame and detects edges
    - updates trackers and finds midpoint between marker centers
    - records same data as neckingpoint v1, only difference is necking distance is now the distance at the midpoint between the markers, instead of the min distance within range of considered horizontal values
- if data label entry empty, set default value to data{n_prev_tracked_entries}
- built Boxplotter UI subwindow for specifying boxplot information
  - prompt for analysis option and datalabels
  - prompt for grouping boxes by conditions or by time ranges
  - by conditions,
    - prompt for user to enter a condition name, then select all labels that belong in that condition
    - button to add the label/condition combo to a list displayed in the ui, condition dict where key is condition and value is list of applicable data labels
  - by time points,
    - prompt user to select all data labels they want analyzed
    - two labelled entries for a t0 and tf of time range, with a button to pop up a new pair of labelled entries for user to keep iteratively entering time ranges they want analyzed
- go button to call boxplot function for either time ranges or conditions depending on which radio button selected
  - actual plotting functions not yet implemented

4/25

- added poissons ratio to outliar removal tool
  - reads from poissons ratio csv file, meaning calculation of poissons must be run first
  - this is because removing index of selected point in poissons does not directly correlate to its predecessor values due to the various calculations done
- added rms displacement to outlier removal
- added poissons ratio to dataselector
- added rms displacement to data selector
- fixed bug in marker velocity and distance where if points removed in one data label they would not plot points from other data labels at those same time stamps
- fixed bug where removing outliers in one data label removed them in others as well
- fixed bug where subsequent datalabels referenced the first data labels time column
- jfc I need to clean up these two functions but for now they work as expected and can plot/remove/analyze different length tracked datasets

4/24

- fixed outlier removal tool opting to remove points finding the indices at the time point selected
  - easy remedy to account for marker deltas having multiple marker entries per time point entry
  - fixed tool removing points across all data labels even when only one selected
- added new button for poissons ratio to plot from csv, and changed old one state that it's a calculation
  - figures saved from either will reflect this
- added data labels to poissons ratio output csv
- formatted poissons output time column to match output of other csv's
- plotting from csv works

4/23

- completed revamp of outlier removal tool
  - similar to data selector, drop down selector allows for choice of analysis
  - this choice populated the appropriate dataframe with the available datalabels in the datafile used by the corresponding analysis tool
  - once selected analysis type and datalabel user can generate plot and view points to click on and remove
    - only one datalabel can be viewed at a time
  - buttons to confirm removals and update csv or undo selections

4/22

- proper formatting for data selector tool
- set fig size of data selector to dynamically adjust its size based on a percentage of monitor's resolution and dpi
- dataselector changed second combobox to listbox to select multiple datasets
- dataselector can now plot either single or multiple datasets

4/18

- after all these fixes from 4/16 and the ones below, the end result is datasets with different times can be plotted together, no more needing all tracked videos to be the same length. Consequently poissons ratio plotting can now handle multiple datsets
- the following are fixes that were broken by the append data refactor on 4/16

  - fixed cell velocity outlier removal
  - surface area plotting
  - surface area outlier removal
  - necking point adds frame start to saved data so if start frame is not frame 0 the first time point will reflect that
  - ^^^ damn that was a dumb idea forget that, time always starts at 0 regardless of start frame
    - made plots weird
  - outlier removal for marker deltas
  - outlier removal for necking point

- now actually adding new features instead of just fixing things that one feature broke
- small button move to tracking operation options are now all on same line
- added ui button for data selector selector
- beginning implementation of data selector
  - DataSelector class opens child window
  - first selector box chooses from available options to analyze
  - 2nd selector box gets populated with available data labels once first box selected
    - else empty
  - go button takes those options and sends to appropriate analysis function

4/16

- closed figures after saving to save memory
- for marker elongation legend changed 'Horizontal Differences' to ' Horizontal Distances'
- adjusted y axis labels for:
  - marker deltas horizontal and euclidean differences
  - horizontal location of necking point
  - Diameter of necking point
  - poissons ratio rate
  - longitudinal strain rate
  - radial strain rate
- major refactor adding time and frames recorded when appending in tracking
  - previously when appending multiple videos it was sufficient to just append the tracking data and keeping time and frames from the original since they should be the same across each recording
  - in preparation for making poissons ratio work with multiple recordings, this needs to be changed since videos won't always be the same length
  - so now when appending datasets, the system implemented previously incrementing a number to recorded data column headers and prepending it to the header now also applies for columns Time and Frame
  - this leads to the next several refactors
- Refactored the following due to this appending data change:
  - updated previously appended removal tool
  - updated plot scatter data tool to take in multiple separate x datasets
  - marker deltas analysis
  - necking point analysis
  - poissons ratio plotting and csv output saving
  - marker velocity (just passed times instead of times[0] in plotting)
- the following now accepts multiple tracked videos:
  - marker deltas
  - necking point
  - poissons ratio
- added legends to poissons ratio plots since it can handle multiple videos tracked now
- fixed velocity boxplot setting tick positions dynamically (would fail if not exactly 2 boxes)

4/14

- revived fourier transform plotting for marker velocity
- added ability to track every 'x' frames (frame_record_interval)
  - defaults to 1 (track every frame)
- updated readme
- extensive documentation overhaul (thank you gpt)

4/12

- plotting bug fix where new legend method didn't account for if plot has no legend

4/9

- added error msg for if a tracker selection was not made
- added ability to skip 100 frames in the selector by holding ctrl and shift
- changed cell velocity y axis label
- fixed bug where scale_frame() wasn't actually scaling by a factor of the monitor resolution, just scaling down to fit in the monitor window
  - now it scales it down to scale_factor (defaults 0.9) \* min of monitor height or width
- frame selector now scales frame for viewing
- adjusted plot scatter function:
  - if more than 3 datasets, now puts legend outside of plot, while maintaining axis size
  - function now saves figs if it is passed in a figure name, rather than saving the fig in the function that calls it
  - dpi arg from plot_customizations.json now taken into account
- if more than 3 datasets in plotting, moves legend outside to the right side of plot
- scatterplot save fig now reads the figure fmt from plot_customs when saving
- added plot_customs and other formatting to boxplot function
- added error checking for box plots to ensure entered conditions was valid (found data for it)
- overlayed the average velocity points to their respective boxes in the boxplot
- added x jitter to points overlayed on boxplot so points don't overlap
- deprecated marker velocity fft until further notice
- updated readme

4/3

- forget vertical, become horizontal
  - ui the long way was not cutting it with more features being added, split top and bottom halfs onto a left and right side (data recording and data analysis)
- added to ui checkbox for doing a boxplot that reveals a labelled entry to enter the conditions that will be pulled from datalabels and grouped together for each box of the plot
- added err check for data labels to ensure there are no commas, as the string splitting of the conditions entry splits on commas
- fixed bug when checking if data label is already being used where sometimes random nans will appear in the list of labels, so now label is stripped of all non strings
- implemented cell velocity boxplot
  - each box will represent a condition
    - a condition is specified as a substring in all the datalabels corresponding to that condition
  - each point in the box is the average velocity of one tracked video of that box's condition

4/2

- various plot formatting

3/28

- removed bar plots for cell velocity
- added ui elements to set axis limits
  - when entering bounds, clicking 'Set limits' will save the values to the plot customs json
  - if empty this defaults to auto
- scatter plot function now reads 2 new variables, "y_lower_bound" and "y_upper_bound"
- these bounds set the axis limits of the plots
  - if it reads 'auto', sets plt.ylim as None for whichever bounds read 'auto'
- fixed data label issue where if no data label entered the plot legend would put 'nan'
  - changed how data label checks for nan using a pd.Series
  - if nan, sets data label to 'data {i}' where {i} is the index of the dataset being plotted
- refactored use of FileMode enum with tk int vars in a way that's more intuitive
- restructured handling of the following ui options, opting for enums instead of string comparisons:
  - time units
  - tracking operation
  - tracker (algorithm) choice
- began noise reduction function for reducing incredibly noisy videos for surface tracking

3/21

- added data labels to remaining analysis functions

3/18

- cleaned up outlier removal tool opting for grid instead of pack for ui elements
- added more enums to replace method of storing user opts
- fixed bug where time interval entries showed up below filemode radio btns when timelapse var checked
- added label and entry for user to indicate a custom data label for the output csv and figures
- added error checking to ensure that label hasn't been used previously in the output file
- replaced old data_label method with the custom one, defaulting to the old one if no label is entered for the following analysis functions:
  - rms_distance
  - marker_velocity

3/12

- bug fix with removal of last appended data button
- tried improve_binarization for necking point, it sucked

3/11

- added tk msg box to ask ok/cancel
- added ui buttons to select to undo previous append operation for each tracking operation
- implemented functionality of those buttons
  - finds n_entities and locates columns with that number (and a dash) and removes them
- added entry to outlier removal window to indicate which set to look for outliers in
- fixed marker velocity outlier removal
- fixed all poissons ratio analysis that broke during column renaming for appending data feature
- added error check for cell velocity outlier removal to check if user entered erroneous dataset choice
- added error checking for if tracker is lost indicating user to adjust params and retry
- marker velocity outlier removal now works with user inputted datasets beyond the first
- added marker spread to outlier removal tool

3/10

- reworked how data is recorded for all tracking operations s.t. initial columns have a 1 prepended to all column headers, instead of renaming column headers to 1 if appending
- surface area tracking now can plot the appended data
- added a checking function to determine if data has multiple appeneded videos tracked or 1 video with multiple trackers
  - cannot handle multiple appended videos with multiple trackers each
- marker distance reworked to handle either multiple trackers 1 video or multiple videos 1 tracker each
- marker velocity reworked same as marker distance
- all poissons ratio related analysis has error checks in place to ensure only 1 tracked video

3/8

- added ui widgets to indicate if overwriting or appending existing tracking data
- recorded data now also includes original video data originates from
- added ability to append data from separate tracking videos to existing data
  - this ability has been added to all 3 available tracking operations performing the same way
  - each column of new data gets prepended a number incr by 1 from number of previous tracked entities

3/7

- refined cell surface area tracking
  - improved binarization function
    - contrast limited adaptive histogram equalization
    - background subtraction
    - morphological closing
    - edge detection mask (1 iter edge dilation)
    - erosion
  - adaptive thresholding after binarization (helps somehow)
  - track specifically the largest contour that is within a threshold distance from the marker center
- record data for surface area
- plot surface area data over time
- added parameters to ui for tweaking surface area
  - bbox size (same as for regular marker tracking)
  - distance from marker threshold for how far to look from the marker for contours

3/4

- added a scrollbar to ui
- surface tracking progress

2/29

- plot difference between euclidean and horizontal distances over time
- plot derivative of poissons ratio over time
- plot derivative of longitudinal strain over time
- plot derivative of radial strain over time

2/28

- whole heap of plot formatting
- major adjustments to outlier removal
  - separate plots generated for marker deltas and marker tracking
  - marker tracking outliers use marker velocity calculations
  - fixed issue where when removing points from marker deltas, it would not remove where clicked
    - this was due to marker deltas points not directly corresponding to tracked marker points (intermediate calculations)
    - multiplying index by n_trackers fixes this
  - applied this fix to marker velocity so multiple markers can be plotted and have outliers removed
  - can now plot multiple markers tracked at once and removing one marker's outlier removes that same time point of the others as well, so no time points have na values
  - formatted plots using plot args from the analysis functions that generate them
- err catch for if len of tracker data is different

2/26

- plot_avgs_bar_data function complete
  - refactored for a simpler approach to variable number of trackers (ensure list is passed in regardless of if 1 or more trackers)
- refactored scatter plotting function to same
- refactored all functions that call these functions appropriately
- added saved dfs for intermediate calculations of cell velocity, cell rms displacement and poissons ratio
- adapted cell displacement to work for multiple trackers
- for cell velocity added ui input to specify number of ranges for bar graph
- account for timelapses (framerate does not directly correlate to time)
  - add to ui checkbox for if is timelapse
  - if yes, prompt for input of time units and frame interval
    - if no, stay default (s)
  - df time column will correspond to this user selection
  - time column in df reflects time units (f string in column header)
  - adjusted analysis functions to get time values from time column by looking for column headers that contain 'Time' instead of an exact match of 'Time(s)'
  - grab time units for plots from this column header
  - adjust all plot_args definitions so the x label contains these units
  - add err msg for if is timelapse box checked and no inputs entered

2/23

- adapted cell velocity to work with multiple trackers
- plot function can now take multiple indep and dep datasets (previously only multiple indep datasets)

2/20

- poissons ratio negative sign
- changed error message in poissons ration to a warning since outlier removal may have effect but still be same video on same time range
- added fft to marker velocity plotting
- implemented single marker total travel distance (RMS displacement)
- adjusted plot_data function to be able to plot 1 or more dependent sets passing in multiple sets as a list
- adjusted marker deltas plot to show horizontal distances and euclidean distances
- added input fields to specify units and conversion factor from pixels to those units
- updated all analysis functions to scale all pixel data by this factor
- updated all plotting to have y label include the user spec'd units converting to
- added confirmation labels on outlier removal too when changes save or undone

2/19

- added plot customizations option in the form of modifiable json file that analysis.py reads before plotting
- separated main ui code and all tracking routines to different files
- added ability to scroll through frames with an arrow key (shift arrow for skipping 10 frames) in frame selector
- added label to indicate this
- added buttons and structure for 3 cell mechanics visualizations
- implemented cell velocity visualizations
- added single marker tracking as an option for outlier removal

2/18

- added tiff to available files for reading
- added longitudinal strain plot for marker deltas
- added radial strain plot to necking point analysis
  - plots necking point length (diameter of gel at necking point) vs time
- added button for poissons ratio
- added error checking for poissons ratio calculations
  - checks if tracking operations previously ran on the same data in the same time range
- added analysis for poissons ratio
  - first runs marker deltas and necking point
  - error checks
  - calculate poissons ratio (radial strain / longitudinal strain)
  - plot
- added outlier removal class to remove outlier points manually via interactive plot
  - when button clicked user can select which output data they want to interact with
    - marker tracker does not work properly as it plots distances and removing points does not directly correlate to distances
  - when clicking points they are removed from plot and plot updates, as well as from data frame
  - also included buttons to save selections and undo them as well

2/16

- frame selector works
  - when clicking frame selector button in main ui, new window opens
  - displays first frame, 2 confirm btns and a slider
  - use the slider to find a frame
  - click confirm start/end selection
  - will save both selection in child class (frame selector window class)
  - main window checks if window was opened (self.child exists) and assigns the child class's frame selections to parent class frame start/end if selections were made
  - else they are just 0 and n_frames - 1
- updated marker tracker, marker selection, and necking point to accommodate frame selections
- formatted ttk buttons in frame select window
- added error catch for frame selections to ensure video has been selected first
- updated all first frame displays to move window to top left so it doesn't need to be moved since some is usually off screen
- frame selections now have a label to indicate selected frames in main ui
- indicators also in frame selections window
- went back to old file structure (scripts in root dir)
- fixed bug when moving window
- all video/img windows scale to 90% of monitor resolution, but when saving data original resolution values used

2/15

- updated file structure
- updated most tk objects to ttk objects (themed tkinter objs)
- added styles to ttk objects
- added frame selector class (and btn in parent window)
- frame selector allows user to select start and end frames of video (in progress)

2/13

- updated horizontal pixel removal parameters to remove left or right x% of edges from necking point consideration separately
- updated ui to have separate inputs
- added plotting functionality for necking point
  - plots x location of necking point vs time
- updated code documentation
- added failed exit status to marker selection if ESC key is pressed
- marker selection circles now have a radius corresponding to user spec'd bbox size
- when selecting markers, first frame resolution is now scaled to ensure full frame is in view and selection can be made anywhere
  - when tracking video plays and when marker locations saved, the original resolution is displayed and recorded, scaling is just for first frame selections and then scaled back to original resolution

2/12

- moved tkinter ui to an object oriented setup for easier expanding
- added various ui parameters for tweaking necking point process
  - added bbox size for marker tracker
  - added binarization pixel intensity threshold for necking pt
  - added percent crop of edges for necking point
- added functions for producing error/warning msgs
  - added error message for if binarize threshold too big/small
  - added error message for if unable to open video for necking and markers
  - added error message for if video was unable to open
- adjusted marker tracking bbox so that initial locations of tracker selections become the center of the tracker, not the top left corner
- added ability to exclude outer x% of horizontal pixels from consideration for necking point
- added ability to choose between 2 tracking methods
  - KCF: good for more stable markers
  - CSRT: good for tracking deformable objects
- removed sys.exit when escaping marker tracking, instead just close window but keep ui open
- added buttons to plot data from video tracking functions
- began plotting functionality for tracking 2 marker distances relative to eachother
- calculate euclidean distances of markers
- plot distances of markers against time (currently in seconds)
- added function to simplify plot formatting code
- added 'figures' folder for saving code
- added exit button that calls sys.exit since just closing window can cause it to freeze up

2/4

- added ui for selecting video and tracking operation

2/3

- ditched contour method, opting to just track edges as curve of contour was so miniscule
- added distance tracking between top and bottom edges
- find min distance location
- added blue vertical lines for tracking all distances visually, only shows every 50th pixel
- min distance is drawn in red
- multiple min locations generally near eachother, choose median of indices at which min occurs
- save frame, time, and x location and y distance where necking occurs each frame to 'Necking_Point_Output.csv'

1/30

- for necking point added frame binarization
- added contour tracking functionality
- contour gradient thresholds
- added area threshold

1/29

- fixed right click to deselect bug
  - right click removed markers from list but not from screen
- documentation
- obtained pixel locations of tracker center at each frame update
- saved these location to df->csv over each frame
  - output is 'Tracking_Output.csv'

1/26

- init commit
- read video file
- ability to select markers from first frame
- ability to deselect markers by right clicking
- initialize trackers from selection
- track markers throughout course of video
