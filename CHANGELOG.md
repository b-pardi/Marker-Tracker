## ISSUES
- outlier removal for marker tracking when points clicked they do not remove the correct ones
    - this is due to the fact that points shown in in the tool are from calculations based on the points being removed, need to plot something different here

## TODO
Housekeeping
- update documentation and readme
- scrollbar to ui

Hydrogel
- plot derivative of poissons, long, and rad strains over time (dot over variable)

Multi purpose


Cell Mechanics
- cell spread area over time


# Changelog

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