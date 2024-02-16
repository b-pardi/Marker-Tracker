# Marker-Tracker
### Computer vision scripts intended to track fiducial markers from videos of viscoelastic hydrogels as they expand over time (updated 2/16/24)

# How to Use
- First run 'install_packages.py'
- If you get an error with this script,
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
    - This will open a window with a slider to scroll through the video
    - Scroll to where you want the tracking to begin and click 'Confirm start frame
    - Repeat for tracking end frame
    - close window and frame selections are saved
- Select either marker tracking or necking point detection
- Specify parameters (detailed below)
- Click submit, each option's process is described below

### Marker Tracking
- Tunable parameters include:
    - Size of bounding box (bbox) for tracker (default 20 pixels)
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