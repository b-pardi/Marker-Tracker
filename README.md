# Marker-Tracker
### Computer vision scripts intended to track fiducial markers from videos of viscoelastic hydrogels as they expand over time

# How to Use
- First run 'install_packages.py'
- If you get an error with this script,
    - In a command prompt, (not anaconda terminal) type 'where python' on windows, or in a mac terminal type 'which python'
    -FCopy and paste the full path that it prints out
        - On windows it should look something like: 'C:\<some path stuff>\Python\Python310\python.exe'
        - In spider, go to tools > preferences > python interpreter
        - Select 'Use the following Python interpreter:'
        - Paste in the path you copied earlier from the terminal
        - Click apply and ok, and restart spyder for changes to take effect
- Run 'main.py' upon completion of above
- Click button to select video file
- Select either marker tracking or necking point detection
- Click submit, each option's process is described below

### Marker Tracking
- After submitting, you'll see window of first frame of the video you selected
- In this window click on the markers you want to track
    - Right click to undo an erroneous selection
- Hit enter to confirm selections and proceed with tracking, or ESC to cancel
    - First window will close, but reopen momentarily, visualizing the tracking process
    - Hit ESC to cancel this process once it begins
- Outputs of the tracking will be saved in 'Tracking_Output.csv'

### Necking Point Detection
- After submitting, just sit back and watch the magic
- Outputs are saved to 'Necking_Point_Detection.csv'