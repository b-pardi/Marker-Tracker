# Marker-Tracker
### Computer vision scripts intended to track fiducial markers from videos of viscoelastic hydrogels as they expand over time

# How to Use
- First run 'install_packages.py'
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