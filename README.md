# Marker-Tracker
### Computer vision scripts intended to track fiducial markers from videos of viscoelastic hydrogels as they expand over time

# How to Use
- To use necking point detection, comment out 2 lines in main calling 'select_markers()' and 'track_markers()'
- To use marker tracking, comment out 'necking_point()' function call

### Necking Point Detection
- just run main.py, change optional args depending on video

### Marker Tracking
- run main.py to see window of first frame
- in this window click on the markers you want to track
    - right click to undo an erroneous selection
- hit enter to confirm selections and proceed with tracking, or ESC to cancel
    - first window will close, but reopen momentarily
    - hit ESC to cancel this process once it begins