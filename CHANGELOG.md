## ISSUES
- for necking point:
    - distances only differ by a few pixels, meaning many locations share a min (usually consecutively)
        - currently take median of these locations
    - at end of video seems to jump to far right as it is the min area, but might be camera issue
        - exclude x% edges of frames?

## TODO
- ui for video path and selecting to do necking our marker tracking
- gaussian blurring
- 'crop' outer x% of video 


# Changelog
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