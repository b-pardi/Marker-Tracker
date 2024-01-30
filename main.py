import cv2
import numpy as np
import pandas as pd

import sys

'''NOTES
- when running software window of first frame opens
- in this window click on the markers you want to track
    - right click to undo an erroneous selection
- hit enter to confirm selections and proceed with tracking, or ESC to cancel
    - first window will close, but reopen momentarily
    - hit ESC to cancel this process once it begins
'''

"""TODO
- when selecting markers, ability to right click and cancel previous selection
- wtf do i do with the tracking information
"""

global video_path
video_path = "videos/2024-01-23_Test video_DI-01232024120123.avi"

def marker_distance(p1, p2):
    
    return np.linalg.norm(np.array(p1) - np.array(p2))


def mouse_callback(event, x, y, flags, params):
    """handle mouse clicks during software execution
    intended for use with the selection of trackers

    Args:
        event (int): represents the mouse event type
        x (int): x coord of selection
        y (int): y coord of selection
        flags (int ): addtl flags of mouse selection state
        params (dict): parameters being passed into the callback
    """    
    first_frame = params['first_frame']
    marker_positions = params["marker_positions"]

    if event == cv2.EVENT_LBUTTONDOWN: # on left click save pos and show on screen
        cur_marker = [(x,y)]
        marker_positions.append(cur_marker)

    if event == cv2.EVENT_RBUTTONDOWN: # on right click remove last selection
        if marker_positions:
            marker_positions.pop()

    cur_frame = first_frame.copy()
    for marker in marker_positions: # draw circles where selections made
        cv2.circle(cur_frame, marker[0], 10, (255, 255, 0), 2) # draw circle where clicked

    cv2.imshow('Select Markers', cur_frame)


def select_markers(cap):
    """event loop for handling initial marker selection

    Args:
        cap (cv2.VideoCapture): loaded video file for selecting markers of

    Returns:
        mouse_params['marker_positions'] (list): selected marker positions
        first_frame (np.Array(np.uint8)): 3D array of 8 bit BGR pixel values of first frame
    """    
    ret, first_frame = cap.read() # get first frame for selection
    cv2.imshow('Select Markers', first_frame) # show first frame

    mouse_params = {"first_frame": first_frame.copy(), "marker_positions": []}
    cv2.setMouseCallback('Select Markers', mouse_callback, mouse_params) # set mouse callback function defn above
    
    # inf loop until user hits esc to cancel or enter to confirm selections
    while True:
        key = cv2.waitKey(1) # wait to capture input
        if key == 27: # 27 is ASCII for escape key
            print("SELECTIONS CANCELLED")
            sys.exit()
        elif key == 13: #13 ASCII for Enter key
            print(f"Selected positions: {mouse_params['marker_positions']}")
            break

    # close windows upon hitting select
    cv2.destroyAllWindows()
    
    return mouse_params['marker_positions'], first_frame


def track_markers(marker_positions, first_frame, cap):
    """main tracking loop of markers selected

    Args:
        marker_positions (list): list of marker positions from user selections
        first_frame (np.Array(np.uint8)): image of first frame of loaded video
        cap (cv2.VideoCapture): loaded video
    """    
    # create trackers
    trackers = []
    for _ in range(len(marker_positions)):
        trackers.append(cv2.TrackerKCF_create())

    # initialize trackers
    for i, mark_pos in enumerate(marker_positions):
        bbox = (mark_pos[0][0], mark_pos[0][1], 20, 20) # 20x20 bounding box
        trackers[i].init(first_frame, bbox)

    # init distances arr
    tracker_data = {'Frame': [], 'Time(s)': [], 'Tracker': [], 'x (px)': [], 'y (px)': []}
    frame_num = 0

    # tracking loop
    while True:
        ret, frame = cap.read()
        frame_num += 1
        if not ret:
            break # break when frame read unsuccessful (end of video or error)
        
        # updating trackers and saving location
        for i, tracker in enumerate(trackers):
            success, bbox = tracker.update(frame)

            if success: # get coords of marker on successful frame update
                x_bbox, y_bbox, w_bbox, h_bbox = [int(coord) for coord in bbox] # get coords of bbox
                marker_center = (x_bbox + w_bbox // 2, y_bbox + h_bbox // 2) # get center of bbox
                
                # record tracker locations
                tracker_data['Frame'].append(frame_num)
                tracker_data['Time(s)'].append(np.float32(frame_num / cap.get(5)))
                tracker_data['Tracker'].append(i+1)
                tracker_data['x (px)'].append(marker_center[0])
                tracker_data['y (px)'].append(marker_center[1])
                cv2.rectangle(frame, (x_bbox, y_bbox), (x_bbox + w_bbox, y_bbox + h_bbox), (0, 255, 0), 2) # update tracker rectangle
            
        cv2.imshow("Tracking...", frame) # show updated frame tracking

        if cv2.waitKey(1) == 27: # cut tracking loop short if ESC hit
            break
    
    dist_df = pd.DataFrame(tracker_data)
    dist_df.set_index('Frame', inplace=True)
    dist_df.to_csv("Tracking_Output.csv")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cap = cv2.VideoCapture(video_path) # load video

    # get video metadata
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(5))
    n_frames = int(cap.get(7))
    print(width, height, fps, n_frames)

    selected_markers, first_frame = select_markers(cap) # prompt to select markers
    track_markers(selected_markers, first_frame, cap)
