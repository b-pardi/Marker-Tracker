'''
Author: Brandon Pardi
Created 1/26/2024

routines for tracking capabilities
'''

import cv2
import numpy as np
import pandas as pd
import screeninfo
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from exceptions import error_popup, warning_popup

def scale_frame(frame, scale_factor=0.9):
    monitor = screeninfo.get_monitors()[0] # get primary monitor resolution

    # get indv scale factors for width and height
    scale_factor_height = monitor.height / frame.shape[0]
    scale_factor_width = monitor.width / frame.shape[1]

    min_scale_factor = min(scale_factor_width, scale_factor_height)

    # resize based on scale factors
    scaled_frame = cv2.resize(frame, (int(frame.shape[1] * min_scale_factor), int(frame.shape[0] * min_scale_factor)))
    return scaled_frame, min_scale_factor

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
    bbox_size = params['bbox_size']
    radius = bbox_size // 2

    # scale frame for display so full image in view even if video res > monitor res
    first_frame_scaled, scale_factor = scale_frame(first_frame)

    if event == cv2.EVENT_LBUTTONDOWN: # on left click save pos and show on screen
        cur_marker = [(int(x / scale_factor), int(y / scale_factor))] # ensure original resolution coords stored
        marker_positions.append(cur_marker)

    if event == cv2.EVENT_RBUTTONDOWN: # on right click remove last selection
        if marker_positions:
            marker_positions.pop()

    cur_frame = first_frame_scaled.copy()
    for marker in marker_positions: # draw circles where selections made
        scaled_x = int(marker[0][0] * scale_factor)
        scaled_y = int(marker[0][1] * scale_factor)
        cv2.circle(cur_frame, (scaled_x, scaled_y), int(radius*scale_factor), (255, 255, 0), 2) # draw circle where clicked

    cv2.imshow('Select Markers', cur_frame)
    cv2.moveWindow('Select Markers', 50, 50)


def select_markers(cap, bbox_size, frame_start):
    """event loop for handling initial marker selection

    Args:
        cap (cv2.VideoCapture): loaded video file for selecting markers of

    Returns:
        mouse_params['marker_positions'] (list): selected marker positions
        first_frame (np.Array(np.uint8)): 3D array of 8 bit BGR pixel values of first frame
    """    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    ret, first_frame = cap.read() # get first frame for selection
    cv2.imshow('Select Markers', first_frame) # show first frame
    cv2.moveWindow('Select Markers', 50, 50)

    mouse_params = {"first_frame": first_frame.copy(), "marker_positions": [], 'bbox_size': bbox_size}
    cv2.setMouseCallback('Select Markers', mouse_callback, mouse_params) # set mouse callback function defn above
    
    # inf loop until user hits esc to cancel or enter to confirm selections
    while True:
        key = cv2.waitKey(1) # wait to capture input
        if key == 27: # 27 is ASCII for escape key
            print("SELECTIONS CANCELLED")
            cv2.destroyAllWindows()
            mouse_params['marker_positions'] = [(-1,-1)] # exit status failed
            break
        elif key == 13: # 13 ASCII for Enter key
            print(f"Selected positions: {mouse_params['marker_positions']}")
            break

    # close windows upon hitting select
    cv2.destroyAllWindows()
    
    return mouse_params['marker_positions'], first_frame


def track_markers(marker_positions, first_frame, frame_start, frame_end, cap, bbox_size, tracker_choice, frame_interval=0, time_units='s'):
    """main tracking loop of markers selected using marker selections from select_markers()
    saves distances of each mark each frame update to 'output/Tracking_Output.csv'

    Args:
        marker_positions (list): list of marker positions from user selections
        first_frame (np.Array(np.uint8)): image of first frame of loaded video
        cap (cv2.VideoCapture): loaded video
        bbox_size(int): length of 1 edge of the bounding box that will be the size of the tracker
        tracker_choice(str): 'KCF' or 'CSRT' determines choice of tracking algorithm used (see README.md for details)
    """    
    # create trackers
    trackers = []
    if tracker_choice == 'KCF':
        for _ in range(len(marker_positions)):
            trackers.append(cv2.TrackerKCF_create())
    elif tracker_choice == 'CSRT':
        for _ in range(len(marker_positions)):
            trackers.append(cv2.TrackerCSRT_create())

    # init trackers
    scaled_first_frame, scale_factor = scale_frame(first_frame)
    for i, mark_pos in enumerate(marker_positions):
        bbox = (int((mark_pos[0][0] - bbox_size // 2) * scale_factor),
                int((mark_pos[0][1] - bbox_size // 2) * scale_factor),
                int(bbox_size * scale_factor),
                int(bbox_size * scale_factor))
        trackers[i].init(scaled_first_frame, bbox)

    # init tracking data dict
    tracker_data = {'Frame': [], f'Time({time_units})': [], 'Tracker': [], 'x (px)': [], 'y (px)': []}
    frame_num = 0

    # tracking loop
    while True:
        ret, frame = cap.read()
        frame_num += 1
        if not ret:
            break  # break when frame read unsuccessful (end of video or error)

        scaled_frame, scale_factor = scale_frame(frame)  # Use the scale_factor obtained from the current frame scaling

        # updating trackers and saving location
        for i, tracker in enumerate(trackers):
            success, bbox = tracker.update(scaled_frame)

            if success:  # get coords of marker on successful frame update
                x_bbox, y_bbox, w_bbox, h_bbox = [int(coord) for coord in bbox]  # get coords of bbox in the scaled frame
                marker_center = (x_bbox + w_bbox // 2, y_bbox + h_bbox // 2)  # get center of bbox

                # record tracker locations using original resolution
                if frame_interval == 0:
                    tracker_data[f'Time({time_units})'].append(np.float32(frame_num / cap.get(5)))
                else:
                    tracker_data[f'Time({time_units})'].append(np.float16(frame_num * frame_interval))
                tracker_data['Frame'].append(frame_num)
                tracker_data['Tracker'].append(i + 1)
                tracker_data['x (px)'].append(int((marker_center[0] / scale_factor)))  # scale back to the original frame resolution
                tracker_data['y (px)'].append(int((marker_center[1] / scale_factor)))
                cv2.rectangle(scaled_frame, (x_bbox, y_bbox), (x_bbox + w_bbox, y_bbox + h_bbox), (0, 255, 0), 2)  # update tracker rectangle

        cv2.imshow("Tracking...", scaled_frame)  # show updated frame tracking

        if cv2.waitKey(1) == 27 or frame_num == frame_end - frame_start:  # cut tracking loop short if ESC hit
            break
    dist_df = pd.DataFrame(tracker_data)
    dist_df.set_index('Frame', inplace=True)
    dist_df.to_csv("output/Tracking_Output.csv")

    cap.release()
    cv2.destroyAllWindows()


def necking_point(cap, frame_start, frame_end, percent_crop_left=0., percent_crop_right=0., binarize_intensity_thresh=120, frame_interval=0, time_units='s'):
    """necking point detection loop
    necking point defined as the most shortest vertical line between two horizontal edges
    frames are preprocessed and then edges are detected, top and bottom most edges are singled out
    each frame the vertical distance between these edges are recorded at every x coordinate and the shortest is highlighted in red
    in cases where there are multiple min distance values, the median x location is chosen
    recorded data saved to "output/Necking_Point_Output.csv"

    Args:
        cap (cv2.VideoCapture): loaded video file for selecting markers of
        percent_crop_left (float, optional): percentage of pixels to remove from consideration from the left side of the frame of necking pt detection. Defaults to 0..
        percent_crop_right (_type_, optional): percentage of pixels to remove from consideration from the right side of the frame of necking pt detection. Defaults to 0..
        binarize_intensity_thresh (int, optional): threshold pixel intensity value for frame binarization. Defaults to 120.
        x_interval (int, optional): interval of horizontal pixels to draw vertical blue lines for visualization purposes. Defaults to 50.
    """    
    x_interval = 50 # interval for how many blue line visuals to display
    frame_num = 0
    dist_data = {'Frame': [], f'Time({time_units})': [], 'x at necking point (px)': [], 'y necking distance (px)': []}
    percent_crop_left *= 0.01
    percent_crop_right *= 0.01

    while True:  # read frame by frame until the end of the video
        ret, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start + frame_num)
        frame_num += 1

        if not ret:
            break

        scaled_frame, scale_factor = scale_frame(frame)  # scale the frame

        gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)  # convert frame to gray
        _, binary_frame = cv2.threshold(gray_frame, binarize_intensity_thresh, 255, cv2.THRESH_BINARY)  # threshold to binarize image

        # error checking for appropriate binarization threshold
        if np.all(binary_frame == 255):
            msg = "Binarization threshold too low,\nfound no pixels below the threshold.\n\nPlease adjust the threshold (default is 120)"
            error_popup(msg)
        if np.all(binary_frame == 0):
            msg = "Binarization threshold too high,\nfound no pixels above the threshold.\n\nPlease adjust the threshold (default is 120)"
            error_popup(msg)

        edges = cv2.Canny(binary_frame, 0, 2)  # edge detection, nums are gradient thresholds

        x_samples = []
        y_distances = []
        y_line_values = []

        frame_draw = scaled_frame.copy()
        frame_draw[edges > 0] = [0, 255, 0]  # draw edges

        # remove x% of edges from consideration of detection
        horizontal_pixels_left = 0
        horizontal_pixels_right = scaled_frame.shape[1]
        if percent_crop_left != 0.:
            left_pixels_removed = int(percent_crop_left * scaled_frame.shape[1])
            horizontal_pixels_left = max(0, left_pixels_removed)
        if percent_crop_right != 0.:
            right_pixels_removed = int(percent_crop_right * scaled_frame.shape[1])
            horizontal_pixels_right = min(scaled_frame.shape[1], scaled_frame.shape[1] - right_pixels_removed)

        for x in range(horizontal_pixels_left, horizontal_pixels_right):
            edge_pixels = np.nonzero(edges[:, x])[0]  # find y coord of edge pixels in cur column

            if edge_pixels.size > 0:  # if edge pixels in cur column,
                dist = np.abs(edge_pixels[0] - edge_pixels[-1])  # find distance of top and bottom edges
                x_samples.append(x)
                y_line_values.append((edge_pixels[0], edge_pixels[-1]))
                y_distances.append(dist)

                if x % x_interval == 0:  # draw visualization lines at every x_interval pixels
                    # draw vertical lines connecting edges for visualization
                    cv2.line(frame_draw, (x, edge_pixels[0]), (x, edge_pixels[-1]), (200, 0, 0), 1)  

        # find index of smallest distance
        # multiple mins occur in typically close together, for now just pick middle of min occurrences
        necking_distance = np.min(y_distances)
        necking_pt_indices = np.where(y_distances == necking_distance)[0]
        necking_pt_ind = int(np.median(necking_pt_indices))

        # record and save data using original resolution
        if frame_interval == 0:
            dist_data[f'Time({time_units})'].append(np.float32(frame_num / cap.get(5)))
        else:
            dist_data[f'Time({time_units})'].append(np.float16(frame_num * frame_interval))
        dist_data['Frame'].append(frame_num)
        dist_data['x at necking point (px)'].append(int((x_samples[necking_pt_ind] / scale_factor)))
        dist_data['y necking distance (px)'].append(necking_distance)

        cv2.line(frame_draw, (x_samples[necking_pt_ind], y_line_values[necking_pt_ind][0]), (x_samples[necking_pt_ind], y_line_values[necking_pt_ind][1]), (0, 0, 255), 2)     

        cv2.imshow('Necking Point Visualization', frame_draw)
        if cv2.waitKey(1) == 27 or frame_end == frame_num+frame_start:
            break

    dist_df = pd.DataFrame(dist_data)
    dist_df.set_index('Frame', inplace=True)
    dist_df.to_csv("output/Necking_Point_Output.csv")

    cap.release()
    cv2.destroyAllWindows()