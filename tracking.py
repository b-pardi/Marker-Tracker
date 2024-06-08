'''
Author: Brandon Pardi
Created 1/26/2024

routines for tracking capabilities
'''

import cv2
import numpy as np
import pandas as pd
import time
import screeninfo
import threading
import queue
from collections import defaultdict

from exceptions import error_popup, warning_popup
from enums import *

def scale_frame(frame, scale_factor=0.9):
    """Sometimes resolution of video is larger than resolution of monitor this software is running on.
    This function scales the resolution of the frame so the entire frame can be seen for selecting markers or just general analysis.
    Scaling preseves aspect ratio, so it determines which dimension is most cut off from viewing (width or height),
    and determines caling ratio for the other dimension based on that.

    Distances of objects being tracked are scaled back up when recording data, so movement of tracked objects are recorded in the original resolution.

    Args:
        frame (numpy.ndarray): indv frame of video being tracked that will be scaled and returned
        scale_factor (float, optional): fraction of monitor resolution to scale image. Defaults to 0.9.

    Returns:
        scaled_frame (numpy.ndarray): scaled version of frame that was passed in
        min_scale_factor (float): Determing scale factor, used to scale values back up before recording data.
    """    
    monitor = screeninfo.get_monitors()[0] # get primary monitor resolution

    # get indv scale factors for width and height
    scale_factor_height = scale_factor * (monitor.height / frame.shape[0])
    scale_factor_width = scale_factor * (monitor.width / frame.shape[1])

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


def select_markers(cap, bbox_size, frame_start, preprocessVals = None):
    """event loop for handling initial marker selection

    Args:
        cap (cv2.VideoCapture): loaded video file for selecting markers of
        bbox_size (int): size of bounding box that is spec'd by user in ui
        frame_start (int): starting frame of video indicated in the frame selection tool

    Returns:
        mouse_params['marker_positions'] (list): selected marker positions
        first_frame (np.Array(np.uint8)): 3D array of 8 bit BGR pixel values of first frame
    """    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    ret, first_frame = cap.read() # get first frame for selection

    gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    if preprocessVals is not None:
        print(preprocessVals)
        first_frame = preprocess_frame(gray_frame, preprocessVals)
    
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

def record_data(file_mode, data_dict, output_fp):
    """
    Saves or appends tracking data to a CSV file based on the provided file mode. This function handles data formatting, 
    checks for empty labels, and ensures correct indexing and merging for new tracking data entries.

    Details:
        - If set to overwrite, the function creates a new DataFrame from the tracking data and saves it to the specified file path.
        - If set to append, the function loads an existing CSV file, increments tracking identifiers, and merges new data with existing data.
        - Ensures that each entry is correctly labeled and indexed, particularly important for maintaining consistent and readable data structures in long-term tracking studies.

    Note:
        - The function assumes that the input data dictionary is structured with keys corresponding to DataFrame column names.
        - It is crucial that the existing data file (when appending) is formatted correctly to prevent data corruption or misalignment.
        - Errors or exceptions might occur if the file operations encounter issues like permissions or disk space errors, which should be handled by the calling function.

    Args:
        file_mode (FileMode): Enum specifying whether to 'OVERWRITE' the existing file or 'APPEND' to it.
        data_dict (dict): Dictionary containing the tracking data to record, with keys as column names and values as lists of data points.
        output_fp (str): File path where the CSV file will be saved or appended.

    Returns:
        None: The function does not return any value but writes data directly to a CSV file specified by `output_fp`.
    """
    if file_mode == FileMode.OVERWRITE:
        # check data label and put default value if empty
        if data_dict['1-data_label'] == '':
            data_dict['1-data_label'] = 'data1'
        dist_df = pd.DataFrame(data_dict)
        dist_df.set_index('1-Frame', inplace=True)
        dist_df.to_csv(output_fp)
    elif file_mode == FileMode.APPEND:
        dist_df = pd.read_csv(output_fp)
        num_prev_trackers = int(dist_df.columns[-1][0])
        print(f"Num previous tracked entities: {num_prev_trackers}")

        if data_dict['1-data_label'] == '':
            data_dict['1-data_label'] = f'data{num_prev_trackers+1}'

        # create new df of current tracking data and merge into previous
        cur_df = pd.DataFrame(data_dict)

        # rename df to indicate which num tracker this is
        '''cur_df.rename(columns={
                '1-Frame': f'{num_prev_trackers+1}-Frame',
                f'1-Time({time_units})': f'{num_prev_trackers+1}-Time({time_units})','1-x at necking point (px)': f'{num_prev_trackers+1}-x at necking point (px)',
                '1-y necking distance (px)': f'{num_prev_trackers+1}-y necking distance (px)',
                '1-video_file_name' : f'{num_prev_trackers+1}-video_file_name',
                '1-data_label': f'{num_prev_trackers+1}-data_label'
            }, inplace=True)'''
        new_col_names = {}
        for col in cur_df.columns:
            if col.startswith('1-'):
                new_col = col.replace('1-', f'{num_prev_trackers+1}-')
                new_col_names[col] = new_col
        cur_df.rename(columns=new_col_names, inplace=True)
        
        dist_df_merged = pd.concat([dist_df, cur_df], axis=1)
        dist_df_merged.to_csv(output_fp, index=False)

def init_trackers(marker_positions, bbox_size, first_frame, tracker_choice=TrackerChoice.KCF):
    """
    Initializes and configures trackers based on specified marker positions, bounding box size, and tracker type. The function
    scales the first frame, sets up the trackers, and initializes them with the position and size of the bounding boxes around the markers.

    Details:
        - The function supports multiple types of trackers, allowing selection based on the tracking requirements and performance characteristics.
        - Trackers are initialized with bounding boxes centered on the provided marker positions, adjusted according to the scale of the first frame.
        - This setup is essential for subsequent tracking operations where accuracy and reliability depend heavily on the initial configuration.

    Note:
        - Ensure that the `first_frame` is the first frame of an openCV video.
        - The marker positions should be accurately determined as they directly affect the tracking performance.
        - This function is typically called at the start of a tracking session to prepare the trackers for the video processing tasks.

    Args:
        marker_positions (list of tuples): Positions of the markers to track, each given as (x, y) coordinates.
        bbox_size (int): Size of the square bounding box for each tracker.
        first_frame (np.array): The first frame of the video used for scaling and initializing the trackers.
        tracker_choice (TrackerChoice, optional): The type of tracker to use, defaults to `TrackerChoice.KCF`. Options are `TrackerChoice.KCF` or `TrackerChoice.CSRT`.

    Returns:
        list: A list of initialized tracker objects ready for use in tracking operations.
    """
    # create trackers
    trackers = []
    if tracker_choice == TrackerChoice.KCF:
        for _ in range(len(marker_positions)):
            trackers.append(cv2.TrackerKCF_create())
    elif tracker_choice == TrackerChoice.CSRT:
        for _ in range(len(marker_positions)):
            trackers.append(cv2.TrackerCSRT_create())
    elif tracker_choice == TrackerChoice.KLT:
        old_gray = cv2.cvtColor(scale_frame(first_frame)[0], cv2.COLOR_BGR2GRAY)
        marker_positions = [mp[0] for mp in marker_positions]
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        #p0 = np.array(marker_positions, dtype=np.float32).reshape(-1, 1, 2)
        return old_gray, p0

    # init trackers
    scaled_first_frame, scale_factor = scale_frame(first_frame)
    for i, mark_pos in enumerate(marker_positions):
        bbox = (int((mark_pos[0][0] - bbox_size // 2) * scale_factor),
                int((mark_pos[0][1] - bbox_size // 2) * scale_factor),
                int(bbox_size * scale_factor),
                int(bbox_size * scale_factor))
        trackers[i].init(scaled_first_frame, bbox)

    return trackers

def preprocess_frame(frame, preprocessVals):
    # Initialize a variable to store the modified frame
    modified_frame = frame.copy()

    # Apply improved smooth
    if preprocessVals["Smoothness"] != 0:
        modified_frame = improve_smoothing(modified_frame, preprocessVals["Smoothness"]/50+.1)

    # Apply sharpening
    if preprocessVals["Blur/Sharpness"] != 0:
        modified_frame = sharpen_frame(modified_frame, preprocessVals["Blur/Sharpness"])

    # Apply contrast enhancement
    if preprocessVals["Contrast"] > 0:
        modified_frame = enhance_contrast(modified_frame, preprocessVals["Contrast"])
    
    # Apply brightness adjustment
    if preprocessVals["Brightness"] != 0:
        modified_frame = adjust_gamma(modified_frame, preprocessVals["Brightness"])

    if preprocessVals["Binarize"]:
        modified_frame = improve_binarization(modified_frame)

    return modified_frame

def enhance_contrast(frame, strength=50):
    # Define parameters for contrast enhancement
    clip_limit = 3.0
    tile_grid_size = (8, 8)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_frame = clahe.apply(frame)
    
    # Adjust contrast strength
    enhanced_frame = cv2.addWeighted(frame, 1 + strength / 5, enhanced_frame, 0.0, 0.0)
    
    return enhanced_frame

def sharpen_frame(frame, strength=1.0):
    if strength > 0:
        # Sharpening
        scaled_strength = strength / 100
        kernel = np.array([[0, -0.2, 0],
                           [-0.2, 2 + 3 * scaled_strength, -0.2],
                           [0, -0.2, 0]])
    else:
        # Blurring
        scaled_strength = abs(strength) / 10
        kernel_size = int(1 + 2 * scaled_strength)
        if kernel_size % 2 == 0:  # Ensure the kernel size is odd
            kernel_size += 1
        blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        return blurred

    # Apply the kernel to the image
    result = cv2.filter2D(frame, -1, kernel)
    return result

def adjust_gamma(frame, gamma=50.0):
    # Apply gamma correction
    if gamma > 0:
        gamma=1-gamma/100
    elif gamma < 0:
        gamma=1-gamma/10
    gamma_corrected = np.array(255 * (frame / 255) ** gamma, dtype='uint8')
    return gamma_corrected


def track_klt_optical_flow(
        marker_positions,
        first_frame,
        frame_start,
        frame_end,
        cap,
        bbox_size,
        tracker_choice,
        frame_record_interval,
        frame_interval,
        time_units,
        file_mode,
        video_file_name,
        data_label):

    prev_gray, p0 = init_trackers(marker_positions, bbox_size, first_frame, tracker_choice)
    iter_lim = 10 # lk stop critera will try max 10 iterations or,
    goal_acc = 0.03 # if accuracy is <= 3%

    print('p0: ', p0, '\n\nprev_gray: ', prev_gray)

    lk_params = {
        'winSize': (15,15),
        'maxLevel': 2, # number of pyramid levels, 2 would be using image at original, half, and quarter scales
        'criteria': (cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, iter_lim, goal_acc)
    }

    tracker_data = defaultdict(list)
    frame_num = frame_start

    while True:
        ret, frame = cap.read()
        frame_num += frame_record_interval
        if frame_record_interval != 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        if not ret:
            break

        scaled_frame, scale_factor = scale_frame(frame)
        enhanced_frame = enhance_contrast(scaled_frame)

        frame_gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)

        # calculates optical flow giving us new pt locations, status of if each pt was found, and err of each pt
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)

        # use points that had a successful update
        good_p1 = p1[st == 1]
        good_p0 = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_p1, good_p0)):
            x1, y1 = new.ravel()
            x0, y0 = old.ravel()
            print(x1, y1)
            tracker_data[f'1-Time({time_units})'].append(np.float16((frame_num - frame_start) / cap.get(cv2.CAP_PROP_FPS)))
            tracker_data['1-Frame'].append(frame_num - frame_start)
            tracker_data['1-Tracker'].append(i + 1)
            tracker_data['1-x (px)'].append(x1)
            tracker_data['1-y (px)'].append(y1)
            cv2.circle(enhanced_frame, (int(x0), int(y0)), 5, (0, 0, 255), -1)
            cv2.circle(enhanced_frame, (int(x1), int(y1)), 5, (0, 255, 0), -1)

        prev_gray = frame_gray.copy() # update old frame to get ready for next loop
        p0 = good_p1.reshape(-1, 1, 2)

        cv2.imshow("Tracking...", enhanced_frame)  # show updated frame tracking

        if cv2.waitKey(1) == 27 or frame_num >= frame_end:  # cut tracking loop short if ESC hit
            break
    
    record_data(file_mode, tracker_data, "output/Tracking_Output.csv")

    cap.release()

def track_markers(
        marker_positions,
        first_frame,
        frame_start,
        frame_end,
        cap,
        bbox_size,
        tracker_choice,
        frame_record_interval,
        frame_interval,
        time_units,
        file_mode,
        video_file_name,
        data_label):
    """Implements the main tracking loop for markers on a video.
    It tracks markers from a user-defined start to end frame and records the data.

    Functionality:
        1. Initializes trackers based on user-selected markers and tracking algorithm.
        2. Scales the first frame and initializes bounding boxes for trackers.
        3. Enters a loop to track markers across specified frames, scaling each frame as processed.
        4. Records tracking data for each frame where markers are successfully tracked.
        5. Saves or appends the tracking data to a CSV file depending on the file mode.
        6. Handles errors by displaying warning messages and terminating tracking if necessary.

    Args:
        marker_positions (list of tuples): Initial positions of markers selected by the user.
        first_frame (np.array): Image of the first frame of the video where markers were selected.
        frame_start (int): The frame number to start tracking.
        frame_end (int): The frame number to end tracking.
        cap (cv2.VideoCapture): Video capture object loaded with the video file.
        bbox_size (int): Size of the bounding box for each tracker.
        tracker_choice (TrackerChoice): Enum specifying the tracking algorithm to use (KCF or CSRT).
        frame_record_interval (int): Interval at which frames are processed and data is recorded.
        frame_interval (float): Real-world time interval between frames, used in time calculations for data recording.
        time_units (str): Units for time (e.g., 'seconds', 'minutes') used in the output data.
        file_mode (FileMode): Specifies whether to overwrite or append data in the output file.
        video_file_name (str): Name of the video file being processed.
        data_label (str): Unique identifier for the tracking session, used in data labeling.
    """    


    trackers = init_trackers(marker_positions, bbox_size, first_frame, tracker_choice)

    # init tracking data dict
    tracker_data = {'1-Frame': [], f'1-Time({time_units})': [], '1-Tracker': [], '1-x (px)': [], '1-y (px)': [], '1-video_file_name': video_file_name, '1-data_label': data_label}
    frame_num = frame_start

    # tracking loop
    while True:
        ret, frame = cap.read()
        frame_num += frame_record_interval
        if frame_record_interval != 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

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
                    tracker_data[f'1-Time({time_units})'].append(np.float16((frame_num - frame_start) / cap.get(5)))
                else:
                    tracker_data[f'1-Time({time_units})'].append(np.float16((frame_num - frame_start) * frame_interval))
                tracker_data['1-Frame'].append(frame_num - frame_start)
                tracker_data['1-Tracker'].append(i + 1)
                tracker_data['1-x (px)'].append(int((marker_center[0] / scale_factor)))  # scale back to the original frame resolution
                tracker_data['1-y (px)'].append(int((marker_center[1] / scale_factor)))
                cv2.rectangle(scaled_frame, (x_bbox, y_bbox), (x_bbox + w_bbox, y_bbox + h_bbox), (0, 255, 0), 2)  # update tracker rectangle

            else:
                msg = "WARNING: Lost tracking marker\n\nPlease retry after adjusting any of the following:\n\n-Parameters\n-Initial tracker placement\n-Frame selection"
                warning_popup(msg)
                cap.release()
                cv2.destroyAllWindows()
                return

        cv2.imshow("Tracking...", scaled_frame)  # show updated frame tracking

        if cv2.waitKey(1) == 27 or frame_num >= frame_end:  # cut tracking loop short if ESC hit
            break
    
    record_data(file_mode, tracker_data, "output/Tracking_Output.csv")

    cap.release()
    cv2.destroyAllWindows()


def necking_point(
        cap,
        frame_start,
        frame_end,
        percent_crop_left,
        percent_crop_right,
        binarize_intensity_thresh,
        frame_record_interval,
        frame_interval,
        time_units,
        file_mode,
        video_file_name,
        data_label
    ):

    """necking point detection loop
    Detects and records the necking point in each frame of a video. The necking point is defined as the shortest vertical distance 
    between two horizontal edges within a specified region of interest. The function preprocesses the frames, performs edge detection, 
    and identifies the top and bottom edges to calculate the necking point, highlighting it in red on the visual output.

    Functionality:
        1. Processes each frame based on the specified interval, applying grayscale conversion, binarization, and edge detection.
        2. Identifies and calculates the shortest vertical distance between detected edges.
        3. Records the x-coordinate and the vertical distance of the necking point in each frame.
        4. Outputs visual representation with necking points highlighted.
        5. Saves the recorded data to a CSV file, either overwriting or appending, based on the file mode.

    Args:
        cap (cv2.VideoCapture): Video capture object loaded with the video.
        frame_start (int): Frame number to start processing.
        frame_end (int): Frame number to end processing.
        percent_crop_left (float): Percentage of the frame's left side to exclude from processing.
        percent_crop_right (float): Percentage of the frame's right side to exclude from processing.
        binarize_intensity_thresh (int): Threshold for binarization of the frame to facilitate edge detection.
        frame_record_interval (int): Interval at which frames are processed.
        frame_interval (float): Real-world time interval between frames, used in time calculations.
        time_units (str): Units of time (e.g., 'seconds', 'minutes') for the output data.
        file_mode (FileMode): Specifies whether to overwrite or append data in the output file.
        video_file_name (str): Name of the video file being processed.
        data_label (str): Unique identifier for the data session.
    """    
    x_interval = 50 # interval for how many blue line visuals to display
    frame_num = frame_start
    dist_data = {'1-Frame': [], f'1-Time({time_units})': [], '1-x at necking point (px)': [], '1-y necking distance (px)': [], '1-video_file_name': video_file_name, '1-detection_method': 'min_distance', '1-data_label': data_label}
    percent_crop_left *= 0.01
    percent_crop_right *= 0.01

    while True:  # read frame by frame until the end of the video
        ret, frame = cap.read()
        frame_num += frame_record_interval
        if frame_record_interval != 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

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
            dist_data[f'1-Time({time_units})'].append(np.float16((frame_num - frame_start) / cap.get(5)))
        else:
            dist_data[f'1-Time({time_units})'].append(np.float16((frame_num - frame_start) * frame_interval))
        dist_data['1-Frame'].append(frame_num - frame_start)
        dist_data['1-x at necking point (px)'].append(int(x_samples[necking_pt_ind] / scale_factor))
        dist_data['1-y necking distance (px)'].append(int(necking_distance / scale_factor))

        cv2.line(frame_draw, (x_samples[necking_pt_ind], y_line_values[necking_pt_ind][0]), (x_samples[necking_pt_ind], y_line_values[necking_pt_ind][1]), (0, 0, 255), 2)     

        cv2.imshow('Necking Point Visualization', frame_draw)
        
        if cv2.waitKey(1) == 27 or frame_end <= frame_num:
            break

    record_data(file_mode, dist_data, "output/Necking_Point_Output.csv")

    cap.release()
    cv2.destroyAllWindows()



def necking_point_midpoint(
        cap,
        marker_positions,
        first_frame,
        bbox_size,
        frame_start,
        frame_end,
        binarize_intensity_thresh,
        frame_record_interval,
        frame_interval,
        time_units,
        file_mode,
        video_file_name,
        data_label
    ):

    """necking point detection loop - midpoint method
    This function detects and records the midpoint between two tracked markers
    across the frames of a video, defining this midpoint as the "necking point."
    It processes each frame to apply grayscale conversion, binarization,
    and edge detection. The function calculates the midpoint x-coordinate between
    the markers and measures the vertical distance at this x-coordinate between
    the top and bottom detected edges. The results are visually represented and
    recorded for analysis.

    Functionality:
        1. Processes frames based on specified intervals, applying grayscale conversion,
           binarization, and edge detection.
        2. Tracks marker positions using predefined bounding boxes.
        3. Calculates the x-coordinate of the midpoint between two markers.
        4. Identifies vertical edges at this midpoint and calculates the vertical distance.
        5. Highlights the necking point in the visual output.
        6. Saves the recorded data to a CSV file, either overwriting or appending,
           based on the specified file mode.

    Args:
        cap (cv2.VideoCapture): Video capture object loaded with the video.
        marker_positions (list of tuples): Initial positions (x, y) of the markers.
        first_frame (np.array): The first frame from the video used to initialize trackers.
        bbox_size (int): Size of the bounding box for tracking in pixels.
        frame_start (int): Frame number to start processing.
        frame_end (int): Frame number to end processing.
        binarize_intensity_thresh (int): Threshold for frame binarization to facilitate edge detection.
        frame_record_interval (int): Interval at which frames are processed (e.g., every nth frame).
        frame_interval (float): Real-world time interval between frames, used in time calculations.
        time_units (str): Units of time (e.g., 'seconds', 'minutes') for the output data.
        file_mode (FileMode): Specifies whether to overwrite or append data in the output file.
        video_file_name (str): Name of the video file being processed.
        data_label (str): Unique identifier for the data session.

    Notes:
        - The function assumes two markers are being tracked. Error checking is done
        in the TrackingUI class in main.py before function is called.
        - Proper setting of the binarization threshold is crucial for accurate edge detection.
          The function includes error checks and pop-ups to adjust this if needed.
    """
    trackers = init_trackers(marker_positions, bbox_size, first_frame)

    frame_num = frame_start
    dist_data = {'1-Frame': [], f'1-Time({time_units})': [], '1-x at necking point (px)': [], '1-y necking distance (px)': [], '1-video_file_name': video_file_name, '1-detection_method': 'midpt', '1-data_label': data_label}

    while True:  # read frame by frame until the end of the video
        ret, frame = cap.read()
        frame_num += frame_record_interval
        if frame_record_interval != 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

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

        frame_draw = scaled_frame.copy()
        frame_draw[edges > 0] = [0, 255, 0]  # draw edges
        
        # update trackers and find their current center location
        marker_centers = []
        for i, tracker in enumerate(trackers):
            success, bbox = tracker.update(scaled_frame)

            if success:
                x_bbox, y_bbox, w_bbox, h_bbox = [int(coord) for coord in bbox]  # get coords of bbox in the scaled frame
                marker_center = (x_bbox + w_bbox // 2, y_bbox + h_bbox // 2)  # get center of bbox
                marker_centers.append(marker_center)

        # find x midpoint between markers
        x_coords = [coord[0] for coord in marker_centers]
        left_marker_x = np.min(x_coords)
        right_marker_x = np.max(x_coords)
        mid_x = (right_marker_x + left_marker_x) // 2


        edge_pixels = np.nonzero(edges[:, mid_x])[0]  # find y coord of edge pixels in cur column

        if edge_pixels.size > 0:  # if edge pixels in cur column,
            midpt_dist = np.abs(edge_pixels[0] - edge_pixels[-1])  # find distance of top and bottom edges

            # draw vertical line connecting edges at midpt for visualization
            cv2.line(frame_draw, (mid_x, edge_pixels[0]), (mid_x, edge_pixels[-1]), (200, 0, 0), 1)

        # record and save data using original resolution
        if frame_interval == 0:
            dist_data[f'1-Time({time_units})'].append(np.float16((frame_num - frame_start) / cap.get(5)))
        else:
            dist_data[f'1-Time({time_units})'].append(np.float16((frame_num - frame_start) * frame_interval))
        dist_data['1-Frame'].append(frame_num - frame_start)
        dist_data['1-x at necking point (px)'].append(int(mid_x / scale_factor))
        dist_data['1-y necking distance (px)'].append(int(midpt_dist / scale_factor))

        cv2.imshow('Necking Point (midpoint method) Visualization', frame_draw)
        
        if cv2.waitKey(1) == 27 or frame_end <= frame_num:
            break

    record_data(file_mode, dist_data, "output/Necking_Point_Output.csv")

    cap.release()
    cv2.destroyAllWindows()

def necking_point_step_approximation(
        cap,
        frame_start,
        frame_end,
        percent_crop_left,
        percent_crop_right,
        binarize_intensity_thresh,
        num_steps,
        frame_record_interval,
        frame_interval,
        time_units,
        file_mode,
        video_file_name,
        data_label
    ):
    
    dist_data = {'1-Frame': [], f'1-Time({time_units})': [], '1-x at necking point (px)': [], '1-y necking distance (px)': [], '1-video_file_name': video_file_name, '1-detection_method': 'step', '1-data_label': data_label}
    frame_num = frame_start
    percent_crop_left *= 0.01
    percent_crop_right *= 0.01

    while True:  # read frame by frame until the end of the video
        ret, frame = cap.read()
        #frame = rotate_image(frame, 15) #used to test if lines were being drawn horizontally
        frame_num += frame_record_interval
        if frame_record_interval != 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

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
        frame_draw[edges>0] = [0,255,0]

        # remove x% of edges from consideration of detection
        horizontal_pixels_left = 0
        horizontal_pixels_right = scaled_frame.shape[1]
        if percent_crop_left != 0.:
            left_pixels_removed = int(percent_crop_left * scaled_frame.shape[1])
            horizontal_pixels_left = max(0, left_pixels_removed)
        if percent_crop_right != 0.:
            right_pixels_removed = int(percent_crop_right * scaled_frame.shape[1])
            horizontal_pixels_right = min(scaled_frame.shape[1], scaled_frame.shape[1] - right_pixels_removed)

        step_length = (horizontal_pixels_right - horizontal_pixels_left) // num_steps
        for x in range(horizontal_pixels_left, horizontal_pixels_right, step_length):
            top_edge_pixels = []
            bottom_edge_pixels = []
            for col in range(x, min(x+step_length, horizontal_pixels_right)):
                col_edge_pixels = np.nonzero(edges[:, col])[0] # find row indices of edge pixels in cur col
                
                if col_edge_pixels.size > 0:
                    top_edge_pixels.append(col_edge_pixels[0])
                    bottom_edge_pixels.append(col_edge_pixels[-1])

            if top_edge_pixels and bottom_edge_pixels:
                avg_y_top = np.mean(top_edge_pixels) 
                avg_y_bottom = np.mean(bottom_edge_pixels)  

                x_samples.append(x + step_length // 2)
                y_line_values.append((avg_y_top, avg_y_bottom))
                dist = np.abs(avg_y_top - avg_y_bottom)
                y_distances.append(dist)                

            # draw blue lines visualizing steps
            cv2.line(frame_draw, (x, int(avg_y_top)), (x + step_length, int(avg_y_top)), (200, 0, 0), 1)
            cv2.line(frame_draw, (x, int(avg_y_bottom)), (x + step_length, int(avg_y_bottom)), (200, 0, 0), 1)

        if y_distances:  # check if y_distances is not empty
            # find index of smallest distance
            necking_distance = np.min(y_distances)
            necking_pt_indices = np.where(y_distances == necking_distance)[0]
            necking_pt_ind = int(np.median(necking_pt_indices))

             # draw lines for steps where current min distance is
            cv2.line(frame_draw, (x_samples[necking_pt_ind] - step_length // 2, int(y_line_values[necking_pt_ind][0])),
                     (x_samples[necking_pt_ind] + step_length // 2, int(y_line_values[necking_pt_ind][0])), (0, 0, 255), 2)
            cv2.line(frame_draw, (x_samples[necking_pt_ind] - step_length // 2, int(y_line_values[necking_pt_ind][1])),
                     (x_samples[necking_pt_ind] + step_length // 2, int(y_line_values[necking_pt_ind][1])), (0, 0, 255), 2)


            # record and save data using original resolution
            if frame_interval == 0:
                dist_data[f'1-Time({time_units})'].append(np.float16((frame_num - frame_start) / cap.get(5)))
            else:
                dist_data[f'1-Time({time_units})'].append(np.float16((frame_num - frame_start) * frame_interval))
            dist_data['1-Frame'].append(frame_num - frame_start)
            dist_data['1-x at necking point (px)'].append(int(x_samples[necking_pt_ind] / scale_factor))
            dist_data['1-y necking distance (px)'].append(int(necking_distance / scale_factor))

           
        cv2.imshow('Necking Point Visualization', frame_draw)
        
        if cv2.waitKey(1) == 27 or frame_end <= frame_num:
            break

    record_data(file_mode, dist_data, "output/Necking_Point_Output.csv")

    cap.release()
    cv2.destroyAllWindows()



def noise_reduction(frame):
    '''WIP function for noisy (staticy) viscoelastic substrate frames'''
    # Apply Median Filtering
    median_filtered = cv2.medianBlur(frame, 7)

    # Apply Non-local Means Denoising
    non_local_means = cv2.fastNlMeansDenoising(median_filtered, h=15, templateWindowSize=5, searchWindowSize=17)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
    equalized = clahe.apply(non_local_means)
    return equalized


def improve_binarization(frame):    
    """
    Enhances the binarization of a grayscale image using various image processing techniques. This function applies
    CLAHE for contrast enhancement, background subtraction to highlight foreground objects, morphological operations
    to refine the image, and edge detection to further define object boundaries.

    Steps:
        1. Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to boost the contrast of the image.
        2. Perform background subtraction using a median blur to isolate foreground features.
        3. Apply morphological closing to close small holes within the foreground objects.
        4. Detect edges using the Canny algorithm, and dilate these edges to enhance their visibility.
        5. Optionally adjust the edge thickness with additional morphological operations like dilation or erosion
           depending on specific requirements (commented out in the code but can be adjusted as needed).

    Note:
        - This function is designed to work with grayscale images and expects a single-channel input.
        - Adjustments to parameters like CLAHE limits, kernel sizes for morphological operations, and Canny thresholds
          may be necessary depending on the specific characteristics of the input image.


    Args:
        frame (np.array): A single-channel (grayscale) image on which to perform binarization improvement.

    Returns:
        np.array: The processed image with enhanced binarization and clearer object definitions.

    """

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # boosts contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
    equalized = clahe.apply(frame)
    
    # Perform Background Subtraction
    # (Assuming a relatively uniform background)
    background = cv2.medianBlur(equalized, 13)
    subtracted = cv2.subtract(equalized, background)
    
    # Use morphological closing to close small holes inside the foreground
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(subtracted, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Use Canny edge detector to find edges and use it as a mask
    edges = cv2.Canny(closing, 30, 140)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    result = cv2.bitwise_or(closing, edges_dilated)

    # if edges too fine after result
    #result = cv2.dilate(result, kernel, iterations=1)
    # if edges too thick after result
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.erode(result, kernel, iterations=1)

    return result

def improve_smoothing(frame, strength=0.9):
    """
    Enhances the input frame by applying Non-Local Means (NLM) denoising followed by a high-pass filter.
    
    NLM averages pixel intensity s.t. similar patches of the image (even far apart) contribute more to the average
    

    Args:
    frame (numpy.ndarray): The input image in grayscale.

    Returns:
    numpy.ndarray: The processed image with improved smoothing and enhanced details.
    """
    noise_level = np.std(frame)
    denoised = cv2.fastNlMeansDenoising(frame, None, h=noise_level*strength, templateWindowSize=7, searchWindowSize=25)
    
    # highlights cental pixel and reduces neighboring pixels
    # passes high frequencies and attenuates low frequencies
    # this kernel represents a discrete ver of the laplacian operator, approximating 2nd order derivative of image
    laplacian_kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])

    # Apply the high-pass filter using convolution
    high_pass = cv2.filter2D(denoised, -1, laplacian_kernel)
    return high_pass

def track_area(
        cap,
        marker_positions,
        first_frame,
        bbox_size,
        frame_start,
        frame_end,
        frame_record_interval,
        frame_interval,
        time_units,
        distance_from_marker_thresh,
        file_mode,
        video_file_name,
        data_label,
        preprocessing_need,
        preprocessVals = None
    ):

    """
    Tracks the surface area of an object in a video, given initial marker positions. This function initializes trackers on the object,
    processes each frame to apply binarization and contour detection, and records the largest contour's area that is close to the marker.

    Functionality:
        1. Initializes trackers using CSRT algorithm at the marker positions in the scaled first frame.
        2. Processes each frame: scales, converts to grayscale, and applies binarization improvements.
        3. Applies adaptive thresholding to the binary frame to enhance contour detection.
        4. Identifies contours and selects the optimal one based on area size and proximity to the initial marker.
        5. Draws the contour and marker on the frame for visualization.
        6. Records the position and area of the selected contour via its centroid.
        7. Saves or appends the tracking data to a CSV file based on the specified file mode.
        8. Displays the tracking in real-time and provides an option to exit by pressing ESC.

    Note:
        - Ensure the video file is properly loaded and parameters are correctly set for accurate tracking.
        - Binarization and contour detection settings might need adjustments based on different video characteristics.

    Args:
        cap (cv2.VideoCapture): Video capture object loaded with the video.
        marker_positions (list of tuples): Initial positions of markers.
        first_frame (np.array): First frame of the video for initializing the tracker.
        bbox_size (int): Size of the bounding box for the tracker.
        frame_start (int): Starting frame number for tracking.
        frame_end (int): Ending frame number for tracking.
        frame_record_interval (int): Number of frames to skip between recordings.
        frame_interval (float): Real-world time interval between frames, used in time calculations.
        time_units (str): Units of time (e.g., 'seconds', 'minutes') for the output data.
        distance_from_marker_thresh (int): Threshold for determining proximity of the contour to the marker.
        file_mode (FileMode enum): Mode for recording data ('FileMode.APPEND' or 'FileMode.OVERWRITE').
        video_file_name (str): Name of the video file being processed.
        data_label (str): Unique identifier for the data session.
        preprocessing_need (PreprocessingIssue enum): type of preprocessing that is need for the video
        """

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    frame_num = frame_start
    area_data = {
        '1-Frame': [],
        f'1-Time({time_units})': [],
        '1-x centroid location': [],
        '1-y centroid location': [],
        '1-cell surface area (px^2)': [],
        '1-video_file_name': video_file_name,
        '1-data_label': data_label
    }

    if preprocessVals is not None:
        first_frame = preprocess_frame(first_frame, preprocessVals)


    trackers = init_trackers(marker_positions, bbox_size, improve_smoothing(first_frame), TrackerChoice.CSRT)

    while frame_num < frame_end:
        ret, frame = cap.read()
        frame_num += frame_record_interval
        if frame_record_interval != 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        if not ret:
            break

        scaled_frame, scale_factor = scale_frame(frame)  # scale the frame

        gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)

        # Frame preprocessing
        if preprocessVals is not None:
            preprocessed_frame = preprocess_frame(gray_frame, preprocessVals)
        else:
            preprocessed_frame = gray_frame

        # preprocessing
        #if preprocessing_need == PreprocessingIssue.NOISY_BG:
        #    preprocessed_frame = improve_binarization(gray_frame)
        #elif preprocessing_need == PreprocessingIssue.HARSH_GRADIENT:
        #    preprocessed_frame = improve_smoothing(gray_frame)
        #else:
        #    preprocessed_frame = gray_frame

        # preprocessed_frame = scaled_frame

        # update tracker position
        success, bbox = trackers[0].update(preprocessed_frame) # currently only 1 tracker will work for testing
        if success:
            x_bbox, y_bbox, w_bbox, h_bbox = [int(coord) for coord in bbox]  # get coords of bbox in the scaled frame
            marker_center = (x_bbox + w_bbox // 2, y_bbox + h_bbox // 2)  # get center of bbox
            cv2.rectangle(scaled_frame, (x_bbox, y_bbox), (x_bbox + w_bbox, y_bbox + h_bbox), (0, 255, 0), 2)  # update tracker rectangle
        else:
            msg = "WARNING: Lost tracking marker\n\nPlease retry after adjusting any of the following\n\n: -Parameters\n-Initial tracker placement\n-Frame selection"
            warning_popup(msg)
            cap.release()
            cv2.destroyAllWindows()
            return

        binary_frame = cv2.adaptiveThreshold(preprocessed_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Segment frame
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # choose optimal contour (largest and near marker)
        max_area, max_area_idx = 0, 0
        centroid = None
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if np.abs(marker_center[0] - cx) < distance_from_marker_thresh and np.abs(marker_center[1] - cy) < distance_from_marker_thresh:
                    if area > max_area:
                        max_area = area
                        max_area_idx = i
                        centroid = (cx,cy)

        # draw the chosen contour
        contour = contours[max_area_idx]
        if centroid:
            cv2.circle(scaled_frame, centroid, 5, (0, 0, 255), -1)
        cv2.drawContours(scaled_frame, [contour], -1, (255, 0, 0), 2)
        cv2.putText(scaled_frame, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # record data
        print(centroid)
        try: 
            area_data['1-Frame'].append(frame_num)
            if frame_interval == 0:
                area_data[f'1-Time({time_units})'].append(np.float32(frame_num / cap.get(5)))
            else:
                area_data[f'1-Time({time_units})'].append(np.float16(frame_num * frame_interval))
            area_data['1-x centroid location'].append(int((centroid[0] / scale_factor)))
            area_data['1-y centroid location'].append(int((centroid[1] / scale_factor)))
            area_data['1-cell surface area (px^2)'].append(max_area)
        except TypeError:
            print("centroid not found")


        #cv2.imshow('Surface Area Tracking', noise_reduced_frame)
        cv2.imshow('Surface Area Tracking', scaled_frame)

        if cv2.waitKey(1) == 27:
            break

        frame_num += 1

    record_data(file_mode, area_data, "output/Surface_Area_Output.csv")

    cap.release()
    cv2.destroyAllWindows()

