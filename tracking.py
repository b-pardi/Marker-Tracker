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


def select_markers(cap, bbox_size, frame_start):
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
    # create trackers
    trackers = []
    if tracker_choice == TrackerChoice.KCF:
        for _ in range(len(marker_positions)):
            trackers.append(cv2.TrackerKCF_create())
    elif tracker_choice == TrackerChoice.CSRT:
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
    tracker_data = {'Frame': [], f'Time({time_units})': [], '1-Tracker': [], '1-x (px)': [], '1-y (px)': [], '1-video_file_name': video_file_name, '1-data_label': data_label}
    frame_num = 0

    # tracking loop
    while True:
        ret, frame = cap.read()
        frame_num += frame_record_interval
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
                    tracker_data[f'Time({time_units})'].append(np.float32(frame_num / cap.get(5)))
                else:
                    tracker_data[f'Time({time_units})'].append(np.float16(frame_num * frame_interval))
                tracker_data['Frame'].append(frame_num)
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

        if cv2.waitKey(1) == 27 or frame_num >= frame_end - frame_start:  # cut tracking loop short if ESC hit
            break
    
    if file_mode == FileMode.OVERWRITE:
        tracker_df = pd.DataFrame(tracker_data)
        tracker_df.set_index('Frame', inplace=True)
        tracker_df.to_csv("output/Tracking_Output.csv")
    elif file_mode == FileMode.APPEND:
        tracker_df = pd.read_csv("output/Tracking_Output.csv")
        num_prev_trackers = int(tracker_df.columns[-1][0])
        print(f"Num previous tracked entities: {num_prev_trackers}")

        # create new df of current tracking data and merge into previous
        cur_df = pd.DataFrame(tracker_data)

        # drop time col since time col in prev df should match
        cur_df.drop(f'Time({time_units})', axis=1, inplace=True)

        # rename df to indicate which num tracker this is
        cur_df.rename(columns={
                '1-Tracker': f'{num_prev_trackers+1}-Tracker',
                '1-x (px)': f'{num_prev_trackers+1}-x (px)',
                '1-y (px)': f'{num_prev_trackers+1}-y (px)',
                '1-video_file_name' : f'{num_prev_trackers+1}-video_file_name',
                '1-data_label': f'{num_prev_trackers+1}-data_label'
            }, inplace=True)
        
        tracker_df_merged = pd.merge(tracker_df, cur_df, on='Frame', how='outer')
        tracker_df_merged.to_csv("output/Tracking_Output.csv", index=False)

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
    frame_num = 0
    dist_data = {'Frame': [], f'Time({time_units})': [], '1-x at necking point (px)': [], '1-y necking distance (px)': [], '1-video_file_name': video_file_name, '1-data_label': data_label}
    percent_crop_left *= 0.01
    percent_crop_right *= 0.01

    while True:  # read frame by frame until the end of the video
        ret, frame = cap.read()
        frame_num += frame_record_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start + frame_num)

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
        dist_data['1-x at necking point (px)'].append(int(x_samples[necking_pt_ind] / scale_factor))
        dist_data['1-y necking distance (px)'].append(int(necking_distance / scale_factor))

        cv2.line(frame_draw, (x_samples[necking_pt_ind], y_line_values[necking_pt_ind][0]), (x_samples[necking_pt_ind], y_line_values[necking_pt_ind][1]), (0, 0, 255), 2)     

        cv2.imshow('Necking Point Visualization', frame_draw)
        if cv2.waitKey(1) == 27 or frame_end <= frame_num + frame_start:
            break

    if file_mode == FileMode.OVERWRITE:
        dist_df = pd.DataFrame(dist_data)
        dist_df.set_index('Frame', inplace=True)
        dist_df.to_csv("output/Necking_Point_Output.csv")
    elif file_mode == FileMode.APPEND:
        dist_df = pd.read_csv("output/Necking_Point_Output.csv")
        num_prev_trackers = int(dist_df.columns[-1][0])
        print(f"Num previous tracked entities: {num_prev_trackers}")

        # create new df of current tracking data and merge into previous
        cur_df = pd.DataFrame(dist_data)

        # drop time col since time col in prev df should match
        cur_df.drop(f'Time({time_units})', axis=1, inplace=True)

        # rename df to indicate which num tracker this is
        cur_df.rename(columns={
                '1-x at necking point (px)': f'{num_prev_trackers+1}-x at necking point (px)',
                '1-y necking distance (px)': f'{num_prev_trackers+1}-y necking distance (px)',
                '1-video_file_name' : f'{num_prev_trackers+1}-video_file_name',
                '1-data_label': f'{num_prev_trackers+1}-data_label'
            }, inplace=True)
        
        dist_df_merged = pd.merge(dist_df, cur_df, on='Frame', how='outer')
        dist_df_merged.to_csv("output/Necking_Point_Output.csv", index=False)

    cap.release()
    cv2.destroyAllWindows()


def noise_reduction(frame):
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
        data_label
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
        6. Records the position and area of the selected contour.
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
        file_mode (FileMode): Specifies whether to overwrite or append data in the output file.
        video_file_name (str): Name of the video file being processed.
        data_label (str): Unique identifier for the data session.
        """

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    frame_num = frame_start
    area_data = {'Frame': [], f'Time({time_units})': [], '1-x cell location': [], '1-y cell location': [], '1-cell surface area (px^2)': [], '1-video_file_name': video_file_name, '1-data_label': data_label}
    
    # init trackers
    trackers = []
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

    while frame_num < frame_end:
        ret, frame = cap.read()
        frame_num += frame_record_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        if not ret:
            break

        # Frame preprocessing
        scaled_frame, scale_factor = scale_frame(frame)  # scale the frame
        gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)

        # update tracker position
        success, bbox = trackers[0].update(scaled_frame) # currently only 1 tracker will work for testing
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

        # preprocessing

        # reduce static noise (WIP)
        #noise_reduced_frame = noise_reduction(gray_frame)

        #_, binary_frame = cv2.threshold(blur_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_frame = improve_binarization(gray_frame)

        # threshold frame again with localized kernel, adds a bit of noise but strengthens contours (idk why this helps but it does)
        adaptive_thresh = cv2.adaptiveThreshold(binary_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Segment frame
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        '''for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 100:  # Filter out small contours that are small blobs
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if np.abs(marker_center[0] - cx) < distance_from_marker_thresh and np.abs(marker_center[1] - cy) < distance_from_marker_thresh:
                        cv2.drawContours(scaled_frame, [contour], -1, (255, 0, 0), 2)
                        cv2.putText(scaled_frame, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)'''

        # choose optimal contour (largest and near marker)
        max_area, max_area_idx = 0, 0
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

        # draw the chosen contour
        contour = contours[max_area_idx]
        cv2.drawContours(scaled_frame, [contour], -1, (255, 0, 0), 2)
        cv2.putText(scaled_frame, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # record data
        area_data['Frame'].append(frame_num)
        if frame_interval == 0:
            area_data[f'Time({time_units})'].append(np.float32(frame_num / cap.get(5)))
        else:
            area_data[f'Time({time_units})'].append(np.float16(frame_num * frame_interval))
        area_data['1-x cell location'].append(int((marker_center[0] / scale_factor)))
        area_data['1-y cell location'].append(int((marker_center[1] / scale_factor)))
        area_data['1-cell surface area (px^2)'].append(max_area)

        #cv2.imshow('Surface Area Tracking', noise_reduced_frame)
        cv2.imshow('Surface Area Tracking', scaled_frame)
        if cv2.waitKey(1) == 27:
            break

        frame_num += 1

    if file_mode == FileMode.OVERWRITE:
        area_df = pd.DataFrame(area_data)
        area_df.set_index('Frame', inplace=True)
        area_df.to_csv("output/Surface_Area_Output.csv")
    elif file_mode == FileMode.APPEND:
        area_df = pd.read_csv("output/Surface_Area_Output.csv")
        num_prev_trackers = int(area_df.columns[-1][0])
        print(f"Num previous tracked entities: {num_prev_trackers}")

        # create new df of current tracking data and merge into previous
        cur_df = pd.DataFrame(area_data)

        # drop time col since time col in prev df should match
        cur_df.drop(f'Time({time_units})', axis=1, inplace=True)

        # rename df to indicate which num tracker this is
        cur_df.rename(columns={
                '1-x cell location': f'{num_prev_trackers+1}-x cell location',
                '1-y cell location': f'{num_prev_trackers+1}-y cell location',
                '1-cell surface area (px^2)': f'{num_prev_trackers+1}-cell surface area (px^2)',
                '1-video_file_name' : f'{num_prev_trackers+1}-video_file_name',
                '1-data_label': f'{num_prev_trackers+1}-data_label'
            }, inplace=True)
        
        area_df_merged = pd.merge(area_df, cur_df, on='Frame', how='outer')
        area_df_merged.to_csv("output/Surface_Area_Output.csv", index=False)

    cap.release()
    cv2.destroyAllWindows()


def track_area_og(cap, frame_start, frame_end, frame_interval, time_units):
    frame_num = frame_start

    while True:  # read frame by frame until the end of the video
        ret, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start + frame_num)
        frame_num += 1
        #time.sleep(0.25)

        if not ret:
            break

        # frame preprocessing
        scaled_frame, scale_factor = scale_frame(frame)  # scale the frame
        gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)  # convert frame to gray
        kernel_size = (5,5)
        blur_frame = cv2.GaussianBlur(gray_frame, kernel_size, 0) # gaussian blur
        _, binary_frame = cv2.threshold(blur_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # threshold to binarize image

        # segment frame
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find contours
        contour_frame = np.zeros_like(binary_frame) 
        cv2.drawContours(scaled_frame, contours, -1, (255, 0, 0), 2)

        # label and segment contours
        for i, contour in enumerate(contours):
            m = cv2.moments(contour) # calculate moment contours
            if m["m00"] != 0: # if contour has a non-zero zeroth spatial moment (surface area)
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                cv2.putText(scaled_frame, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    

        cv2.imshow('Surface Area Tracking', scaled_frame)
        if cv2.waitKey(1) == 27 or frame_end == frame_num+frame_start:
            break

    cap.release()
    cv2.destroyAllWindows()