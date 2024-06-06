def frame_capture_thread(
        cap,
        frame_queue,
        message_queue,
        stop_event,
        frame_end,
        frame_start,
        frame_record_interval
    ):
    """
    Thread function to capture video frames from a specified start to end frame at given intervals and push them to a processing queue.

    Details:
        - Sequentially reads frames from the video capture object `cap`, starting at `frame_start` and ending at `frame_end`, at intervals defined by `frame_record_interval`.
        - Each frame read is placed into `frame_queue` for further processing by other threads.
        - If frames cannot be read or the queue is full, appropriate actions are taken, including setting a stop event or logging a message.
        - The function also listens for a stop event that, when set, will terminate the frame capture early, allowing for graceful shutdowns.

    Note:
        - This function is crucial for maintaining a steady flow of frames to processing threads in a multithreaded video analysis application.
        - Proper error handling and inter-thread communication are vital to ensure robust performance and to handle any issues that arise during frame capture.
        - The frame capture process stops either when all frames are read, an error occurs, or a stop event is triggered.
        - All threaded tracking functions utilize this as their first thread

    Args:
        cap (cv2.VideoCapture): The video capture object from which frames are read.
        frame_queue (queue.Queue): The queue to which frames are pushed for further processing.
        message_queue (queue.Queue): Queue for passing error messages or important notifications.
        stop_event (threading.Event): An event that can be set to signal the thread to stop processing.
        frame_end (int): The frame number at which to stop capturing.
        frame_start (int): The frame number at which to start capturing.
        frame_record_interval (int): The number of frames to skip between captures, defining the capture interval.

    Returns:
        None: This function does not return any values but signals completion or errors through the `frame_queue` and `message_queue`.
    """
    frame_num = frame_start
    while frame_num <= frame_end and not stop_event.is_set():
        #print(f"cur frame num in capture thread: {frame_num}")
        if stop_event.is_set():
            break
        ret, frame = cap.read()
        
        # only need to concern with setting frames if interval other than 1, which is default of cap.read()
        if frame_record_interval != 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        if not ret:
            #message_queue.put("Error: Frame failed to read")
            stop_event.set()
            break
        try:
            frame_queue.put((frame_num, frame), block=True, timeout=1)
        except queue.Full:
            print("QUEUE FULL FRAME DROPPED")
        frame_num += frame_record_interval
        
        if cv2.waitKey(1) == 27: 
            stop_event.set()
            break
    frame_queue.put(None) # sentinel value

def frame_tracker_processing_thread(
        frame_queue,
        message_queue,
        stop_event,
        trackers,
        scale_frame,
        tracker_data,
        frame_start,
        frame_end,
        frame_interval,
        time_units,
        fps
    ):
    """
    Thread function that processes video frames to track markers using specified tracking algorithms. 
    It retrieves frames from a queue, applies scaling and tracking, and records the results in a shared data structure.

    Details:
        - Pulls video frames from a queue for processing, applies scaling to adjust frame size as necessary, and uses tracking algorithms to locate markers.
        - Updates a shared dictionary with tracking information, including the frame number, time, and marker coordinates.
        - Manages thread termination based on a stop event, and communicates any issues via a message queue.
        - This function is part of a multithreaded tracking system designed to handle large video files or real-time video streams efficiently.

    Note:
        - Ensure that the tracking data structure is thread-safe if accessed by multiple threads.
        - Adjustments in tracker settings or frame processing parameters might be necessary depending on the specific requirements of the tracking task.
        - Error handling includes sending messages about tracking failures or other issues and may signal other threads to stop if critical errors occur.

    Args:
        frame_queue (queue.Queue): Queue from which the thread retrieves frames to process.
        message_queue (queue.Queue): Queue for sending error messages or important notifications.
        stop_event (threading.Event): Event to signal the thread to stop processing.
        trackers (list): List of tracker objects used to track markers in the frames.
        scale_frame (function): Function to scale frames before processing.
        tracker_data (dict): Dictionary to store results of the tracking.
        frame_start (int): Frame number where processing starts.
        frame_end (int): Frame number where processing ends.
        frame_interval (float): Interval between frames, used in calculations for time data.
        time_units (str): Units of time used for recording in the output data.
        fps (int): Frames per second of the video, used in time calculations.

    Returns:
        None: This function does not return any values but updates the tracker_data dictionary and handles threads' synchronization and communication.
    """
    while True:
        try:
            queue_item = frame_queue.get(timeout=1)  # Get frames from the queue
        except queue.Empty:
            continue

        if queue_item is None:
            break
        else:
            frame_num, frame = queue_item
            #print(f"cur frame num in processing thread: {frame_num}")

        if frame_num >= frame_end:
            break

        scaled_frame, scale_factor = scale_frame(frame)

        for i, tracker in enumerate(trackers):
            success, bbox = tracker.update(scaled_frame)
            if success:
                x_bbox, y_bbox, w_bbox, h_bbox = [int(coord) for coord in bbox]
                marker_center = (x_bbox + w_bbox // 2, y_bbox + h_bbox // 2)

                # record tracker locations using original resolution
                if tracker_data:
                    if frame_interval == 0:
                        tracker_data[f'1-Time({time_units})'].append(np.float16((frame_num - frame_start) / fps))
                    else:
                        tracker_data[f'1-Time({time_units})'].append(np.float16((frame_num - frame_start) * frame_interval))
                    tracker_data['1-Frame'].append(frame_num - frame_start)
                    tracker_data['1-Tracker'].append(i + 1)
                    tracker_data['1-x (px)'].append(int((marker_center[0] / scale_factor)))  # scale back to the original frame resolution
                    tracker_data['1-y (px)'].append(int((marker_center[1] / scale_factor)))
                    
                cv2.rectangle(scaled_frame, (x_bbox, y_bbox), (x_bbox + w_bbox, y_bbox + h_bbox), (0, 255, 0), 2)  # update tracker rectangle

            else:
                msg = f"WARNING: Lost tracking marker at {bbox}\n\nPlease retry after adjusting any of the following:\n\n-Parameters\n-Initial tracker placement\n-Frame selection"
                message_queue.put(msg)
                stop_event.set()
                break

        cv2.imshow("Tracking", scaled_frame)
        frame_queue.task_done()
        if cv2.waitKey(1) == 27:  # Check for ESC key
            stop_event.set()

def track_markers_threaded(
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
        data_label
    ):
    """
    Executes multithreaded marker tracking within a video sequence, using separate threads for capturing frames
    and processing marker tracking to efficiently gather tracking data.

    Details:
        - Utilizes specialized trackers to follow markers across frames, starting from a specified frame and ending at another.
        - Frame data is captured and processed in parallel threads to enhance performance, especially suitable for long-duration or high-resolution videos.
        - Tracking results, including positions and timestamps, are stored for analysis or subsequent use, and final data is recorded based on specified file modes.

    Note:
        - Ensure all tracker settings and initializations are correct before execution to prevent tracking failures.
        - Proper handling of threading operations is crucial for performance and to prevent data corruption or loss.
        - The function coordinates the stopping and error management via events and queues to handle unforeseen issues during processing.

    Args:
        marker_positions (list of tuples): Initial positions of markers within the first frame.
        first_frame (np.array): The first frame of the video used to initialize the trackers.
        frame_start (int): Frame number at which to start processing.
        frame_end (int): Frame number at which to end processing.
        cap (cv2.VideoCapture): Video capture object loaded with the video.
        bbox_size (int): Size of the bounding box used for each tracker.
        tracker_choice (enum): The type of tracker to be used (e.g., 'KCF', 'CSRT').
        frame_record_interval (int): Interval at which frames are recorded.
        frame_interval (float): Time interval between frames, used for time calculations.
        time_units (str): Units of time for recording in the output data.
        file_mode (FileMode): Specifies whether to overwrite or append data in the output file.
        video_file_name (str): Name of the video file being processed.
        data_label (str): Label to categorize or describe the data session.

    Returns:
        None: The function does not return any values but outputs tracking data to a file and handles threads' synchronization and communication.
    """
    fps = cap.get(5)

    trackers = init_trackers(marker_positions, bbox_size, first_frame, tracker_choice)

    # Initialize tracking data storage
    tracker_data = {
        '1-Frame': [],
        f'1-Time({time_units})': [],
        '1-Tracker': [],
        '1-x (px)': [],
        '1-y (px)': [],
        '1-video_file_name': video_file_name,
        '1-data_label': data_label
    }

    # Create a thread-safe queue and an event to signal thread termination
    frame_queue = queue.Queue(maxsize=10)
    message_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    # Start the capture and processing threads
    capture_thread = threading.Thread(
        target=frame_capture_thread,
        args=(
            cap,
            frame_queue,
            message_queue,
            stop_event,
            frame_end,
            frame_start,
            frame_record_interval
        ), daemon=True
    )
    processing_thread = threading.Thread(
        target=frame_tracker_processing_thread,
        args=(
            frame_queue,
            message_queue,
            stop_event,
            trackers,
            scale_frame,
            tracker_data,
            frame_start,
            frame_end,
            frame_interval,
            time_units,
            fps
        ), daemon=True
    )

    capture_thread.start()
    processing_thread.start()

    # Wait for threads to finish
    capture_thread.join()
    print("capture_thread done")

    processing_thread.join()
    print("processing_thread done")

    # check if either thread failed
    while not message_queue.empty():
        warning_popup(message_queue.get())

    # Data output handling
    record_data(file_mode, tracker_data, "output/Tracking_Output.csv")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


def frame_edge_processing(
        frame_queue,
        message_queue,
        dist_data,
        stop_event,
        scale_frame,
        frame_start,
        frame_end,
        frame_interval,
        time_units,
        fps,
        binarize_intensity_thresh,
        percent_crop_left=0.,
        percent_crop_right=0.,
        x_interval=0
    ):
    """
    Thread function to process video frames for detecting minimal vertical distances at necking points. 
    It retrieves frames from a queue, applies image processing, and identifies minimal distances between edges to determine necking points.

    Details:
        - Converts frames to grayscale and applies binary thresholding to facilitate edge detection.
        - Utilizes the Canny algorithm to detect edges and calculates the minimal vertical distance between top and bottom edges across the frame.
        - Records detected necking point data, including the frame number, time, and the vertical distance, into a shared data dictionary.
        - Visualizes detected edges and necking points on the frames for real-time monitoring and verification.
        - The function dynamically adjusts to cropping parameters to focus analysis on specific regions of interest within the frames.

    Note:
        - Proper synchronization mechanisms using threading events and queues must be in place to manage the flow of data between threads.
        - Parameters for image processing (like binarization thresholds and cropping percentages) may need to be adjusted based on specific video content characteristics.
        - This thread is designed to run as part of a multithreaded application where video processing tasks are distributed among multiple threads for efficiency.

    Args:
        frame_queue (queue.Queue): Queue from which the thread retrieves frames to process.
        message_queue (queue.Queue): Queue for passing error messages or important notifications.
        dist_data (dict): Dictionary to store results of the necking point calculations.
        stop_event (threading.Event): Event to signal the thread to stop processing.
        scale_frame (function): Function to scale frames before processing.
        frame_start (int): Frame number where processing starts.
        frame_end (int): Frame number where processing ends.
        frame_interval (float): Time interval between frames, used in calculations.
        time_units (str): Units of time for output data.
        fps (int): Frames per second of the video, used in time calculations.
        binarize_intensity_thresh (int): Threshold for binarization used in edge detection.
        percent_crop_left (float): Percentage of the frame width to crop from the left side.
        percent_crop_right (float): Percentage of the frame width to crop from the right side.
        x_interval (int): Interval for drawing vertical lines during visualization, providing spatial reference.

    Returns:
        None: This function does not return any value. It manages internal state and communicates via queues.
    """
    while True:
        try:
            queue_item = frame_queue.get(timeout=1)
        except queue.Empty:
            print("QUEUE EMPTY")
            continue

        if queue_item is None:
            break
        else:
            frame_num, frame = queue_item

        if frame_num >= frame_end:
            break

        scaled_frame, scale_factor = scale_frame(frame)

        gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)  # convert frame to gray
        _, binary_frame = cv2.threshold(gray_frame, binarize_intensity_thresh, 255, cv2.THRESH_BINARY)  # threshold to binarize image

        # error checking for appropriate binarization threshold
        if np.all(binary_frame == 255):
            msg = "Binarization threshold too low,\nfound no pixels below the threshold.\n\nPlease adjust the threshold (default is 120)"
            message_queue.put(msg)
            stop_event.set()
        if np.all(binary_frame == 0):
            msg = "Binarization threshold too high,\nfound no pixels above the threshold.\n\nPlease adjust the threshold (default is 120)"
            message_queue.put(msg)
            stop_event.set()

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

                if x % x_interval == 0 and x != 0:  # draw visualization lines at every x_interval pixels
                    # draw vertical lines connecting edges for visualization
                    cv2.line(frame_draw, (x, edge_pixels[0]), (x, edge_pixels[-1]), (200, 0, 0), 1)  

        # find index of smallest distance
        # multiple mins occur in typically close together, for now just pick middle of min occurrences
        necking_distance = np.min(y_distances)
        necking_pt_indices = np.where(y_distances == necking_distance)[0]
        necking_pt_ind = int(np.median(necking_pt_indices))

        # record and save data using original resolution
        if frame_interval == 0:
            dist_data[f'1-Time({time_units})'].append(np.float16((frame_num - frame_start) / fps))
        else:
            dist_data[f'1-Time({time_units})'].append(np.float16((frame_num - frame_start) * frame_interval))
        dist_data['1-Frame'].append(frame_num - frame_start)
        dist_data['1-x at necking point (px)'].append(int(x_samples[necking_pt_ind] / scale_factor))
        dist_data['1-y necking distance (px)'].append(int(necking_distance / scale_factor))

        cv2.line(frame_draw, (x_samples[necking_pt_ind], y_line_values[necking_pt_ind][0]), (x_samples[necking_pt_ind], y_line_values[necking_pt_ind][1]), (0, 0, 255), 2)     

        cv2.imshow('Necking Point Visualization', frame_draw)
        frame_queue.task_done()
        if cv2.waitKey(1) == 27 or frame_end <= frame_num:
            stop_event.set()

def necking_point_threaded(
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

    fps = cap.get(5)
    x_interval = 50 # interval for how many blue line visuals to display
    frame_num = frame_start
    dist_data = {
        '1-Frame': [],
        f'1-Time({time_units})': [],
        '1-x at necking point (px)': [],
        '1-y necking distance (px)': [],
        '1-video_file_name': video_file_name,
        '1-detection_method': 'min_distance',
        '1-data_label': data_label
    }
    """
    Conducts multithreaded analysis to determine the necking point in a video by finding the minimum vertical distance between 
    detected edges across frames. It uses two threads: one for capturing frames and another for processing them to identify necking points.

    Details:
        - The processing involves binarizing the frame, detecting edges, and calculating the minimal vertical distance between these edges.
        - Configurable parameters allow for adjustments to how the frames are processed, such as cropping and binarization threshold.
        - Captured data about the necking points, including their positions and distances, are recorded for further analysis.
        - This method is particularly useful in materials science and engineering contexts where deformation behaviors are studied.

    Note:
        - Ensure the video capture object (`cap`) is properly initialized and capable of seeking to specific frames.
        - Parameters such as cropping percentages and binarization threshold should be tuned based on the specific characteristics of the video content.
        - This function orchestrates the start and coordination of threads, handling synchronization issues such as queue management and thread stopping.

    Args:
        cap (cv2.VideoCapture): Video capture object loaded with the video.
        frame_start (int): Frame number to start processing.
        frame_end (int): Frame number to end processing.
        percent_crop_left (float): Percentage of the frame width to crop from the left side.
        percent_crop_right (float): Percentage of the frame width to crop from the right side.
        binarize_intensity_thresh (int): Threshold for binarization used in edge detection.
        frame_record_interval (int): Interval at which frames are captured.
        frame_interval (float): Real-world time interval between frames, used in time calculations.
        time_units (str): Units of time for the output data.
        file_mode (FileMode enum): Mode for recording data ('FileMode.APPEND' or 'FileMode.OVERWRITE').
        video_file_name (str): Name of the video file being processed.
        data_label (str): Label to categorize or describe the data session.

    Returns:
        None: The function does not return any value but records the processed data to an output file and handles visualizations and errors.
    """
    percent_crop_left *= 0.01
    percent_crop_right *= 0.01

    # Create a thread-safe queue and an event to signal thread termination
    frame_queue = queue.Queue(maxsize=10)
    message_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    # Start the capture and processing threads
    capture_thread = threading.Thread(
        target=frame_capture_thread,
        args=(
            cap,
            frame_queue,
            message_queue,
            stop_event,
            frame_end,
            frame_start,
            frame_record_interval,
        ), daemon=True
    )
    processing_thread = threading.Thread(
        target=frame_edge_processing,
        args=(
            frame_queue,
            message_queue,
            dist_data,
            stop_event,
            scale_frame,
            frame_start,
            frame_end,
            frame_interval,
            time_units,
            fps,
            binarize_intensity_thresh,
            percent_crop_left,
            percent_crop_right,
            x_interval
        ), daemon=True
    )

    capture_thread.start()
    processing_thread.start()

    capture_thread.join()
    processing_thread.join()

    while not message_queue.empty():
        warning_popup(message_queue.get())

    record_data(file_mode, dist_data, "output/Necking_Point_Output.csv")

    cap.release()
    cv2.destroyAllWindows()

    def frame_tracker_midpt_finder_thread(
        frame_queue,
        tracker_midpt_queue,
        message_queue,
        stop_event,
        trackers,
        frame_end
    ):
    """
    Thread function that processes video frames to find the midpoint between markers tracked in each frame.
    It retrieves frames from a queue, applies tracking algorithms, and places results into another queue for further processing.

    Details:
        - The thread retrieves frames from a queue and applies tracking algorithms to identify markers' positions.
        - Calculates the midpoint between two tracked markers and forwards the scaled frame and midpoint information to another processing stage.
        - Handles frame processing from start to end frame, placing results in the tracker midpoint queue.
        - Provides feedback through a message queue if tracking is lost or an error occurs, and can signal other threads to stop via a stop event.

    Note:
        - Ensure the input frame data is suitable for tracking, i.e., the frames should be clear and the markers visible.
        - The function handles errors and tracking failures by sending messages and potentially stopping all processing.
        - This thread uses computer vision techniques that can be computationally intensive, depending on the video resolution and tracker complexity.

    Args:
        frame_queue (queue.Queue): Queue from which the thread retrieves frames to process.
        tracker_midpt_queue (queue.Queue): Queue where processed frame data and midpoints are put for further processing.
        message_queue (queue.Queue): Queue for sending error messages or important notifications.
        stop_event (threading.Event): Event to signal the thread to stop processing.
        trackers (list): List of tracker objects used to track markers in the frames.
        scale_frame (function): Function to scale frames before processing.
        frame_start (int): Frame number where processing starts.
        frame_end (int): Frame number where processing ends.

    Returns:
        None: This function does not return any value. It communicates through queues and updates internal state as needed.
    """
    while True:
        try:
            queue_item = frame_queue.get(timeout=1)  # Get frames from the queue
        except queue.Empty:
            continue
        if queue_item:
            frame_num, frame = queue_item
            #print(f"cur frame num in frame_tracker_midpt_finder thread: {frame_num}")
        else:
            tracker_midpt_queue.put(None)
            break
        if frame_num >= frame_end:
            tracker_midpt_queue.put(None)
            break

        #print(f"cur frame num in frame_tracker_midpt_finder_thread: {frame_num}")

        scaled_frame, scale_factor = scale_frame(frame)
        
        marker_centers = []
        for i, tracker in enumerate(trackers):
            success, bbox = tracker.update(scaled_frame)
            if success:
                x_bbox, y_bbox, w_bbox, h_bbox = [int(coord) for coord in bbox]
                marker_center = (x_bbox + w_bbox // 2, y_bbox + h_bbox // 2)
                marker_centers.append(marker_center)

                # find x midpoint between markers
                x_coords = [coord[0] for coord in marker_centers]
                left_marker_x = np.min(x_coords)
                right_marker_x = np.max(x_coords)
                mid_x = (right_marker_x + left_marker_x) // 2
                
                #cv2.rectangle(scaled_frame, (x_bbox, y_bbox), (x_bbox + w_bbox, y_bbox + h_bbox), (0, 255, 0), 2)  # update tracker rectangle

            else:
                msg = f"WARNING: Lost tracking marker at {bbox}\n\nPlease retry after adjusting any of the following:\n\n-Parameters\n-Initial tracker placement\n-Frame selection"
                message_queue.put(msg)
                stop_event.set()
                break

        tracker_midpt_queue.put((frame_num, scaled_frame, scale_factor, (x_bbox, y_bbox, w_bbox, h_bbox), mid_x))

        #cv2.imshow("Tracking", scaled_frame)
        frame_queue.task_done()
        if cv2.waitKey(1) == 27:  # Check for ESC key
            stop_event.set()

def frame_midpt_edge_detection(
        frame_queue,
        tracker_midpt_queue,
        message_queue,
        dist_data,
        stop_event,
        frame_start,
        frame_end,
        frame_interval,
        time_units,
        fps,
        binarize_intensity_thresh
    ):
    """
    Thread function for detecting midpoints between edges in video frames, specifically focused on identifying and measuring
    necking points within a video sequence. This function retrieves pre-processed frames and tracks specific metrics.

    Details:
        - The function pulls frames from a queue that have been pre-processed for tracker midpoint identification.
        - It applies further image processing to enhance edges and calculate vertical distances at specified midpoints.
        - The thread utilizes adaptive binarization and edge detection techniques to isolate features of interest.
        - Detected midpoints and distances are recorded and visualized in real-time, and data is accumulated for later analysis or output.
        - Thread control is managed through stop events and error messages are communicated through a message queue.

    Note:
        - Proper synchronization mechanisms using threading events and queues must be in place to manage the flow of data between threads.
        - Binarization and edge detection parameters may need to be adjusted based on video content and desired detection sensitivity.
        - This function is designed to run as part of a multithreaded application where video processing tasks are distributed among multiple threads.

    Args:
        frame_queue (queue.Queue): Not directly used in this thread but included for potential future use.
        tracker_midpt_queue (queue.Queue): Queue from which the thread retrieves pre-processed frames.
        message_queue (queue.Queue): Queue for passing error messages or important notifications.
        dist_data (dict): Dictionary to store results of the necking point calculations.
        stop_event (threading.Event): Event to signal the thread to stop processing.
        frame_start (int): Frame number where processing starts.
        frame_end (int): Frame number where processing ends.
        frame_interval (float): Interval between frames, used in calculations for time data.
        time_units (str): Units for time used in the output data.
        fps (int): Frames per second of the video, used in time calculations.
        binarize_intensity_thresh (int): Threshold for binarization used in edge detection.

    Returns:
        None: This function does not return any value but updates the dist_data dictionary and communicates via queues.
    """
    while True:
        try:
            tracker_item = tracker_midpt_queue.get(timeout=1)
        except queue.Empty:
            print("QUEUE EMPTY")
            continue

        if tracker_item is None:
            break

        frame_num, scaled_frame, scale_factor, _, mid_x = tracker_item
        if frame_num >= frame_end:
            break

        gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)  # convert frame to gray
        _, binary_frame = cv2.threshold(gray_frame, binarize_intensity_thresh, 255, cv2.THRESH_BINARY)  # threshold to binarize image

        # error checking for appropriate binarization threshold
        if np.all(binary_frame == 255):
            msg = "Binarization threshold too low,\nfound no pixels below the threshold.\n\nPlease adjust the threshold (default is 120)"
            message_queue.put(msg)
            stop_event.set()
        if np.all(binary_frame == 0):
            msg = "Binarization threshold too high,\nfound no pixels above the threshold.\n\nPlease adjust the threshold (default is 120)"
            message_queue.put(msg)
            stop_event.set()

        edges = cv2.Canny(binary_frame, 0, 2)  # edge detection, nums are gradient thresholds

        frame_draw = scaled_frame.copy()
        frame_draw[edges > 0] = [0, 255, 0]  # draw edges

        edge_pixels = np.nonzero(edges[:, mid_x])[0]  # find y coord of edge pixels in cur column

        if edge_pixels.size > 0:  # if edge pixels in cur column,
            midpt_dist = np.abs(edge_pixels[0] - edge_pixels[-1])  # find distance of top and bottom edges

            # draw vertical line connecting edges at midpt for visualization
            cv2.line(frame_draw, (mid_x, edge_pixels[0]), (mid_x, edge_pixels[-1]), (200, 0, 0), 1)

        # record and save data using original resolution
        if frame_interval == 0:
            dist_data[f'1-Time({time_units})'].append(np.float16((frame_num - frame_start) / fps))
        else:
            dist_data[f'1-Time({time_units})'].append(np.float16((frame_num - frame_start) * frame_interval))
        dist_data['1-Frame'].append(frame_num - frame_start)
        dist_data['1-x at necking point (px)'].append(int(mid_x / scale_factor))
        dist_data['1-y necking distance (px)'].append(int(midpt_dist / scale_factor))

        cv2.imshow('Necking Point (midpoint method) Visualization', frame_draw)

        tracker_midpt_queue.task_done()
        if cv2.waitKey(1) == 27 or frame_end <= frame_num:
            stop_event.set()


def necking_point_midpoint_threaded(
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
    """
    Conducts multithreaded tracking to analyze necking points in a video sequence, using threads for capturing frames,
    updating marker positions, and detecting edges to determine midpoints. This is the alternate version to regular necking point, just finding min overall distance.

    Details:
        - Initiates threads for frame capture, marker updates, and midpoint edge detection.
        - Uses a queue system to manage frame processing and data flow between threads.
        - Tracks necking points by analyzing video frames to detect edges and calculate midpoints using configured trackers.
        - Collects and records data related to the detected necking points including frame number, time, and necking dimensions.
        - Efficiently handles video processing by delegating tasks to separate threads, improving performance for high-resolution or long-duration videos.

    Note:
        - Proper synchronization mechanisms using threading events and queues should be in place to manage data flow between threads and handle thread lifecycle.
        - Errors from threads are collected in a message queue and displayed after processing.

    Args:
        cap (cv2.VideoCapture): Video capture object loaded with the video.
        marker_positions (list of tuples): Initial positions of markers.
        first_frame (np.array): The first frame of the video for initializing the tracker.
        bbox_size (int): Size of the bounding box for the tracker.
        frame_start (int): Frame number to start processing.
        frame_end (int): Frame number to end processing.
        binarize_intensity_thresh (int): Threshold for binarization used in edge detection.
        frame_record_interval (int): Interval at which frames are processed.
        frame_interval (float): Real-world time interval between frames.
        time_units (str): Units for time (e.g., 'seconds', 'minutes') used in the output data.
        file_mode (FileMode enum): Mode for recording data ('FileMode.APPEND' or 'FileMode.OVERWRITE').
        video_file_name (str): Name of the video file being processed.
        data_label (str): Data label for the tracking session.

    Returns:
        None: This function does not return values but saves processed data to an output file and displays errors if any.
    """
    trackers = init_trackers(marker_positions, bbox_size, first_frame)

    frame_num = frame_start
    dist_data = {
        '1-Frame': [],
        f'1-Time({time_units})': [],
        '1-x at necking point (px)': [],
        '1-y necking distance (px)': [],
        '1-video_file_name': video_file_name,
        '1-detection_method': 'midpt',
        '1-data_label': data_label
    }

    fps = cap.get(5)
    frame_queue = queue.Queue(maxsize=10)
    tracker_midpt_queue = queue.Queue(maxsize=10)
    message_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=frame_capture_thread,
        args=(
            cap,
            frame_queue,
            message_queue,
            stop_event,
            frame_end,
            frame_start,
            frame_record_interval
        ), daemon=True
    )

    marker_update_thread = threading.Thread(
        target=frame_tracker_midpt_finder_thread, 
        args=(
            frame_queue,
            tracker_midpt_queue,
            message_queue,
            stop_event,
            trackers,
            frame_end
        ), daemon=True
    )

    edge_detection_thread = threading.Thread(
        target=frame_midpt_edge_detection,
        args=(
            frame_queue,
            tracker_midpt_queue,
            message_queue,
            dist_data,
            stop_event,
            frame_start,
            frame_end,
            frame_interval,
            time_units,
            fps,
            binarize_intensity_thresh
        ), daemon=True
    )

    # Start threads
    capture_thread.start()
    marker_update_thread.start()
    edge_detection_thread.start()

    # Wait for threads to finish
    capture_thread.join()
    print("capture done")
    marker_update_thread.join()
    print("marker done")
    edge_detection_thread.join()
    print("edge_detection_thread done")

    while not message_queue.empty():
        error_popup(message_queue.get())

    record_data(file_mode, dist_data, "output/Necking_Point_Output.csv")

    cap.release()
    cv2.destroyAllWindows()

    def frame_preprocessing_thread(
        tracker_frame_queue,
        processed_tracker_frame_queue,
        message_queue,
        stop_event,
        frame_start,
        frame_end,
        frame_interval
    ):
    """
    Thread function for preprocessing video frames to enhance features for further analysis like contour detection.
    This function retrieves frames from a queue, applies image preprocessing techniques, and places the processed frames into another queue.

    Details:
        - The function continuously retrieves frames from the tracker_frame_queue, which contain raw or minimally processed frames.
        - It applies grayscale conversion, adaptive binarization, and other image processing operations to enhance the frames for subsequent contour detection.
        - Processed frames are then pushed to the processed_tracker_frame_queue for the next stage in the processing pipeline.
        - This thread is designed to run concurrently with other threads handling frame capture, contour detection, and data recording.

    Note:
        - This function assumes that the frames are already scaled appropriately and only require further image processing.
        - Binarization and other image transformations are optimized for contour detection but may need adjustments depending on specific requirements.
        - Proper synchronization mechanisms (using threading events and queues) should be in place to manage the flow of data between threads and handle the lifecycle of the thread.

    Args:
        tracker_frame_queue (queue.Queue): Queue from which the thread retrieves frames for processing.
        processed_tracker_frame_queue (queue.Queue): Queue where the thread puts processed frames for further analysis.
        message_queue (queue.Queue): Queue for logging messages or errors.
        stop_event (threading.Event): An event to signal the thread to stop processing.
        frame_start (int): Frame number where processing starts.
        frame_end (int): Frame number where processing ends.
        frame_interval (float): Interval between frames, used in certain processing scenarios.

    Returns:
        None: This function does not return any value. It manages internal state and communicates via queues.
    """

    while True:
        try:
            queue_item = tracker_frame_queue.get(timeout=1)
        except queue.Empty:
            print("tracker_frame_queue EMPTY")
            continue

        if queue_item is None:
            processed_tracker_frame_queue.put(None)
            break

        if queue_item:
            frame_num, scaled_frame, scale_factor, tracker_bbox, _ = queue_item

        if frame_num >= frame_end:
            processed_tracker_frame_queue.put(None)
            break

        #print(f"cur frame num in frame_preprocessing_thread: {frame_num}")

        gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)
        
        #_, binary_frame = cv2.threshold(blur_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_frame = improve_binarization(gray_frame)

        # threshold frame again with localized kernel, adds a bit of noise but strengthens contours (idk why this helps but it does)
        adaptive_thresh = cv2.adaptiveThreshold(binary_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        processed_frame = adaptive_thresh

        processed_tracker_frame_queue.put((frame_num, scaled_frame, processed_frame, scale_factor, tracker_bbox))
        tracker_frame_queue.task_done()

        if cv2.waitKey(1) == 27:
            stop_event.set()

def find_contours_near_trackers_thread(
        processed_tracker_frame_queue,
        message_queue,
        stop_event,
        frame_start,
        frame_end,
        frame_interval,
        time_units,
        fps,
        distance_from_marker_thresh,
        area_data
    ):
    """
    Thread function to find contours near tracker positions in processed frames and record the relevant data.
    It retrieves frames from a queue, analyzes them for contours, and updates area data with centroid locations and surface areas.

    Details:
        - This thread continuously retrieves processed frames from a queue and searches for contours close to the tracked markers.
        - Each contour's proximity to the marker center is assessed, and the largest contour within the threshold distance is selected.
        - Information such as centroid location and area of the selected contour is recorded.
        - The thread visualizes the contours and tracking information on the frames for monitoring purposes.
        - Data is continuously updated into a shared dictionary intended for later analysis or saving.

    Note:
        - Ensure the frame data passed to this thread is pre-processed appropriately to highlight relevant contours, as done using the frame_preprocessing_thread().
        - This thread utilizes OpenCV functions for contour detection and moment calculation, which require binary images.
        - Proper error handling and synchronization mechanisms (e.g., threading events and queues) should be in place to handle the thread's lifecycle and communication.

    Args:
        processed_tracker_frame_queue (queue.Queue): Queue from which the thread retrieves processed frames.
        message_queue (queue.Queue): Queue for logging messages or errors.
        stop_event (threading.Event): An event to signal the thread to stop processing.
        frame_start (int): Frame number where processing starts.
        frame_end (int): Frame number where processing ends.
        frame_interval (float): Real-world time interval between frames.
        time_units (str): Units of time for the output data.
        fps (int): Frames per second of the video, used for time calculations.
        distance_from_marker_thresh (float): Threshold for determining closeness of contours to the tracker.
        area_data (dict): Dictionary to store results of the area calculations.

    Returns:
        None: This function does not return any value. It updates the area_data dictionary and can signal via message_queue.
    """
    while True:
        try:
            queue_item = processed_tracker_frame_queue.get(timeout=1)
            #print(queue_item)
        except queue.Empty:
            print("processed_tracker_frame_queue EMPTY")
            continue

        if queue_item is None:
            break

        if queue_item:
            frame_num, scaled_frame, processed_frame, scale_factor, tracker_bbox = queue_item
            x_bbox, y_bbox, w_bbox, h_bbox = tracker_bbox

        if frame_num >= frame_end:
            break

        #print(f"cur frame num in find_contours_near_trackers_thread: {frame_num}")

        # Segment frame
        contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        marker_center = (x_bbox + w_bbox // 2, y_bbox + h_bbox // 2)  # get center of bbox
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

        # draw bbox of tracker
        cv2.rectangle(scaled_frame, (x_bbox, y_bbox), (x_bbox + w_bbox, y_bbox + h_bbox), (0, 255, 0), 2)  # update tracker rectangle

        #cv2.imshow('Surface Area Tracking', noise_reduced_frame)
        cv2.imshow('Surface Area Tracking', scaled_frame)

        # record data
        area_data['1-Frame'].append(frame_num)
        if frame_interval == 0:
            area_data[f'1-Time({time_units})'].append(np.float32(frame_num / fps))
        else:
            area_data[f'1-Time({time_units})'].append(np.float16(frame_num * frame_interval))
        area_data['1-x centroid location'].append(int((centroid[0] / scale_factor)))
        area_data['1-y centroid location'].append(int((centroid[1] / scale_factor)))
        area_data['1-cell surface area (px^2)'].append(max_area)

        processed_tracker_frame_queue.task_done()

        if cv2.waitKey(1) == 27:
            stop_event.set()

def track_area_threaded(
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
    Conducts threaded video tracking to analyze the surface area of objects in a video sequence.
    This function utilizes multiple threads to manage different stages of the tracking process:
    frame capture, marker tracking, frame preprocessing, and contour detection.

    Details:
        - Initiates multiple threads: for capturing frames, updating marker positions, preprocessing frames,
          and detecting contours near trackers.
        - Data from tracking, such as centroid locations and calculated surface areas, are collected and
          stored in a structured manner.
        - Efficiently handles video processing by delegating tasks to separate threads, improving performance
          especially for high-resolution or long-duration videos.
        - Utilizes queues to manage data flow between threads and uses events to handle thread termination and errors.

    Note:
        - Proper synchronization mechanisms are used to prevent data races and ensure data consistency.
        - Errors from threads are collected in a message queue and displayed after processing.
        - This function is intended for use in environments where threading is supported and efficient.

    Args:
        cap (cv2.VideoCapture): Video capture object loaded with the video.
        marker_positions (list of tuples): Initial positions of markers.
        first_frame (np.array): The first frame of the video for initializing the tracker.
        bbox_size (int): Size of the bounding box for the tracker.
        frame_start (int): Frame number to start processing.
        frame_end (int): Frame number to end processing.
        frame_record_interval (int): Interval at which frames are processed.
        frame_interval (float): Real-world time interval between frames, used in time calculations.
        time_units (str): Units for time (e.g., 'seconds', 'minutes') used in the output data.
        distance_from_marker_thresh (float): Distance threshold for identifying relevant contours near markers.
        file_mode (FileMode enum): Mode for recording data ('FileMode.APPEND' or 'FileMode.OVERWRITE').
        video_file_name (str): Name of the video file being processed.
        data_label (str): Data label for the tracking session.

    Returns:
        None: This function does not return values but saves processed data to an output file and displays errors if any.
    """

    fps = cap.get(5)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    frame_num = frame_start
    area_data = {
        '1-Frame':[],
        f'1-Time({time_units})': [],
        '1-x centroid location': [],
        '1-y centroid location': [],
        '1-cell surface area (px^2)': [],
        '1-video_file_name': video_file_name,
        '1-data_label': data_label
    }
    
    trackers = init_trackers(marker_positions, bbox_size, first_frame, TrackerChoice.CSRT)

    frame_queue = queue.Queue(maxsize=10)
    tracker_frame_queue = queue.Queue(maxsize=10)
    processed_tracker_frame_queue = queue.Queue(maxsize=10)
    message_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=frame_capture_thread,
        args=(
            cap,
            frame_queue,
            message_queue,
            stop_event,
            frame_end,
            frame_start,
            frame_record_interval
        ), daemon=True
    )

    marker_update_thread = threading.Thread(
        target=frame_tracker_midpt_finder_thread, 
        args=(
            frame_queue,
            tracker_frame_queue,
            message_queue,
            stop_event,
            trackers,
            frame_end
        ), daemon=True
    )

    frame_preprocess_thread = threading.Thread(
        target=frame_preprocessing_thread,
        args=(
            tracker_frame_queue,
            processed_tracker_frame_queue,
            message_queue,
            stop_event,
            frame_start,
            frame_end,
            frame_interval
        )
    )

    contours_thread = threading.Thread(
        target=find_contours_near_trackers_thread,
        args=(
            processed_tracker_frame_queue,
            message_queue,
            stop_event,
            frame_start,
            frame_end,
            frame_interval,
            time_units,
            fps,
            distance_from_marker_thresh,
            area_data
        )
    )

    # Start threads
    capture_thread.start()
    marker_update_thread.start()
    frame_preprocess_thread.start()
    contours_thread.start()

    # Wait for threads to finish
    capture_thread.join()
    print("capture done")
    marker_update_thread.join()
    print("marker done")
    frame_preprocess_thread.join()
    print("frame binarization done")
    contours_thread.join()
    print("contour detection done")

    while not message_queue.empty():
        error_popup(message_queue.get())

    record_data(file_mode, area_data, "output/Surface_Area_Output.csv")

    cap.release()
    cv2.destroyAllWindows()

