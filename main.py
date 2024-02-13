'''
Author: Brandon Pardi
Created 1/26/2024
'''

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os

from exceptions import error_popup, warning_popup
import analysis

VIDEO_PATH = ""

class TrackingUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Marker Tracker")
        
        # file browse button
        data_label_var = tk.StringVar()
        file_btn = tk.Button(self.root, text="Browse for video file", command=lambda: get_file(data_label_var))
        file_btn.grid(row=0, column=0, padx=32, pady=24)

        # file name label
        data_label_var.set("File not selected")
        data_label = tk.Label(self.root, textvariable=data_label_var)
        data_label.grid(row=1, column=0, pady=(0,8))

        # radios for selecting operation
        self.operation_intvar = tk.IntVar()
        self.operation_intvar.set(0)
        operation_frame = tk.Frame(self.root)
        operation_tracking_radio = tk.Radiobutton(operation_frame, text="Marker tracking", variable=self.operation_intvar, value=1, command=self.handle_radios, indicatoron=0)
        operation_tracking_radio.grid(row=0, column=0, padx=4, pady=16)
        operation_necking_radio = tk.Radiobutton(operation_frame, text="Necking point detection", variable=self.operation_intvar, value=2, command=self.handle_radios, indicatoron=0)
        operation_necking_radio.grid(row=0, column=1, padx=4, pady=16)
        operation_frame.grid(row=2, column=0)
        self.select_msg = tk.Label(self.root, text="Select from above for more customizable parameters")
        self.select_msg.grid(row=3, column=0)

        # options for marker tracking
        self.tracking_frame = tk.Frame(self.root)
        bbox_size_label = tk.Label(self.tracking_frame, text="Tracker bounding box size (px)")
        bbox_size_label.grid(row=0, column=0, padx=4, pady=8)
        self.bbox_size_entry = tk.Entry(self.tracking_frame, width=10)
        self.bbox_size_entry.insert(0, "20")
        self.bbox_size_entry.grid(row=0, column=1, padx=4, pady=8)

        self.tracker_choice_intvar = tk.IntVar()
        self.tracker_choice_intvar.set(0)
        tracker_choice_label = tk.Label(self.tracking_frame, text="Choose tracking algorithm")
        tracker_choice_label.grid(row=1, column=0, columnspan=2, padx=4, pady=(12,4))
        tracker_KCF_radio = tk.Radiobutton(self.tracking_frame, text="KCF tracker\n(best for consistent shape tracking)", variable=self.tracker_choice_intvar, value=0, indicatoron=0)
        tracker_KCF_radio.grid(row=2, column=0, padx=4)
        tracker_CSRT_radio = tk.Radiobutton(self.tracking_frame, text="CSRT tracker\n(best for deformable shape tracking)", variable=self.tracker_choice_intvar, value=1, indicatoron=0)
        tracker_CSRT_radio.grid(row=2, column=1, padx=4)

        # options for necking point
        self.necking_frame = tk.Frame(self.root)
        percent_crop_label = tk.Label(self.necking_frame, text="% of video width to\ncrop outter edges of\n(blank for none)")
        percent_crop_label.grid(row=0, column=0, padx=4, pady=8)        
        self.percent_crop_entry = tk.Entry(self.necking_frame, width=10)
        self.percent_crop_entry.insert(0, "0")
        self.percent_crop_entry.grid(row=0, column=1, padx=4, pady=8)

        binarize_intensity_thresh_label = tk.Label(self.necking_frame, text="pixel intensity value\nfor frame binarization\n(0-255)")
        binarize_intensity_thresh_label.grid(row=1, column=0, padx=4, pady=8)
        self.binarize_intensity_thresh_entry = tk.Entry(self.necking_frame, width=10)
        self.binarize_intensity_thresh_entry.insert(0, "120")
        self.binarize_intensity_thresh_entry.grid(row=1, column=1, padx=4, pady=8)

        # submit buttons
        submit_frame = tk.Frame()
        track_btn = tk.Button(submit_frame, text="Begin tracking", command=self.on_submit_tracking, width=20, height=2)
        track_btn.grid(row=0, column=0, columnspan=2, padx=32, pady=12)
        marker_deltas_btn = tk.Button(submit_frame, text="Marker deltas analysis", command=analysis.test, width=20, height=1)
        marker_deltas_btn.grid(row=1, column=0, padx=16, pady=4)
        analyze_data_btn = tk.Button(submit_frame, text="placeholder", command=analysis.test, width=20, height=1)
        analyze_data_btn.grid(row=1, column=1, padx=16, pady=4)
        submit_frame.grid(row=20, column=0)
        
    def handle_radios(self):
        option = self.operation_intvar.get()

        match option:
            case 1:
                self.select_msg.grid_forget()
                self.necking_frame.grid_forget()
                self.tracking_frame.grid(row=3, column=0)
            case 2:
                self.select_msg.grid_forget()
                self.tracking_frame.grid_forget()
                self.necking_frame.grid(row=3, column=0)

    def on_submit_tracking(self):
        global VIDEO_PATH
        cap = cv2.VideoCapture(VIDEO_PATH) # load video
        if not cap.isOpened():
            msg = "Error: Couldn't open video file.\nPlease ensure one was selected, and it is not corrupted."
            error_popup(msg)
        option = self.operation_intvar.get()

        match option:
            case 0:
                msg = "ERROR: Please select a radio button for a tracking operation."
                error_popup(msg)
            case 1:
                print("Beginning Marker Tracking Process...")
                bbox_size = int(self.bbox_size_entry.get())
                if self.tracker_choice_intvar.get() == 0:
                    tracker_choice = 'KCF'
                elif self.tracker_choice_intvar.get() == 1:
                    tracker_choice = 'CSRT'

                selected_markers, first_frame = select_markers(cap) # prompt to select markers
                track_markers(selected_markers, first_frame, cap, bbox_size, tracker_choice)
            case 2:
                percent_crop = float(self.percent_crop_entry.get())
                binarize_intensity_thresh = int(self.binarize_intensity_thresh_entry.get())

                print("Beginning Necking Point")
                necking_point(cap, percent_crop, binarize_intensity_thresh)


def get_file(label_var):
    fp = filedialog.askopenfilename(initialdir=os.path.join(os.getcwd(), 'videos'),
                                    title='Browse for video file',
                                    filetypes=[("Audio Video Interleave", "*.avi"),
                                               ("MPEG-4 Part 14", "*.mp4"),
                                               ("Matroska", "*.mkv"),
                                               ("QuickTime Movie", "*.mov"),
                                               ("Windows Media Video", "*.wmv"),
                                               ("Flash Video", "*.flv"),
                                               ("WebM", "*.webm"),
                                               ("MPEG Video", "*.mpeg"),
                                               ("MPEG-1/2 Video", "*.mpg")
                                               ])
    
    if fp:
        label_var.set(os.path.basename(fp))

    global VIDEO_PATH
    VIDEO_PATH = fp
    

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
        elif key == 13: # 13 ASCII for Enter key
            print(f"Selected positions: {mouse_params['marker_positions']}")
            break

    # close windows upon hitting select
    cv2.destroyAllWindows()
    
    return mouse_params['marker_positions'], first_frame


def track_markers(marker_positions, first_frame, cap, bbox_size, tracker_choice):
    """main tracking loop of markers selected
    saves distances of each mark each frame update to 'output/Tracking_Output.csv'

    Args:
        marker_positions (list): list of marker positions from user selections
        first_frame (np.Array(np.uint8)): image of first frame of loaded video
        cap (cv2.VideoCapture): loaded video
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
    for i, mark_pos in enumerate(marker_positions):
        bbox = (mark_pos[0][0] - bbox_size//2, mark_pos[0][1] - bbox_size//2, bbox_size, bbox_size)
        trackers[i].init(first_frame, bbox)

    # init tracking data dict
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
    dist_df.to_csv("output/Tracking_Output.csv")

    cap.release()
    cv2.destroyAllWindows()


def necking_point(cap, percent_crop=0., binarize_intensity_thresh=120, x_interval=50):
    frame_num = 0
    dist_data = {'Frame': [], 'Time(s)': [], 'x at necking point (px)': [], 'y necking distance (px)': []}
    percent_crop *= 0.01

    while True: # read frame by frame until end of video
        ret, frame = cap.read()
        frame_num += 1
        #time.sleep(0.5)
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # frame already gray, but not read as such
        _, binary_frame = cv2.threshold(gray_frame, binarize_intensity_thresh, 1,cv2.THRESH_BINARY) # threshold to binarize img
        
        # error checking for appropriate binarization threshold
        if np.all(binary_frame == 1):
            msg = "Binarization threshold too low,\nfound no pixels below the threshold.\n\nPlease adjust the threshold (default is 120)"
            error_popup(msg)
        if np.all(binary_frame == 0):
            msg = "Binarization threshold too high,\nfound no pixels above the threshold.\n\nPlease adjust the threshold (default is 120)"
            error_popup(msg)   

        edges = cv2.Canny(binary_frame, 0, 2) # edge detection, nums are gradient thresholds

        x_samples = []
        y_distances = []
        y_line_values = []

        frame_draw = frame.copy() 
        frame_draw[edges > 0] = [0, 255, 0]  # draw edges

        # remove x% of edges from consideration of detection
        horizontal_pixels_range = (0, frame.shape[1])
        if percent_crop != 0.:
            horizonatal_pixels_removed = int(percent_crop*frame.shape[1])
            print(horizonatal_pixels_removed)
            horizontal_pixels_range = (horizonatal_pixels_removed//2, frame.shape[1] - horizonatal_pixels_removed//2)
            print(horizontal_pixels_range)

        for x in range(horizontal_pixels_range[0], horizontal_pixels_range[1]):
            edge_pixels = np.nonzero(edges[:,x])[0] # find y coord of edge pixels in cur column

            if edge_pixels.size > 0: # if edge pixels in cur column, 
                dist = np.abs(edge_pixels[0] - edge_pixels[-1]) # find distance of top and bottom edges
                x_samples.append(x)
                y_line_values.append((edge_pixels[0], edge_pixels[-1]))
                y_distances.append(dist)

                if x % x_interval == 0: # draw visualization lines at every x_interval pixels
                    # draw vertical lines connecting edges for visualization
                    cv2.line(frame_draw, (x, edge_pixels[0]), (x, edge_pixels[-1]), (200, 0, 0), 1)  

        # find index of smallest distance
        # multiple mins occur in typically close together, for now just pick middle of min occurences
        necking_distance = np.min(y_distances)
        necking_pt_indices = np.where(y_distances==necking_distance)[0]
        necking_pt_ind = int(np.median(necking_pt_indices))
        #print(y_distances[necking_pt_ind], y_distances)
        cv2.line(frame_draw, (x_samples[necking_pt_ind], y_line_values[necking_pt_ind][0]), (x_samples[necking_pt_ind], y_line_values[necking_pt_ind][1]), (0,0,255), 2)     

        # record and save data
        dist_data['Frame'].append(frame_num)
        dist_data['Time(s)'].append(np.float32(frame_num / cap.get(5)))
        dist_data['x at necking point (px)'].append(x_samples[necking_pt_ind])
        dist_data['y necking distance (px)'].append(necking_distance)

        cv2.imshow('Necking Point Visualization', frame_draw)
        if cv2.waitKey(1) == 27:
            break

    dist_df = pd.DataFrame(dist_data)
    dist_df.set_index('Frame', inplace=True)
    dist_df.to_csv("output/Necking_Point_Output.csv")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    root = tk.Tk()
    window = TrackingUI(root)
    root.mainloop()
