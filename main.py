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
import time
import sys



VIDEO_PATH = ""

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


def window():
    root = tk.Tk()
    
    # file browse button
    data_label_var = tk.StringVar()
    file_btn = tk.Button(root, text="Browse for video file", command=lambda: get_file(data_label_var))
    file_btn.pack(padx=32,pady=24)

    # file name label
    data_label_var.set("File not selected")
    data_label = tk.Label(root, textvariable=data_label_var)
    data_label.pack(pady=(0,8))

    # radios for selecting operation
    operation_intvar = tk.IntVar()
    operation_intvar.set(0)
    operation_frame = tk.Frame(root)
    operation_tracking_radio = tk.Radiobutton(operation_frame, text="Marker tracking", variable=operation_intvar, value=1)
    operation_tracking_radio.grid(row=0, column=0, pady=16)
    operation_necking_radio = tk.Radiobutton(operation_frame, text="Necking point detection", variable=operation_intvar, value=2)
    operation_necking_radio.grid(row=0, column=1, pady=16)
    operation_frame.pack()

    # submit button
    submit_btn = tk.Button(root, text="Submit", command=lambda: main(operation_intvar.get()))
    submit_btn.pack(padx=32, pady=12)
    
    root.mainloop()

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
    saves distances of each mark each frame update to 'output/Tracking_Output.csv'

    Args:
        marker_positions (list): list of marker positions from user selections
        first_frame (np.Array(np.uint8)): image of first frame of loaded video
        cap (cv2.VideoCapture): loaded video
    """    
    # create trackers
    trackers = []
    for _ in range(len(marker_positions)):
        trackers.append(cv2.TrackerKCF_create())

    # init trackers
    for i, mark_pos in enumerate(marker_positions):
        bbox = (mark_pos[0][0], mark_pos[0][1], 20, 20) # 20x20 bounding box
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


def necking_point(cap, binarize_thresh=120, x_interval=50):
    if not cap.isOpened():
        print("Error: Couldn't open video file.")
        return
    
    frame_num = 0
    dist_data = {'Frame': [], 'Time(s)': [], 'x at necking point (px)': [], 'y necking distance (px)': []}

    while True: # read frame by frame until end of video
        ret, frame = cap.read()
        frame_num += 1
        #time.sleep(0.5)
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # frame already gray, but not read as such
        _, binary_frame = cv2.threshold(gray_frame, binarize_thresh, 1,cv2.THRESH_BINARY) # threshold to binarize img
        edges = cv2.Canny(binary_frame, 0, 2) # edge detection, nums are gradient thresholds

        x_samples = []
        y_distances = []
        y_line_values = []

        frame_draw = frame.copy() 
        frame_draw[edges > 0] = [0, 255, 0]  # draw edges

        for x in range(0, edges.shape[1]):
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
        print(y_distances[necking_pt_ind], y_distances)
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


def main(operation):
    global VIDEO_PATH
    cap = cv2.VideoCapture(VIDEO_PATH) # load video

    match operation:
        case 0:
            print("ERROR: Please select a radio option")
        case 1:
            print("Beginning Marker Tracking Process...")
            selected_markers, first_frame = select_markers(cap) # prompt to select markers
            track_markers(selected_markers, first_frame, cap)
        case 2:
            print("Beginning Necking Point")
            necking_point(cap)
        
    # get video metadata
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(5))
    n_frames = int(cap.get(7))
    print(width, height, fps, n_frames)


if __name__ == '__main__':
    window()
