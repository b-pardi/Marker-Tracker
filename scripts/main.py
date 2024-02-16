'''
Author: Brandon Pardi
Created 1/26/2024
'''

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import screeninfo
from PIL import Image, ImageTk

import os
import sys

from exceptions import error_popup, warning_popup
import analysis


class TrackingUI:
    """class for handling Tkinter window and its functionalities"""    
    def __init__(self, root):
        self.root = root
        self.root.title("Marker Tracker - M3B Lab")

        # ttk widget stylings
        radio_btn_style = ttk.Style()
        radio_btn_style.configure("Outline.TButton", borderwidth=2, relief="solid", padding=(2, 5), foreground="black")
        btn_style = ttk.Style()
        btn_style.configure("Regular.TButton", padding=(10,5), relief="raised", width=20)

        # "Outline.TButton" style map
        radio_btn_style.map("Outline.TButton",
                foreground=[('selected', 'blue'), ('!selected', 'black')],
                background=[('selected', 'blue'), ('!selected', 'white')])
        radio_btn_style.configure("Outline.TButton", width=20)
        
        # file browse button
        self.video_path = ""
        self.data_label_var = tk.StringVar()
        file_btn = ttk.Button(self.root, text="Browse for video file", command=self.get_file, style='Regular.TButton')
        file_btn.grid(row=0, column=0, padx=32, pady=24)

        # file name label
        self.data_label_var.set("File not selected")
        data_label = ttk.Label(self.root, textvariable=self.data_label_var)
        data_label.grid(row=1, column=0, pady=(0,8))

        # frame selection button
        self.frame_start = -1
        self.frame_end = -1
        self.child = None
        self.frame_selector_btn = ttk.Button(self.root, text="Select start/end frames", command=self.select_frames, style='Regular.TButton')
        self.frame_selector_btn.grid(row=2, pady=16)

        # radios for selecting operation
        self.operation_intvar = tk.IntVar()
        self.operation_intvar.set(0)
        operation_frame = tk.Frame(self.root)
        operation_tracking_radio = ttk.Radiobutton(operation_frame, text="Marker tracking", variable=self.operation_intvar, value=1, command=self.handle_radios, width=25, style='Outline.TButton')
        operation_tracking_radio.grid(row=0, column=0, padx=4, pady=16)
        operation_necking_radio = ttk.Radiobutton(operation_frame, text="Necking point detection", variable=self.operation_intvar, value=2, command=self.handle_radios, width=25, style='Outline.TButton')
        operation_necking_radio.grid(row=0, column=1, padx=4, pady=16)
        operation_frame.grid(row=3, column=0)
        self.select_msg = ttk.Label(self.root, text="Select from above for more customizable parameters")
        self.select_msg.grid(row=4, column=0)

        # options for marker tracking
        self.tracking_frame = tk.Frame(self.root)
        bbox_size_label = ttk.Label(self.tracking_frame, text="Tracker bounding box size (px)")
        bbox_size_label.grid(row=0, column=0, padx=4, pady=8)
        self.bbox_size_entry = ttk.Entry(self.tracking_frame, width=10)
        self.bbox_size_entry.insert(0, "20")
        self.bbox_size_entry.grid(row=0, column=1, padx=4, pady=8)

        self.tracker_choice_intvar = tk.IntVar()
        self.tracker_choice_intvar.set(0)
        tracker_choice_label = ttk.Label(self.tracking_frame, text="Choose tracking algorithm")
        tracker_choice_label.grid(row=1, column=0, columnspan=2, padx=4, pady=(12,4))
        tracker_KCF_radio = ttk.Radiobutton(self.tracking_frame, text="KCF tracker\n(best for consistent shape tracking)", variable=self.tracker_choice_intvar, value=0, width=36, style='Outline.TButton')
        tracker_KCF_radio.grid(row=2, column=0, padx=4)
        tracker_CSRT_radio = ttk.Radiobutton(self.tracking_frame, text="CSRT tracker\n(best for deformable shape tracking)", variable=self.tracker_choice_intvar, value=1, width=36, style='Outline.TButton')
        tracker_CSRT_radio.grid(row=2, column=1, padx=4)

        # options for necking point
        self.necking_frame = tk.Frame(self.root)
        percent_crop_label = ttk.Label(self.necking_frame, text="% of video width to\nexclude outter edges of\n(0 for none)")
        percent_crop_label.grid(row=0, column=0, rowspan=2, padx=4, pady=8)
        percent_crop_left_label = ttk.Label(self.necking_frame, text="left edge") 
        percent_crop_left_label.grid(row=0, column=1)    
        self.percent_crop_left_entry = ttk.Entry(self.necking_frame, width=10)
        self.percent_crop_left_entry.insert(0, "0")
        self.percent_crop_left_entry.grid(row=1, column=1, padx=4, pady=8)

        percent_crop_right_label = ttk.Label(self.necking_frame, text="right edge")     
        percent_crop_right_label.grid(row=0, column=2)    
        self.percent_crop_right_entry = ttk.Entry(self.necking_frame, width=10)
        self.percent_crop_right_entry.insert(0, "0")
        self.percent_crop_right_entry.grid(row=1, column=2, padx=4, pady=8)

        binarize_intensity_thresh_label = ttk.Label(self.necking_frame, text="pixel intensity value\nfor frame binarization\n(0-255)")
        binarize_intensity_thresh_label.grid(row=2, column=0, padx=4, pady=8)
        self.binarize_intensity_thresh_entry = ttk.Entry(self.necking_frame, width=10)
        self.binarize_intensity_thresh_entry.insert(0, "120")
        self.binarize_intensity_thresh_entry.grid(row=2, column=1, columnspan=2, padx=4, pady=8)

        # submit buttons
        submit_frame = tk.Frame()
        track_btn = ttk.Button(submit_frame, text="Begin tracking", command=self.on_submit_tracking, style='Regular.TButton')
        track_btn.grid(row=0, column=0, columnspan=2, padx=32, pady=24)
        marker_deltas_btn = ttk.Button(submit_frame, text="Marker deltas analysis", command=analysis.analyze_marker_deltas, style='Regular.TButton')
        marker_deltas_btn.grid(row=1, column=0, padx=4, pady=4)
        analyze_data_btn = ttk.Button(submit_frame, text="Necking point analysis", command=analysis.analyze_necking_point, style='Regular.TButton')
        analyze_data_btn.grid(row=1, column=1, padx=4, pady=4)
        exit_btn = ttk.Button(submit_frame, text='Exit', command=sys.exit, style='Regular.TButton')
        exit_btn.grid(row=10, column=0, columnspan=2, padx=32, pady=(24,12))
        submit_frame.grid(row=20, column=0)
        
    def select_frames(self):
        self.child = FrameSelector(self.root, self.video_path)

    def handle_radios(self):
        """blits options for the corresponding radio button selected"""        
        option = self.operation_intvar.get()
        match option:
            case 1:
                self.select_msg.grid_forget()
                self.necking_frame.grid_forget()
                self.tracking_frame.grid(row=5, column=0)
            case 2:
                self.select_msg.grid_forget()
                self.tracking_frame.grid_forget()
                self.necking_frame.grid(row=5, column=0)

    def on_submit_tracking(self):
        """calls the appropriate functions with user spec'd args when tracking start button clicked"""        
        cap = cv2.VideoCapture(self.video_path) # load video
        if not cap.isOpened():
            msg = "Error: Couldn't open video file.\nPlease ensure one was selected, and it is not corrupted."
            error_popup(msg)
        option = self.operation_intvar.get()

        # set frame start/end
        self.frame_start = 0
        self.frame_end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        if self.child: # if frame select window exists (if it was opened via the btn),
            if self.child.start_selection_flag: # if a start frame sel made,
                self.frame_start = self.child.frame_start_select
            else: 
                self.frame_start = 0
            if self.child.end_selection_flag: # if an end frame sel made,
                self.frame_end = self.child.frame_end_select
            else:
                self.frame_end = self.child.n_frames - 1            

        print(f"frame start: {self.frame_start}, frame end: {self.frame_end}")

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

                selected_markers, first_frame = select_markers(cap, bbox_size, self.frame_start) # prompt to select markers
                print(f"marker locs: {selected_markers}")
                if not selected_markers.__contains__((-1,-1)): # select_markers returns list of -1 if selections cancelled
                    track_markers(selected_markers, first_frame, self.frame_start, self.frame_end, cap, bbox_size, tracker_choice)
            case 2:
                percent_crop_right = float(self.percent_crop_right_entry.get())
                percent_crop_left = float(self.percent_crop_left_entry.get())
                binarize_intensity_thresh = int(self.binarize_intensity_thresh_entry.get())

                print("Beginning Necking Point")
                necking_point(cap, percent_crop_left, percent_crop_right, binarize_intensity_thresh)

    def get_file(self):
        """util function to prompt a file browser to select the video file that will be tracked

        Args:
            data_label_var (tk.Label): tkinter label object to be updated with the chosen file name/path
        """    
        fp = filedialog.askopenfilename(initialdir=os.path.join(os.getcwd(), '../videos'),
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
            self.data_label_var.set(os.path.basename(fp))
            self.video_path = fp

class FrameSelector:
    def __init__(self, parent, video_path):
        self.parent = parent
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_start = 0
        self.frame_end = self.n_frames - 1

        self.start_selection_flag = False
        self.end_selection_flag = False

        self.child_window = tk.Toplevel(self.parent)
        self.child_window.title("Select Start and End Frames")
        self.child_window.geometry("+50+50")  # Adjust the values as needed

        
        self.frame_select_label = ttk.Label(self.child_window, text="Use slider to select start and end frames")
        self.frame_select_label.pack(pady=10)
        
        self.confirm_start_button = ttk.Button(self.child_window, text="Confirm start frame", command=self.confirm_start)
        self.confirm_start_button.pack(pady=10)
        self.confirm_end_button = ttk.Button(self.child_window, text="Confirm end frame", command=self.confirm_end)
        self.confirm_end_button.pack(pady=10)
        
        self.frame_display = ttk.Label(self.child_window, text="test")
        self.frame_display.pack(pady=10)
        self.slider = ttk.Scale(self.child_window, from_=0, to=self.n_frames - 1, orient="horizontal", length=400,
                                command=self.update_frames)
        self.slider.set(0)
        self.slider.pack(pady=10)

        self.child_window.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_frames(self, value):
        self.frame_start = int(float(value))
        self.frame_end = int(float(self.slider.get()))
        self.frame_display.config(text=f"Selected Frames: {self.frame_start} to {self.frame_end}")
        
        # Update displayed frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_start)
        ret, frame = self.cap.read()
        if ret:
            frame, _ = scale_frame(frame)
            self.display_frame(frame)

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame)) # pillow to interact with imgs in tkinter
        self.frame_display.imgtk = imgtk
        self.frame_display.configure(image=imgtk)

    def confirm_start(self):
        self.start_selection_flag = True
        self.frame_start_select = int(float(self.slider.get()))
        print(f"Selected start: {self.frame_start_select}")

    def confirm_end(self):
        self.end_selection_flag = True
        self.frame_end_select = int(float(self.slider.get()))
        print(f"Selected end: {self.frame_end_select}")

    def on_close(self):
        self.cap.release()
        self.child_window.destroy()
        print(f"Selected start: {self.parent.frame_start}")


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
    cv2.moveWindow('Your Window', 50, 50)


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
    cv2.moveWindow('Your Window', 50, 50)

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


def track_markers(marker_positions, first_frame, frame_start, frame_end, cap, bbox_size, tracker_choice):
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

        if cv2.waitKey(1) == 27 or frame_num == frame_end-frame_start: # cut tracking loop short if ESC hit
            break

    
    dist_df = pd.DataFrame(tracker_data)
    dist_df.set_index('Frame', inplace=True)
    dist_df.to_csv("../output/Tracking_Output.csv")

    cap.release()
    cv2.destroyAllWindows()


def necking_point(cap, percent_crop_left=0., percent_crop_right=0., binarize_intensity_thresh=120, x_interval=50):
    """necking point detection loop
    necking point defined as the most shortest vertical line between two horizontal edges
    frames are preprocessed and then edges are detected, top and bottom most edges are singled out
    each frame the vertical distance between these edges are recorded at every x coordinate and the shortest is highlighted in red
    in cases where there are multiple min distance values, the median x location is chosen
    recorded data saved to "output/Necking_Point_Output.csv"

    Args:
        cap (cv2.VideoCapture): loaded video file for selecting markers of
        percent_crop_left (float, optional): percentage of pixels to remove from consideration from left side of frame of necking pt detection. Defaults to 0..
        percent_crop_right (_type_, optional): percentage of pixels to remove from consideration from right side of frame of necking pt detection. Defaults to 0..
        binarize_intensity_thresh (int, optional): threshold pixel intensity value for frame binarization. Defaults to 120.
        x_interval (int, optional): interval of horizontal pixels to draw vertical blue lines for visualization purposes. Defaults to 50.
    """    
    frame_num = 0
    dist_data = {'Frame': [], 'Time(s)': [], 'x at necking point (px)': [], 'y necking distance (px)': []}
    percent_crop_left *= 0.01
    percent_crop_right *= 0.01
    print(percent_crop_left)

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
        horizontal_pixels_left = 0
        horizontal_pixels_right = frame.shape[1]
        if percent_crop_left != 0.:
            left_pixels_removed = int(percent_crop_left*frame.shape[1])
            horizontal_pixels_left = max(0, left_pixels_removed)
        if percent_crop_right != 0.:
            right_pixels_removed = int(percent_crop_right*frame.shape[1])
            horizontal_pixels_right = min(frame.shape[1], frame.shape[1] - right_pixels_removed)

        for x in range(horizontal_pixels_left, horizontal_pixels_right):
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
    dist_df.to_csv("../output/Necking_Point_Output.csv")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    root = tk.Tk()
    window = TrackingUI(root)
    root.mainloop()
