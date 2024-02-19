'''
Author: Brandon Pardi
Created 1/26/2024

main interface code that calls and handles tracking and analysis routines
'''

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import os
import sys

from exceptions import error_popup, warning_popup
import analysis
import tracking


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
        file_btn = ttk.Button(self.root, text="Browse for video file", command=self.get_file, style='Regular.TButton')
        file_btn.grid(row=0, column=0, padx=32, pady=24)

        # file name label
        self.data_label_var = tk.StringVar()
        self.data_label_var.set("File not selected")
        data_label = ttk.Label(self.root, textvariable=self.data_label_var)
        data_label.grid(row=1, column=0, pady=(0,8))

        # frame selection button
        self.frame_start = -1
        self.frame_end = -1
        self.child = None
        self.frame_label_var = tk.StringVar()
        self.frame_label_var.set("Frame range: FULL")
        self.frame_label = ttk.Label(self.root, textvariable=self.frame_label_var)
        self.frame_selector_btn = ttk.Button(self.root, text="Select start/end frames", command=self.select_frames, style='Regular.TButton')
        self.frame_selector_btn.grid(row=2, pady=(12,4))
        self.frame_label.grid(row=3, column=0, pady=(0,8))

        # radios for selecting operation
        self.operation_intvar = tk.IntVar()
        self.operation_intvar.set(0)
        operation_frame = tk.Frame(self.root)
        operation_tracking_radio = ttk.Radiobutton(operation_frame, text="Marker tracking", variable=self.operation_intvar, value=1, command=self.handle_radios, width=25, style='Outline.TButton')
        operation_tracking_radio.grid(row=0, column=0, padx=4, pady=16)
        operation_necking_radio = ttk.Radiobutton(operation_frame, text="Necking point detection", variable=self.operation_intvar, value=2, command=self.handle_radios, width=25, style='Outline.TButton')
        operation_necking_radio.grid(row=0, column=1, padx=4, pady=16)
        operation_frame.grid(row=4, column=0)
        self.select_msg = ttk.Label(self.root, text="Select from above for more customizable parameters")
        self.select_msg.grid(row=5, column=0)

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
        track_btn.grid(row=0, column=0, columnspan=2, padx=32, pady=(24,4))
        remove_outliers_button = ttk.Button(submit_frame, text="Remove outliers", command=self.remove_outliers, style='Regular.TButton')
        remove_outliers_button.grid(row=1, column=0, columnspan=2, padx=32, pady=(4,24))
        
        marker_deltas_btn = ttk.Button(submit_frame, text="Marker deltas analysis", command=analysis.analyze_marker_deltas, style='Regular.TButton')
        marker_deltas_btn.grid(row=2, column=0, padx=4, pady=4)
        necking_pt_btn = ttk.Button(submit_frame, text="Necking point analysis", command=analysis.analyze_necking_point, style='Regular.TButton')
        necking_pt_btn.grid(row=3, column=0, padx=4, pady=4)
        poissons_ratio_btn = ttk.Button(submit_frame, text="Poisson's ratio", command=analysis.poissons_ratio, style='Regular.TButton')
        poissons_ratio_btn.grid(row=4, column=0, padx=4, pady=4)

        cell_velocity_btn = ttk.Button(submit_frame, text="Marker velocity", command=analysis.single_marker_velocity, style='Regular.TButton')
        cell_velocity_btn.grid(row=2, column=1, padx=4, pady=4)
        cell_distance_btn = ttk.Button(submit_frame, text="Marker distance", command=analysis.single_marker_distance, style='Regular.TButton')
        cell_distance_btn.grid(row=3, column=1, padx=4, pady=4)
        cell_spread_btn = ttk.Button(submit_frame, text="Marker spread", command=analysis.single_marker_velocity, style='Regular.TButton')
        cell_spread_btn.grid(row=4, column=1, padx=4, pady=4)

        exit_btn = ttk.Button(submit_frame, text='Exit', command=sys.exit, style='Regular.TButton')
        exit_btn.grid(row=10, column=0, columnspan=2, padx=32, pady=(24,12))
        submit_frame.grid(row=20, column=0)
        
    def select_frames(self):
        if self.video_path != "":
            self.child = FrameSelector(self.root, self.video_path, self.frame_label_var)
        else:
            msg = "Select a video before opening the frame selector"
            error_popup(msg)

    def remove_outliers(self):
        OutlierRemoval(self.root)

    def handle_radios(self):
        """blits options for the corresponding radio button selected"""        
        option = self.operation_intvar.get()
        match option:
            case 1:
                self.select_msg.grid_forget()
                self.necking_frame.grid_forget()
                self.tracking_frame.grid(row=6, column=0)
            case 2:
                self.select_msg.grid_forget()
                self.tracking_frame.grid_forget()
                self.necking_frame.grid(row=6, column=0)

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
            if self.child.end_selection_flag: # if an end frame sel made,
                self.frame_end = self.child.frame_end_select   

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

                selected_markers, first_frame = tracking.select_markers(cap, bbox_size, self.frame_start) # prompt to select markers
                print(f"marker locs: {selected_markers}")
                if not selected_markers.__contains__((-1,-1)): # select_markers returns list of -1 if selections cancelled
                    tracking.track_markers(selected_markers, first_frame, self.frame_start, self.frame_end, cap, bbox_size, tracker_choice)
            case 2:
                percent_crop_right = float(self.percent_crop_right_entry.get())
                percent_crop_left = float(self.percent_crop_left_entry.get())
                binarize_intensity_thresh = int(self.binarize_intensity_thresh_entry.get())

                print("Beginning Necking Point")
                tracking.necking_point(cap, self.frame_start, self.frame_end, percent_crop_left, percent_crop_right, binarize_intensity_thresh)

    def get_file(self):
        """util function to prompt a file browser to select the video file that will be tracked

        Args:
            data_label_var (tk.Label): tkinter label object to be updated with the chosen file name/path
        """    
        fp = filedialog.askopenfilename(initialdir=os.path.join(os.getcwd(), '../videos'),
                                        title='Browse for video file',
                                        filetypes=[("Audio Video Interleave", "*.avi"),
                                                ("MPEG-4 Part 14", "*.mp4"),
                                                ("Tag Image File Format", "*.tiff"),
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
    def __init__(self, parent, video_path, parent_label_var):
        self.parent = parent
        self.video_path = video_path
        self.parent_label_var = parent_label_var
        self.child_label_var = tk.StringVar()
        self.child_label_var.set("Select start and end frames of video\nuse slider at bottom or arrow keys (Shift + arrow to move 10 frames)\nCurrent frame: 0")
        self.cap = cv2.VideoCapture(self.video_path)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_start = 0
        self.frame_end = self.n_frames - 1

        self.start_selection_flag = False
        self.end_selection_flag = False

        self.child_window = tk.Toplevel(self.parent)
        self.child_window.title("Select Start and End Frames")
        self.child_window.geometry("+50+50")  # Adjust the values as needed

        # btn styling
        btn_style = ttk.Style()
        btn_style.configure("Regular.TButton", padding=(10,2), relief="raised", width=20)
        btn_style.map("Outline.TButton",
                foreground=[('selected', 'blue'), ('!selected', 'black')],
                background=[('selected', 'blue'), ('!selected', 'white')])
        
        self.frame_select_label = ttk.Label(self.child_window, textvariable=self.child_label_var)
        self.frame_select_label.pack(pady=10)
        
        self.confirm_start_button = ttk.Button(self.child_window, text="Confirm start frame", command=self.confirm_start, style='Regular.TButton')
        self.confirm_start_button.pack()
        self.confirm_end_button = ttk.Button(self.child_window, text="Confirm end frame", command=self.confirm_end, style='Regular.TButton')
        self.confirm_end_button.pack()
        
        self.frame_display = ttk.Label(self.child_window, text="test")
        self.frame_display.pack(pady=10)
        self.slider = ttk.Scale(self.child_window, from_=0, to=self.n_frames - 1, orient="horizontal", length=400,
                                command=self.update_frames)
        self.slider.set(0)
        self.slider.pack(pady=10)

        self.child_window.focus_set()
        self.child_window.bind("<Left>", lambda event: self.on_left_arrow(event))
        self.child_window.bind("<Right>", lambda event: self.on_right_arrow(event))


        self.child_window.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_left_arrow(self, event):
        direction = -10 if event.state & 0x1 else -1
        new_frame = max(0, self.frame_start + direction)
        self.slider.set(new_frame)

    def on_right_arrow(self, event):
        direction = 10 if event.state & 0x1 else 1
        new_frame = min(self.n_frames - 1, self.frame_start + direction)
        self.slider.set(new_frame)

    def update_frames(self, value):
        self.frame_start = int(float(value))
        self.frame_end = int(float(self.slider.get()))
        self.frame_display.config(text=f"Selected Frames: {self.frame_start} to {self.frame_end}")
        self.child_label_var.set(f"Select start and end frames of video\nUse slider at bottom or arrow keys to scroll through video\n(Shift + arrow to move 10 frames)\n\n\nCurrent frame: {int(float(self.slider.get()))}\n")
        
        # Update displayed frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_start)
        ret, frame = self.cap.read()
        if ret:
            frame, _ = tracking.scale_frame(frame)
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
        if not self.start_selection_flag:
            self.frame_start_select = 0
        if not self.end_selection_flag:
            self.frame_end_select = self.n_frames - 1
        self.parent_label_var.set(f"Frame start: {self.frame_start_select}, Frame end: {self.frame_end_select}")
        print("Selections confirmed!")


class OutlierRemoval:
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(self.parent)
        self.parent.title("Select outlier points to remove them")

        self.fig = None
        self.canvas = None

        # file options
        self.output_files = {
            'marker_tracking': 'output/Tracking_Output.csv',
            'necking_point': 'output/Necking_Point_Output.csv',
            'single_marker': 'output/Tracking_Output.csv'
        }

        self.selected_file = tk.StringVar()
        # radio buttons for file selection
        self.marker_radio = ttk.Radiobutton(self.window, text="Marker tracking output", variable=self.selected_file, value=self.output_files['marker_tracking'], command=self.load_plot)
        self.marker_radio.pack()
        self.necking_radio = ttk.Radiobutton(self.window, text="Necking point output", variable=self.selected_file, value=self.output_files['necking_point'], command=self.load_plot)
        self.necking_radio.pack()
        self.marker_vel_radio = ttk.Radiobutton(self.window, text="Single Marker output", variable=self.selected_file, value=self.output_files['single_marker'], command=self.load_plot)
        self.marker_vel_radio.pack()

        self.undo_removals_button = ttk.Button(self.window, text="Undo Selections", command=self.undo_selections)
        self.undo_removals_button.pack(pady=12)
        self.confirm_button = ttk.Button(self.window, text="Confirm Removal", comman=self.confirm)
        self.confirm_button.pack()

    def load_data(self, is_updating=False):
        if not is_updating:
            fp = self.selected_file.get()
            self.df = pd.read_csv(fp)

        # get appropriate columns
        self.x_col = 'Time(s)'
        self.x = list(self.df[self.x_col].unique())
        self.y = None
        if self.selected_file.get().__contains__('Tracking'):
            if self.df['Tracker'].unique().shape[0] > 1:
                _, self.y = analysis.analyze_marker_deltas(self.df, False)
            else:
                self.x, self.y = analysis.single_marker_velocity(self.df, False)
            
        if self.selected_file.get().__contains__('Necking'):
            _, self.y = analysis.analyze_necking_point(self.df, False)

    def create_figure(self):
        # Create a new figure and axes
        if not self.fig:
            self.fig, self.ax = plt.subplots()

    def plot_data(self, is_updating=False):
        self.load_data(is_updating)

        # Clear existing figure and axes
        if self.ax:
            self.ax.clear()
        else:
            self.create_figure()
        self.plot = self.ax.scatter(self.x, self.y, picker=True)

        # only draw if new canvas
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack()
        self.fig.canvas.mpl_connect('pick_event', self.onclick)

    def onclick(self, event):
        if event.name == 'pick_event':
            x_click = event.mouseevent.xdata
            idx = (np.abs(self.x - x_click)).argmin() # remove closest point
            print(idx)

            # find and remove clicked pt
            x = self.df.iloc[idx][self.x_col]
            idx_remove = self.df[self.df[self.x_col] == x].index
            self.df.drop(index=idx_remove, inplace=True)
            self.df.reset_index(inplace=True, drop=True)
            del self.y[idx_remove[0]]
            del self.x[idx_remove[0]]
            print(x, idx_remove, self.df)

            # update plot
            self.plot_data(True)
            self.plot.set_offsets(list(zip(self.x, self.y)))
            self.canvas.draw()

    def undo_selections(self):
        plt.clf()
        self.fig = None
        self.ax = None
        self.plot_data()

    def confirm(self):
        #fp, ext = os.path.splitext(self.selected_file.get())
        #self.output_fp = fp + '_outliers_removed' + ext
        self.df.to_csv(self.selected_file.get())

    def load_plot(self):
        self.create_figure()
        self.plot_data()





if __name__ == '__main__':
    root = tk.Tk()
    window = TrackingUI(root)
    root.mainloop()
