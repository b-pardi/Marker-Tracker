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
import json

import os
import sys

from exceptions import error_popup, warning_popup, warning_prompt
from enums import *
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
        
        # scrollbar
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.scrollbar_frame = tk.Frame(self.root)
        self.scrollbar_frame.grid(row=0, column=1, sticky="ns")
        self.scrollbar_frame.grid_rowconfigure(0, weight=1)
        self.scrollbar_frame.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self.scrollbar_frame)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar = ttk.Scrollbar(self.scrollbar_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        #self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.scrollable_frame.bind("<Configure>", self.on_frame_configure)
        
        # section 1 houses tracking/recording related widgets
        self.section1 = tk.Frame(self.scrollable_frame)
        sect1_header = ttk.Label(self.section1, text="Tracking & Recording", font=('TkDefaultFont', 14, 'bold'))
        sect1_header.grid(row=0, column=0)

        # file browse button
        self.video_path = ""
        file_btn = ttk.Button(self.section1, text="Browse for video file", command=self.get_file, style='Regular.TButton')
        file_btn.grid(row=1, column=0, padx=32, pady=24)

        # file name label
        self.data_label_var = tk.StringVar()
        self.data_label_var.set("File not selected")
        data_label = ttk.Label(self.section1, textvariable=self.data_label_var)
        data_label.grid(row=2, column=0, pady=(0,8))

        # frame selection button
        self.frame_start = -1
        self.frame_end = -1
        self.child = None
        self.frame_label_var = tk.StringVar()
        self.frame_label_var.set("Frame range: FULL")
        self.frame_label = ttk.Label(self.section1, textvariable=self.frame_label_var)
        self.frame_selector_btn = ttk.Button(self.section1, text="Select start/end frames", command=self.select_frames, style='Regular.TButton')
        self.frame_selector_btn.grid(row=3, pady=(12,4))
        self.frame_label.grid(row=4, column=0, pady=(0,8))

        # frame interval input
        self.frame_interval = 0
        self.time_units = TimeUnits.SECONDS
        self.is_timelapse_var = tk.IntVar()
        is_timelapse_check = ttk.Checkbutton(self.section1, text="Is this video a timelapse? ", variable=self.is_timelapse_var, onvalue=1, offvalue=0, command=self.handle_checkbuttons)
        is_timelapse_check.grid(row=5, column=0)
        self.frame_interval_frame = tk.Frame(self.section1)
        time_units_label = ttk.Label(self.frame_interval_frame, text="Units of time:")
        time_units_label.grid(row=0, column=0, columnspan=3, pady=6)

        time_units_frame = tk.Frame(self.frame_interval_frame)
        self.time_units_var = tk.StringVar()
        self.time_units_var.set(TimeUnits.UNSELECTED.value)
        time_units_seconds_radio = ttk.Radiobutton(time_units_frame, text='s', variable=self.time_units_var, value=TimeUnits.SECONDS.value)
        time_units_seconds_radio.grid(row=0, column=0)
        time_units_minutes_radio = ttk.Radiobutton(time_units_frame, text='min', variable=self.time_units_var, value=TimeUnits.MINUTES.value)
        time_units_minutes_radio.grid(row=0, column=1)
        time_units_hours_radio = ttk.Radiobutton(time_units_frame, text='hr', variable=self.time_units_var, value=TimeUnits.HOURS.value)
        time_units_hours_radio.grid(row=0, column=2)

        frame_interval_label = ttk.Label(self.frame_interval_frame, text="Frame interval: \n(image is taken every\n'x' unit of time)")
        frame_interval_label.grid(row=2, column=0, padx=16, pady=16)
        self.frame_interval_entry = ttk.Entry(self.frame_interval_frame, width=10)
        self.frame_interval_entry.grid(row=2, column=1)
        time_units_frame.grid(row=1, column=0, columnspan=2)

        # indicate if appending data
        self.append_or_overwrite_frame = tk.Frame(self.section1)
        self.append_or_overwrite_var = tk.IntVar()
        self.append_or_overwrite_var.set(FileMode.UNSELECTED.value)
        append_or_overwrite_label = ttk.Label(self.append_or_overwrite_frame, text="Append or overwrite existing tracking data?\n")
        append_or_overwrite_label.grid(row=0, column=0, columnspan=2)
        self.append_radio = ttk.Radiobutton(self.append_or_overwrite_frame, text="Append", variable=self.append_or_overwrite_var, value=FileMode.APPEND.value)
        self.append_radio.grid(row=1, column=0)
        self.overwrite_radio = ttk.Radiobutton(self.append_or_overwrite_frame, text="Overwrite", variable=self.append_or_overwrite_var, value=FileMode.OVERWRITE.value)
        self.overwrite_radio.grid(row=1, column=1)
        self.append_or_overwrite_frame.grid(row=7, column=0, pady=12)

        # skip frames when tracking option
        skip_frames_frame = tk.Frame(self.section1)
        skip_frames_label_prefix = ttk.Label(skip_frames_frame, text="Track every ")
        skip_frames_label_prefix.grid(row=0, column=0)
        self.skip_frames_entry = ttk.Entry(skip_frames_frame, width=4)
        self.skip_frames_entry.insert(0, '1')
        self.skip_frames_entry.grid(row=0, column=1)
        skip_frames_label_postfix = ttk.Label(skip_frames_frame, text="frame(s)")
        skip_frames_label_postfix.grid(row=0, column=2)
        skip_frames_frame.grid(row=8, column=0, pady=12)

        # optional range identifier
        range_identifier_label = ttk.Label(self.section1, text="Enter an optional identifier label for the tracking data: ")
        range_identifier_label.grid(row=9, column=0, pady=(8,4))
        self.range_identifier_entry = ttk.Entry(self.section1)
        self.range_identifier_entry.grid(row=10, column=0, pady=(0,8))

        # radios for selecting operation
        self.operation_intvar = tk.IntVar()
        self.operation_intvar.set(TrackingOperation.UNSELECTED.value)
        operation_frame = tk.Frame(self.section1)
        operation_tracking_radio = ttk.Radiobutton(operation_frame, text="Marker tracking", variable=self.operation_intvar, value=TrackingOperation.MARKERS.value, command=self.handle_radios, width=25, style='Outline.TButton')
        operation_tracking_radio.grid(row=0, column=0, padx=4, pady=(16, 4))
        operation_necking_radio = ttk.Radiobutton(operation_frame, text="Necking point detection", variable=self.operation_intvar, value=TrackingOperation.NECKING.value, command=self.handle_radios, width=25, style='Outline.TButton')
        operation_necking_radio.grid(row=0, column=1, padx=4, pady=(16, 4))
        operation_area_radio = ttk.Radiobutton(operation_frame, text="Surface area tracking", variable=self.operation_intvar, value=TrackingOperation.AREA.value, command=self.handle_radios, width=25, style='Outline.TButton')
        operation_area_radio.grid(row=1, column=0, columnspan=2, padx=4, pady=(4, 16))
        operation_frame.grid(row=11, column=0)
        self.select_msg = ttk.Label(self.section1, text="Select from above for more customizable parameters")
        self.select_msg.grid(row=12, column=0)

        # options for marker tracking
        self.tracking_frame = tk.Frame(self.section1)
        bbox_tracking_size_label = ttk.Label(self.tracking_frame, text="Tracker bounding box size (px)")
        bbox_tracking_size_label.grid(row=0, column=0, padx=4, pady=8)
        self.bbox_size_tracking_entry = ttk.Entry(self.tracking_frame, width=10)
        self.bbox_size_tracking_entry.insert(0, "100")
        self.bbox_size_tracking_entry.grid(row=0, column=1, padx=4, pady=8)

        self.tracker_choice_intvar = tk.IntVar()
        self.tracker_choice_intvar.set(TrackerChoice.UNSELECTED.value)
        tracker_choice_label = ttk.Label(self.tracking_frame, text="Choose tracking algorithm")
        tracker_choice_label.grid(row=1, column=0, columnspan=2, padx=4, pady=(12,4))
        tracker_KCF_radio = ttk.Radiobutton(self.tracking_frame, text="KCF tracker\n(best for consistent shape tracking)", variable=self.tracker_choice_intvar, value=TrackerChoice.KCF.value, width=36, style='Outline.TButton')
        tracker_KCF_radio.grid(row=2, column=0, padx=4)
        tracker_CSRT_radio = ttk.Radiobutton(self.tracking_frame, text="CSRT tracker\n(best for deformable shape tracking)", variable=self.tracker_choice_intvar, value=TrackerChoice.CSRT.value, width=36, style='Outline.TButton')
        tracker_CSRT_radio.grid(row=2, column=1, padx=4)

        # options for necking point
        self.necking_frame = tk.Frame(self.section1)
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

        # options for surface area tracking
        self.area_frame = tk.Frame(self.section1)
        bbox_area_size_label = ttk.Label(self.area_frame, text="Tracker bounding box size (px)")
        bbox_area_size_label.grid(row=0, column=0, padx=4, pady=8)
        self.bbox_size_area_entry = ttk.Entry(self.area_frame, width=10)
        self.bbox_size_area_entry.insert(0, "100")
        self.bbox_size_area_entry.grid(row=0, column=1, padx=4, pady=8)
        distance_from_marker_thresh_label = ttk.Label(self.area_frame, text="Max distance from marker to find contours (px)")
        distance_from_marker_thresh_label.grid(row=1, column=0, padx=4, pady=8)
        self.distance_from_marker_thresh_entry = ttk.Entry(self.area_frame, width=10)
        self.distance_from_marker_thresh_entry.insert(0, "150")
        self.distance_from_marker_thresh_entry.grid(row=1, column=1, padx=4, pady=8)

        # tracking operation buttons
        track_record_frame = tk.Frame(self.section1)
        track_btn = ttk.Button(track_record_frame, text="Begin tracking", command=self.on_submit_tracking, style='Regular.TButton')
        track_btn.grid(row=0, column=0, columnspan=2, padx=32, pady=(24,4))
        remove_outliers_button = ttk.Button(track_record_frame, text="Remove outliers", command=self.remove_outliers, style='Regular.TButton')
        remove_outliers_button.grid(row=1, column=0, columnspan=2, padx=32, pady=(4,24))
        undo_buttons_frame = tk.Frame(track_record_frame)
        undo_label = ttk.Label(undo_buttons_frame, text="Undo previous appended tracking data recording.")
        undo_label.grid(row=0, column=0, columnspan=3)
        undo_marker_tracking_append_btn = ttk.Button(undo_buttons_frame, text="Marker tracking", command=lambda: self.undo_last_tracking_append("output/Tracking_Output.csv"), style='Regular.TButton')
        undo_marker_tracking_append_btn.grid(row=1, column=0)        
        undo_necking_point_append_btn = ttk.Button(undo_buttons_frame, text="Necking point", command=lambda: self.undo_last_tracking_append("output/Necking_Point_Output.csv"), style='Regular.TButton')
        undo_necking_point_append_btn.grid(row=1, column=1)    
        undo_surface_tracking_append_btn = ttk.Button(undo_buttons_frame, text="Surface area", command=lambda: self.undo_last_tracking_append("output/Surface_Area_Output.csv"), style='Regular.TButton')
        undo_surface_tracking_append_btn.grid(row=1, column=2)
        undo_buttons_frame.grid(row=2, column=0, columnspan=2, pady=(8,20))
        track_record_frame.grid(row=18, column=0)
        self.section1.grid(row=0, column=0)

        # section 2 pertains to data analysis
        section2 = tk.Frame(self.scrollable_frame)
        sect2_header = ttk.Label(section2, text="Data Analysis", font=('TkDefaultFont', 14, 'bold'))
        sect2_header.grid(row=0, column=0)

        # some data dependent graph options (more in plot_opts.json)
        graph_opts_frame = tk.Frame(section2)
        graph_opts_label = ttk.Label(graph_opts_frame, text="Data dependent graph options")
        graph_opts_label.grid(row=0, column=0, columnspan=2, pady=(0,12))
        conversion_factor_label = ttk.Label(graph_opts_frame, text="Enter the conversion factor\nto convert pixels to\nyour desired units: ")
        conversion_factor_label.grid(row=1, column=0)
        self.conversion_factor_entry = ttk.Entry(graph_opts_frame)
        self.conversion_factor_entry.insert(0, "1.0")
        self.conversion_factor_entry.grid(row=1, column=1, padx=32)
        conversion_units_label = ttk.Label(graph_opts_frame, text="Enter the units that result\nfrom this conversion\n(mm, µm, nm, etc.): ")
        conversion_units_label.grid(row=2, column=0, pady=(0,16))
        self.conversion_units_entry = ttk.Entry(graph_opts_frame)
        self.conversion_units_entry.insert(0, "pixels")
        self.conversion_units_entry.grid(row=2, column=1, padx=32)

        axis_limits_label = ttk.Label(graph_opts_frame, text="Set 'y' axis limits for plots in units specified above\n(Leave blank for default)")
        axis_limits_label.grid(row=3, column=0, pady=(0, 6), columnspan=2)
        lower_limit_label = ttk.Label(graph_opts_frame, text="Lower")
        lower_limit_label.grid(row=4, column=0)
        upper_limit_label = ttk.Label(graph_opts_frame, text="Upper")
        upper_limit_label.grid(row=4, column=1)
        self.lower_limit_entry = ttk.Entry(graph_opts_frame)
        self.lower_limit_entry.grid(row=5, column=0, pady=(0, 16))
        self.upper_limit_entry = ttk.Entry(graph_opts_frame)
        self.upper_limit_entry.grid(row=5, column=1, pady=(0, 16))
        set_limits_btn = ttk.Button(graph_opts_frame, text="Set limits", command=self.set_axis_limits, style='Regular.TButton')
        set_limits_btn.grid(row=6, column=0, columnspan=2)
        graph_opts_frame.grid(row=1, column=0, pady=16)

        # submission fields/buttons
        submission_frame = tk.Frame(section2)
        marker_deltas_btn = ttk.Button(submission_frame, text="Marker deltas analysis", command=lambda: analysis.analyze_marker_deltas((float(self.conversion_factor_entry.get()), self.conversion_units_entry.get())), style='Regular.TButton')
        marker_deltas_btn.grid(row=6, column=0, padx=4, pady=4)
        necking_pt_btn = ttk.Button(submission_frame, text="Necking point analysis", command=lambda: analysis.analyze_necking_point((float(self.conversion_factor_entry.get()), self.conversion_units_entry.get())), style='Regular.TButton')
        necking_pt_btn.grid(row=7, column=0, padx=4, pady=4)
        poissons_ratio_btn = ttk.Button(submission_frame, text="Poisson's ratio", command=lambda: analysis.poissons_ratio((float(self.conversion_factor_entry.get()), self.conversion_units_entry.get())), style='Regular.TButton')
        poissons_ratio_btn.grid(row=8, column=0, padx=4, pady=4)

        cell_distance_btn = ttk.Button(submission_frame, text="Marker distance", command=lambda: analysis.marker_distance((float(self.conversion_factor_entry.get()), self.conversion_units_entry.get())), style='Regular.TButton')
        cell_distance_btn.grid(row=6, column=1, padx=4, pady=4)
        cell_spread_btn = ttk.Button(submission_frame, text="Marker spread", command=lambda: analysis.single_marker_spread((float(self.conversion_factor_entry.get()), self.conversion_units_entry.get())), style='Regular.TButton')
        cell_spread_btn.grid(row=7, column=1, padx=4, pady=4)
        cell_velocity_btn = ttk.Button(submission_frame, text="Marker velocity", command=lambda: analysis.marker_velocity((float(self.conversion_factor_entry.get()), self.conversion_units_entry.get())), style='Regular.TButton')
        cell_velocity_btn.grid(row=8, column=1, padx=4, pady=4)
        
        self.cell_velocity_boxplot_opts_var = tk.IntVar()
        cell_velocity_boxplot_opts_check = ttk.Checkbutton(submission_frame, text="Velocity boxplot", variable=self.cell_velocity_boxplot_opts_var, onvalue=1, offvalue=0, command=self.handle_boxplot_opts_blit)
        cell_velocity_boxplot_opts_check.grid(row=9, column=1)
        self.cell_velocity_boxplot_opts_frame = tk.Frame(submission_frame)
        self.boxplot_keywords_label = ttk.Label(self.cell_velocity_boxplot_opts_frame, text="Enter list of conditions used in data labels\nfor boxplot categories separated by commas")
        self.boxplot_keywords_label.grid(row=0, column=0, pady=(16,4))
        self.boxplot_keywords_entry = ttk.Entry(self.cell_velocity_boxplot_opts_frame, width=40)
        self.boxplot_keywords_entry.grid(row=1, column=0)
        self.boxplot_button = ttk.Button(self.cell_velocity_boxplot_opts_frame, text="Cell velocities boxplot", command=self.call_boxplot, style='Regular.TButton')
        self.boxplot_button.grid(row=2, column=0, pady=(8, 20))

        exit_btn = ttk.Button(submission_frame, text='Exit', command=sys.exit, style='Regular.TButton')
        exit_btn.grid(row=20, column=0, columnspan=2, padx=32, pady=(24,12))
        submission_frame.grid(row=2, column=0)

        self.adjust_window_size()        
        self.scrollable_frame.bind("<Enter>", lambda e: self.scrollable_frame.bind_all("<MouseWheel>", self.on_mousewheel))
        self.scrollable_frame.bind("<Leave>", lambda e: self.scrollable_frame.unbind_all("<MouseWheel>"))
        section2.grid(row=0, column=1, padx=(64,16),sticky='n')

    def on_frame_configure(self, event):
        '''Reset the scroll region to encompass the inner frame and adjust window size if necessary'''
        # Update the canvas's scrollregion
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        # Check if the frame width has increased and adjust the canvas and window size
        frame_width = self.scrollable_frame.winfo_reqwidth()
        canvas_width = self.canvas.winfo_width()
        if frame_width > canvas_width:
            self.canvas.config(width=frame_width)
            self.root.geometry(f"{frame_width}x{self.root.winfo_height()}")

    def adjust_window_size(self):
        # Ensure Tkinter processes all geometry changes
        self.root.update_idletasks()

        # Get the required size of the scrollable_frame
        required_width = self.scrollable_frame.winfo_reqwidth()
        required_height = self.scrollable_frame.winfo_reqheight()

        # Adjust the canvas size to match the required size
        self.canvas.config(width=required_width, height=required_height)

        # Update the scrollregion to match the new size of scrollable_frame
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        # Now, adjust the window size if necessary
        self.update_window_size(required_width, required_height)

    def update_window_size(self, required_width, required_height):
        # Get the current window size
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()

        # Determine the new window size needed to accommodate the content
        new_width = max(window_width, required_width + 20) # Adding 20 for some padding
        new_height = max(window_height, required_height + 20) # Same reason

        # Configure the root window's size
        self.root.geometry(f"{new_width}x{new_height}")

    def set_axis_limits(self):
        with open("plot_opts/plot_customizations.json", 'r') as plot_customs_file:
            plot_customs_modified = json.load(plot_customs_file)

        try:
            y_lower = float(self.lower_limit_entry.get())
        except ValueError:
            y_lower = 'auto'
        try:
            y_upper = float(self.upper_limit_entry.get())
        except ValueError:
            y_upper = 'auto'

        plot_customs_modified['y_lower_bound'] = y_lower
        plot_customs_modified['y_upper_bound'] = y_upper
        print(plot_customs_modified)
        with open("plot_opts/plot_customizations.json", 'w') as plot_customs_file:
            json.dump(plot_customs_modified, plot_customs_file, indent=4)

    def select_frames(self):
        if self.video_path != "":
            self.child = FrameSelector(self.root, self.video_path, self.frame_label_var)
        else:
            msg = "Select a video before opening the frame selector"
            error_popup(msg)

    def remove_outliers(self):
        OutlierRemoval(self.root, (float(self.conversion_factor_entry.get()), self.conversion_units_entry.get()))

    def on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def handle_checkbuttons(self):
        if self.is_timelapse_var.get() == 1:
            self.frame_interval_frame.grid(row=6, column=0)
        else:
            self.frame_interval_frame.grid_forget()

    def handle_radios(self):
        """blits options for the corresponding radio button selected"""  
        option = TrackingOperation(self.operation_intvar.get())
        match option:
            case TrackingOperation.MARKERS:
                self.select_msg.grid_forget()
                self.necking_frame.grid_forget()
                self.area_frame.grid_forget()
                self.tracking_frame.grid(row=15, column=0)
            case TrackingOperation.NECKING:
                self.select_msg.grid_forget()
                self.tracking_frame.grid_forget()
                self.area_frame.grid_forget()
                self.necking_frame.grid(row=15, column=0)
            case TrackingOperation.AREA:
                self.select_msg.grid_forget()
                self.necking_frame.grid_forget()
                self.tracking_frame.grid_forget()
                self.area_frame.grid(row=15, column=0)

    def undo_last_tracking_append(self, fp):
        df = pd.read_csv(fp, index_col=None)
        print(df)
        n_entities = int(df.columns[-1][0])
        dropped_cols = [col for col in df.columns if f"{n_entities}-" in col]
        df.drop(columns=dropped_cols, inplace=True)
        msg = f"WARNING: This will remove the most recent tracking operation from: {fp}\n\n"+\
        "Click Ok to continue, or Cancel to exit"
        user_resp = warning_prompt(msg)
        if user_resp: # if user indicated to cont
            df.set_index('Frame')
            print(df)
            df.to_csv(fp, index=False)

    def check_data_label(self, output_fp, cur_label):
        data_label_err_flag = False
        out_df = pd.read_csv(output_fp)
        label_cols = [col for col in out_df.columns if 'data_label' in col]
        for col in label_cols:

            label = out_df[col].unique()
            for item in label:
                print(type(item), item)
                label_parsed = [item for item in label if isinstance(item, str)]
            print(label, label_parsed, cur_label)
            if cur_label == label_parsed[0]: # compare all prev range_ids to current
                msg = "WARNING: Current data label already exists in the output data.\n"+\
                "Please use a different data label for new data."
                warning_popup(msg)
                data_label_err_flag = True
                break
        
        return data_label_err_flag

    def handle_boxplot_opts_blit(self):
        if self.cell_velocity_boxplot_opts_var.get() == 1:
            self.cell_velocity_boxplot_opts_frame.grid(row=10, column=0, columnspan=2)
        else:
            self.cell_velocity_boxplot_opts_frame.grid_forget()

    def call_boxplot(self):
        conditions = self.boxplot_keywords_entry.get()
        conditions = conditions.split(',')
        conditions_parsed = [c.strip() for c in conditions]

        analysis.velocity_boxplot(conditions_parsed, (float(self.conversion_factor_entry.get()), self.conversion_units_entry.get()))

    def on_submit_tracking(self):
        """calls the appropriate functions with user spec'd args when tracking start button clicked"""        
        cap = cv2.VideoCapture(self.video_path) # load video
        if not cap.isOpened():
            msg = "Error: Couldn't open video file.\nPlease ensure one was selected, and it is not corrupted."
            error_popup(msg)
        option = TrackingOperation(self.operation_intvar.get())

        video_name = os.path.basename(self.video_path)

        # set frame start/end
        self.frame_start = 0
        self.frame_end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        if self.child: # if frame select window exists (if it was opened via the btn),
            if self.child.start_selection_flag: # if a start frame sel made,
                self.frame_start = self.child.frame_start_select
            if self.child.end_selection_flag: # if an end frame sel made,
                self.frame_end = self.child.frame_end_select   
        print(f"frame start: {self.frame_start}, frame end: {self.frame_end}")

        # handle if is timelapse inputs
        if self.is_timelapse_var.get() == 1:
            if self.frame_interval_entry.get() == '' or self.time_units_var.get == TimeUnits.UNSELECTED.value:
                msg = "ERROR: User indicated video is a timelapse, but did not specify time interval and/or time units"
                error_popup(msg)
                return
            else:
                self.frame_interval = float(self.frame_interval_entry.get())
                self.time_units = self.time_units_var.get()
        else:
            self.frame_interval = 0
            self.time_units = TimeUnits.SECONDS.value

        # handle overwrite/append selection
        if self.append_or_overwrite_var.get() == FileMode.UNSELECTED.value:
            msg = "ERROR: Please indicate if this tracking operation will append or overwrite existing tracking data"
            error_popup(msg)
            return
        else:
            file_mode = FileMode(self.append_or_overwrite_var.get())

        # handle frame skipping entry
        frame_record_interval = int(self.skip_frames_entry.get())

        # handle optional range identifier entry
        range_id = self.range_identifier_entry.get()
        data_label_err_flag = False
        if range_id.__contains__(','): # check if range_id has commas
            msg = "Error: Please avoid using commas in data labels."
            data_label_err_flag = True
            error_popup(msg)

        match option:
            case TrackingOperation.UNSELECTED:
                msg = "ERROR: Please select a radio button for a tracking operation."
                error_popup(msg)
            case TrackingOperation.MARKERS:
                print("Beginning Marker Tracking Process...")
                bbox_size = int(self.bbox_size_tracking_entry.get())
                tracker_choice = TrackerChoice(self.tracker_choice_intvar.get())
                if tracker_choice == TrackerChoice.UNSELECTED:
                    msg = "WARNING: Please select a tracking method\n\neither KCF for rigid shapes,\nor CSRT for deformable"
                    error_popup(msg)
                    return

                # check if range_id already used
                if file_mode == FileMode.APPEND: # only need to check prev ids if appending
                    data_label_err_flag = self.check_data_label('output/Tracking_Output.csv', range_id)

                if not data_label_err_flag:
                    selected_markers, first_frame = tracking.select_markers(cap, bbox_size, self.frame_start) # prompt to select markers
                    print(f"marker locs: {selected_markers}")
                    if not selected_markers.__contains__((-1,-1)): # select_markers returns list of -1 if selections cancelled
                        tracking.track_markers(
                            selected_markers,
                            first_frame,
                            self.frame_start,
                            self.frame_end,
                            cap,
                            bbox_size,
                            tracker_choice,
                            frame_record_interval,
                            self.frame_interval,
                            self.time_units,
                            file_mode,
                            video_name,
                            range_id
                        )
            case TrackingOperation.NECKING:
                percent_crop_right = float(self.percent_crop_right_entry.get())
                percent_crop_left = float(self.percent_crop_left_entry.get())
                binarize_intensity_thresh = int(self.binarize_intensity_thresh_entry.get())

                # check if range_id already used
                if file_mode == FileMode.APPEND: # only need to check prev ids if appending
                    data_label_err_flag = self.check_data_label('output/Necking_Point_Output.csv', range_id)

                if not data_label_err_flag:
                    print("Beginning Necking Point")
                    tracking.necking_point(
                        cap,
                        self.frame_start,
                        self.frame_end,
                        percent_crop_left,
                        percent_crop_right,
                        binarize_intensity_thresh,
                        frame_record_interval,
                        self.frame_interval,
                        self.time_units,
                        file_mode,
                        video_name,
                        range_id
                    )

            case TrackingOperation.AREA:
                # check if range_id already used
                if file_mode == FileMode.APPEND: # only need to check prev ids if appending
                    data_label_err_flag = self.check_data_label('output/Tracking_Output.csv', range_id)

                if not data_label_err_flag:
                    bbox_size = int(self.bbox_size_area_entry.get())
                    distance_from_marker_thresh = int(self.distance_from_marker_thresh_entry.get())
                    selected_markers, first_frame = tracking.select_markers(cap, bbox_size, self.frame_start) # prompt to select markers
                    print(f"marker locs: {selected_markers}")
                    if not selected_markers.__contains__((-1,-1)): # select_markers returns list of -1 if selections cancelled
                        tracking.track_area(cap,
                            selected_markers,
                            first_frame,
                            bbox_size,
                            self.frame_start,
                            self.frame_end,
                            frame_record_interval,
                            self.frame_interval,
                            self.time_units,
                            distance_from_marker_thresh,
                            file_mode,
                            video_name,
                            range_id
                        )

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
        self.frame_select_label.grid(pady=10)
        
        self.confirm_start_button = ttk.Button(self.child_window, text="Confirm start frame", command=self.confirm_start, style='Regular.TButton')
        self.confirm_start_button.grid()
        self.confirm_end_button = ttk.Button(self.child_window, text="Confirm end frame", command=self.confirm_end, style='Regular.TButton')
        self.confirm_end_button.grid()
        
        self.frame_display = ttk.Label(self.child_window, text="test")
        self.frame_display.grid(pady=10)
        self.slider = ttk.Scale(self.child_window, from_=0, to=self.n_frames - 1, orient="horizontal", length=400,
                                command=self.update_frames)
        self.slider.set(0)
        self.slider.grid(pady=10)

        self.child_window.focus_set()
        self.child_window.bind("<Left>", lambda event: self.on_left_arrow(event))
        self.child_window.bind("<Right>", lambda event: self.on_right_arrow(event))


        self.child_window.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_left_arrow(self, event):
        if event.state & 0x1 and event.state & 0x4:
            direction = -100
        elif event.state & 0x1:  # Only Shift is pressed
            direction = -10
        else:
            direction = -1
        new_frame = max(0, self.frame_start + direction)
        self.slider.set(new_frame)

    def on_right_arrow(self, event):
        if event.state & 0x1 and event.state & 0x4:
            direction = 100
        elif event.state & 0x1:  # Only Shift is pressed
            direction = 10
        else:
            direction = 1
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
            frame, _ = tracking.scale_frame(frame, 0.75)
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
    def __init__(self, parent, user_units):
        self.parent = parent
        self.user_units = user_units
        self.window = tk.Toplevel(self.parent)
        self.parent.title("Select outlier points to remove them")

        self.fig = None
        self.canvas = None
        self.labels = []

        # file options
        self.output_files = {
            'marker_deltas': 'output/Tracking_Output.csv',
            'necking_point': 'output/Necking_Point_Output.csv',
            'marker_tracking': 'output/Tracking_Output.csv',
            'surface_area': 'output/Surface_Area_Output.csv'
        }

        # prompt for input of which dataset
        self.which_dataset = 1
        self.which_dataset_label = ttk.Label(self.window, text="Enter number of which dataset to remove outliers from: ")
        self.which_dataset_label.grid(row=0, column=0, columnspan=2)
        self.which_dataset_entry = ttk.Entry(self.window)
        self.which_dataset_entry.insert(0, "1")
        self.which_dataset_entry.grid(row=0, column=2, padx=4, pady=8, columnspan=2)

        # radio buttons for file selection
        self.selected_data = tk.StringVar()
        self.marker_radio = ttk.Radiobutton(self.window, text="Marker deltas (Hydrogels)", variable=self.selected_data, value='marker_deltas', command=self.load_plot)
        self.marker_radio.grid(row=1, column=0, padx=6, pady=8)
        self.necking_radio = ttk.Radiobutton(self.window, text="Necking point (Hydrogels)", variable=self.selected_data, value='necking_point', command=self.load_plot)
        self.necking_radio.grid(row=1, column=1, padx=6, pady=8)
        self.marker_vel_radio = ttk.Radiobutton(self.window, text="Marker velocities (Cell tracking)", variable=self.selected_data, value='marker_tracking', command=self.load_plot)
        self.marker_vel_radio.grid(row=1, column=2, padx=6, pady=8)
        self.surface_area_radio = ttk.Radiobutton(self.window, text="Surface area (Cell tracking)", variable=self.selected_data, value='surface_area', command=self.load_plot)
        self.surface_area_radio.grid(row=1, column=3, padx=6, pady=8)

        self.undo_removals_button = ttk.Button(self.window, text="Undo Selections", command=self.undo_selections)
        self.undo_removals_button.grid(row=5, column=0, columnspan=2, pady=8)
        self.confirm_button = ttk.Button(self.window, text="Confirm Removal", comman=self.confirm)
        self.confirm_button.grid(row=5, column=2, columnspan=2, pady=8)

        self.confirm_label = ttk.Label(self.window, text="Points removed!", font=('TkDefaultFont', 12, 'bold'))
        self.removed_label = ttk.Label(self.window, text="Selections undone", font=('TkDefaultFont', 12, 'bold'))

    def load_data(self, is_updating=False):
        self.which_dataset = int(self.which_dataset_entry.get())
        if not is_updating: # initial loading and grabbing of relevant columns of dataset
            self.data_type = self.selected_data.get()
            self.fp = self.output_files[self.selected_data.get()]
            self.df = pd.read_csv(self.fp)
            self.x_col, self.x_label, _ = analysis.get_time_labels(self.df)
            relevant_columns = [col for col in self.df.columns if f"{self.which_dataset}-" in col]
            relevant_columns = ['Frame', self.x_col] + relevant_columns
            self.df = self.df[relevant_columns]

        # get appropriate columns
        self.x, self.y = None, None
        if self.data_type == 'marker_deltas':
            # time and marker distances
            self.x, self.y, self.plot_args = analysis.analyze_marker_deltas(self.user_units, self.df, False)
            self.n_sets = 2
        elif self.data_type == 'necking_point':
            # time and necking radial strain (necking pt length as a ratio)
            self.x, self.y, self.plot_args = analysis.analyze_necking_point(self.user_units, self.df, False)
            self.n_sets = 1
        elif self.data_type == 'marker_tracking':
            # time and velocity of marker
            try:
                self.x, self.y, self.plot_args, self.n_sets = analysis.marker_velocity(self.user_units, self.df, False, self.which_dataset)
            except IndexError as e:
                print(e)
                msg = f"Could not find dataset: {self.which_dataset} in the output data file"
                error_popup(msg)
        elif self.data_type == 'surface_area':
            # time and surface area of object being tracked
            try:
                self.x, self.y, self.plot_args, self.n_sets = analysis.single_marker_spread(self.user_units, self.df, False, self.which_dataset)
            except IndexError as e:
                print(e)
                msg = f"Could not find dataset: {self.which_dataset} in the output data file"
                error_popup(msg)

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

        # check if list 1D or 2D and plot accordingly   
        if all(isinstance(item, list) for item in self.y) or all(isinstance(item, np.ndarray) for item in self.y):
            for i, (self.cur_x, self.cur_y) in enumerate(zip(self.x, self.y)):
                self.plot = self.ax.plot(self.cur_x, self.cur_y, 'o', label=f"Tracker {i}", picker=True)
            self.ax.legend(loc='best')

        elif all(not isinstance(item, list) for item in self.y):
            self.plot = self.ax.plot(self.x, self.y, 'o', label=None, picker=True)

        plt.xlabel(self.plot_args['x_label'])
        plt.ylabel(self.plot_args['y_label'])
        plt.title(f"Outlier Removal for {self.data_type}")

        # only draw if new canvas
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().grid(row=10, column=0, columnspan=4)
        self.fig.canvas.mpl_connect('pick_event', self.onclick)

    def onclick(self, event):
        if event.name == 'pick_event':
            x_click = event.mouseevent.xdata
            # n_sets entries for each time point
            idx = (np.abs(self.x - x_click)).argmin() * self.n_sets

            # find and remove clicked pt
            x = self.df.iloc[idx][self.x_col]
            idx_remove = self.df[self.df[self.x_col] == x].index
            print(idx_remove)
            self.df.drop(index=idx_remove, inplace=True)
            self.df.reset_index(inplace=True, drop=True)

            # update plot
            self.plot_data(True)
            self.canvas.draw()

    def undo_selections(self):
        self.removed_label.grid(row=8, column=0, columnspan=4, pady=8)
        self.window.after(5000, lambda: self.removed_label.grid_forget())
        plt.clf()
        self.fig = None
        self.ax = None
        self.plot_data()
        
    def confirm(self):
        self.confirm_label.grid(row=9, column=0, columnspan=4, pady=8)
        self.window.after(5000, lambda: self.confirm_label.grid_forget())
        orig_df = pd.read_csv(self.fp)
        print(orig_df)
        for col in self.df.columns:
            orig_df[col] = self.df[col]
        orig_df.dropna(inplace=True)
        orig_df.set_index('Frame')
        print(orig_df)
        orig_df.to_csv(self.output_files[self.data_type], index=False)

    def load_plot(self):
        self.create_figure()
        self.plot_data()


if __name__ == '__main__':
    root = tk.Tk()
    window = TrackingUI(root)
    root.mainloop()
