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
        self.root.iconphoto(False, tk.PhotoImage(file="ico/m3b_comp.png"))


        # ttk widget stylings
        radio_btn_style = ttk.Style()
        radio_btn_style.configure("Outline.TButton", borderwidth=2, relief="solid", padding=(2, 5), foreground="black")
        btn_style = ttk.Style()
        btn_style.configure("Regular.TButton", padding=(10,5), relief="raised", width=20)
        small_text_btn_style = ttk.Style()
        small_text_btn_style.configure("LessYPadding.TButton", padding=(10,0), relief="raised", width=20, anchor='center')

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

        # buttons for selecting operation
        self.operation_intvar = tk.IntVar()
        self.operation_intvar.set(TrackingOperation.UNSELECTED.value)
        operation_frame = tk.Frame(self.section1)
        operation_tracking_radio = ttk.Radiobutton(operation_frame, text="Marker tracking", variable=self.operation_intvar, value=TrackingOperation.MARKERS.value, command=self.handle_radios, width=25, style='Outline.TButton')
        operation_tracking_radio.grid(row=0, column=0, padx=4, pady=(16, 4))
        operation_necking_radio = ttk.Radiobutton(operation_frame, text="Necking point detection", variable=self.operation_intvar, value=TrackingOperation.NECKING.value, command=self.handle_radios, width=25, style='Outline.TButton')
        operation_necking_radio.grid(row=0, column=1, padx=4, pady=(16, 4))
        operation_area_radio = ttk.Radiobutton(operation_frame, text="Surface area tracking", variable=self.operation_intvar, value=TrackingOperation.AREA.value, command=self.handle_radios, width=25, style='Outline.TButton')
        operation_area_radio.grid(row=0, column=2, padx=4, pady=(16,4))
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
        track_record_frame.grid(row=18, column=0, pady=(8,20))
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
        poissons_ratio_calc_btn = ttk.Button(submission_frame, text="Poisson's ratio\n(calculate)", command=lambda: analysis.poissons_ratio((float(self.conversion_factor_entry.get()), self.conversion_units_entry.get())), style='LessYPadding.TButton')
        poissons_ratio_calc_btn.grid(row=8, column=0, padx=4, pady=4)
        poissons_ratio_csv_btn = ttk.Button(submission_frame, text="Poisson's ratio\n(from csv)", command=lambda: analysis.poissons_ratio_csv((float(self.conversion_factor_entry.get()), self.conversion_units_entry.get())), style='LessYPadding.TButton')
        poissons_ratio_csv_btn.grid(row=9, column=0, padx=4, pady=4)

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

        data_selector_button = ttk.Button(submission_frame, text="Data selector", command=self.data_selector, style='Regular.TButton')
        data_selector_button.grid(row=19, column=0, columnspan=2, pady=(24,0))
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

    def data_selector(self):
        DataSelector(self.root, (float(self.conversion_factor_entry.get()), self.conversion_units_entry.get()))

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
            df.set_index('1-Frame')
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
        self.child_window.iconphoto(False, tk.PhotoImage(file="ico/m3b_comp.png"))

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
        self.window.title("Select outlier points to remove them")
        self.window.iconphoto(False, tk.PhotoImage(file="ico/m3b_comp.png"))

        # Get monitor resolution and set figure size
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        system_dpi = self.window.winfo_fpixels('1i')  # Fetches the DPI in some environments
        fig_width_in = (screen_width * 0.6) / system_dpi
        fig_height_in = (screen_height * 0.5) / system_dpi
        self.fig = plt.Figure(figsize=(fig_width_in, fig_height_in), dpi=system_dpi)
        self.ax = self.fig.add_subplot(111)

        # file options
        self.output_files = {
            'Longitudinal strain': 'output/Tracking_Output.csv',
            'Radial strain': 'output/Necking_Point_Output.csv',
            "Poisson's ratio (calculate)": 'output/Necking_Point_Output.csv',
            "Poisson's ratio (from csv)": 'output/poissons_ratio.csv',
            'Marker velocity': 'output/Tracking_Output.csv',
            'Marker RMS distance': 'output/Tracking_Output.csv',
            'Surface area': 'output/Surface_Area_Output.csv'
        }

        self.function_map = {
            'Longitudinal strain': analysis.analyze_marker_deltas,
            'Radial strain': analysis.analyze_necking_point,
            "Poisson's ratio": analysis.poissons_ratio,
            'Marker velocity': analysis.marker_velocity,
            'Marker RMS distance': analysis.marker_distance,
            'Surface area': analysis.single_marker_spread
        }

        # dynamically set figure/window size
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        system_dpi = self.window.winfo_fpixels('1i')  # This fetches the DPI in some environments
        fig_width_in = (screen_width * 0.4) / system_dpi
        fig_height_in = (screen_height * 0.5) / system_dpi

        # Setup for file selection and data label listbox
        file_selection_frame = tk.Frame(self.window)
        analysis_selector_label = ttk.Label(file_selection_frame, text="Select file type:")
        analysis_selector_label.grid(row=0, column=0)
        self.analysis_selector = ttk.Combobox(file_selection_frame, values=list(self.output_files.keys()))
        self.analysis_selector.set('Select analysis type')
        self.analysis_selector.grid(row=1, column=0, padx=8, pady=12)

        data_label_selector_label = ttk.Label(file_selection_frame, text="Select data labels:")
        data_label_selector_label.grid(row=0, column=1)
        self.data_label_selector = tk.Listbox(file_selection_frame, selectmode='single', exportselection=0, height=4)
        self.data_label_selector.grid(row=1, column=1, padx=8, pady=12)
        file_selection_frame.grid(row=0, column=0)

        # Control buttons
        buttons_frame = tk.Frame(self.window)
        self.generate_button = ttk.Button(buttons_frame, text='Generate plot', command=self.generate_plot)
        self.generate_button.grid(row=0, column=0, pady=20, padx=4)
        self.undo_removals_button = ttk.Button(buttons_frame, text="Undo Selections", command=self.undo_selections)
        self.undo_removals_button.grid(row=0, column=1, pady=8)
        self.confirm_button = ttk.Button(buttons_frame, text="Confirm Removal", command=self.confirm)
        self.confirm_button.grid(row=0, column=2, pady=8)
        buttons_frame.grid(row=1, column=0)

        # notif labels (show when clicking confirm or undo)
        self.confirm_label = ttk.Label(self.window, text="Points removed!", font=('TkDefaultFont', 12, 'bold'))
        self.removed_label = ttk.Label(self.window, text="Selections undone", font=('TkDefaultFont', 12, 'bold'))
        
        # Set up the matplotlib figure and canvas
        self.figure = plt.Figure(figsize=(fig_width_in, fig_height_in), dpi=system_dpi)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
        self.canvas.get_tk_widget().grid(row=5, column=0, padx=10, pady=10)

        # Bind combobox selection change to load data and update listbox
        self.analysis_selector.bind('<<ComboboxSelected>>', self.load_data_labels)
        self.cid = self.canvas.mpl_connect('button_press_event', self.on_click)

    def load_data_labels(self, event):
        selected_analysis = self.analysis_selector.get()
        self.csv_file_path = self.output_files.get(selected_analysis)
        if self.csv_file_path:
            self.df = pd.read_csv(self.csv_file_path)
            label_columns = [col for col in self.df.columns if 'data_label' in col]
            unique_values = set()
            self.label_to_dataset = {}
            for col in label_columns:
                dataset_num = int(col.split('-')[0])
                values = self.df[col].dropna().unique()
                for value in values:
                    unique_values.add(value)
                    self.label_to_dataset[value] = dataset_num  # Map each unique value to its dataset number
            self.data_label_selector.delete(0, tk.END)  # Clear the current listbox entries
            for value in sorted(unique_values):
                self.data_label_selector.insert(tk.END, value)
        else:
            self.data_label_selector.delete(0, tk.END)  # Clear the listbox if no csv file is selected
            self.data_label_selector.insert(tk.END, 'Choose from first selector')


    def generate_plot(self, is_refreshing=False):
        self.analysis_choice = self.analysis_selector.get()
        analysis_function = self.function_map.get(self.analysis_choice)
        if not is_refreshing:
            print("Generating plot...")
            selected_labels = [self.data_label_selector.get(idx) for idx in self.data_label_selector.curselection()]
            self.dataset_index = None
            if selected_labels:
                for column in self.df.columns:
                    if self.df[column].isin(selected_labels).any():
                        self.dataset_index = int(column.split('-', 1)[0])
                        print(f"Dataset index found: {self.dataset_index} in column: {column}")
                        break

        self.x_data, self.y_data, self.plot_args, _ = analysis_function(self.user_units, self.df, False, self.dataset_index)

        with open("plot_opts/plot_customizations.json", 'r') as plot_customs_file:
            plot_customs = json.load(plot_customs_file)
        font = plot_customs['font']

        self.ax.clear()  # Clear current plot
        self.ax.plot(self.x_data[0], self.y_data[0], 'o', markersize=4, label=self.plot_args['data_label'][0])  # Plot new data
        plt.xticks(fontsize=plot_customs['value_text_size'], fontfamily=font)
        plt.yticks(fontsize=plot_customs['value_text_size'], fontfamily=font) 
        self.ax.set_xlabel(self.plot_args['x_label'], fontsize=plot_customs['label_text_size'], fontfamily=font)
        self.ax.set_ylabel(self.plot_args['y_label'], fontsize=plot_customs['label_text_size'], fontfamily=font)
        plt.tick_params(axis='both', direction=plot_customs['tick_dir'])
        self.ax.set_title(self.plot_args['title'], fontsize=plot_customs['title_text_size'], fontfamily=font)
        y_lower_bound = None if plot_customs['y_lower_bound'] == 'auto' else float(plot_customs['y_lower_bound'])
        y_upper_bound = None if plot_customs['y_upper_bound'] == 'auto' else float(plot_customs['y_upper_bound'])    
        plt.ylim(y_lower_bound, y_upper_bound)
        self.canvas.draw()
        plt.tight_layout()

    def on_click(self, event):
        # Ignore clicks outside axes
        if event.inaxes is not self.ax:
            return

        # find index of the nearest point
        distances = np.sqrt((event.xdata - np.array(self.x_data[0]))**2 + (event.ydata - np.array(self.y_data[0]))**2)
        index = np.argmin(distances)

        relevant_columns = [col for col in self.df.columns if col.startswith(f"{self.dataset_index}-")]

        time_col, _, _ = analysis.get_time_labels(self.df, self.dataset_index)
        time_point = self.x_data[0][index]
        indices_to_remove = self.df[(self.df[time_col] == time_point)].index.tolist()
        print("indices:", indices_to_remove, "\nrel cols: ", relevant_columns)

        for col in relevant_columns:
            self.df.loc[indices_to_remove, col] = np.nan

        self.update_plot()

    def update_plot(self):
        # Clear the axes and replot with updated DataFrame
        self.ax.clear()
        self.generate_plot(is_refreshing=True)

    def undo_selections(self):
        self.removed_label.grid(row=2, column=0, pady=8)
        self.window.after(5000, lambda: self.removed_label.grid_forget())
        self.df = pd.read_csv(self.output_files[self.analysis_choice]) # reload original dataframe
        self.update_plot()
        
    def confirm(self):
        self.confirm_label.grid(row=3, column=0, pady=8)
        self.window.after(5000, lambda: self.confirm_label.grid_forget())
        orig_df = pd.read_csv(self.csv_file_path)
        print(orig_df)
        for col in self.df.columns:
            orig_df[col] = self.df[col]
        #orig_df.dropna(inplace=True)
        print(orig_df)
        orig_df.to_csv(self.csv_file_path, index=False)


class DataSelector:
    def __init__(self, parent, user_units):
        self.parent = parent
        self.user_units = user_units
        self.window = tk.Toplevel(self.parent)
        self.window.title("Select a region of a plot")
        self.window.iconphoto(False, tk.PhotoImage(file="ico/m3b_comp.png"))

        self.label_to_dataset = {}
        self.output_files = {
            'Longitudinal strain': 'output/Tracking_Output.csv',
            'Radial strain': 'output/Necking_Point_Output.csv',
            "Poisson's ratio (calculate)": 'output/Necking_Point_Output.csv',
            "Poisson's ratio (from csv)": 'output/poissons_ratio.csv',
            'Marker velocity': 'output/Tracking_Output.csv',
            'Marker RMS distance': 'output/Tracking_Output.csv',
            'Surface area': 'output/Surface_Area_Output.csv'
        }

        self.function_map = {
            'Longitudinal strain': analysis.analyze_marker_deltas,
            'Radial strain': analysis.analyze_necking_point,
            "Poisson's ratio (calculate)": analysis.poissons_ratio,
            "Poisson's ratio (from csv)": self.plot_poissons_from_csv,
            'Marker velocity': analysis.marker_velocity,
            'Marker RMS distance': analysis.marker_distance,
            'Surface area': analysis.single_marker_spread
        }

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        system_dpi = self.window.winfo_fpixels('1i')  # This fetches the DPI in some environments
        fig_width_in = (screen_width * 0.4) / system_dpi
        fig_height_in = (screen_height * 0.5) / system_dpi

        # Create a combobox widget for user units selection
        data_selection_frame = tk.Frame(self.window)
        analysis_selector_label = ttk.Label(data_selection_frame, text="Select analysis option")
        analysis_selector_label.grid(row=0, column=0)
        self.analysis_selector = ttk.Combobox(data_selection_frame, values=list(self.output_files.keys()))
        self.analysis_selector.set('Select analysis option')  # Set default placeholder text
        self.analysis_selector.grid(row=1, column=0, padx=8, pady=12)  # Pack the combobox into the window

        # Create a second combobox for data labels, initially empty
        data_label_selector_label = ttk.Label(data_selection_frame, text="Select data label(s)")
        data_label_selector_label.grid(row=0, column=1)
        self.data_label_selector = tk.Listbox(data_selection_frame, selectmode='multiple', exportselection=0, height=5)
        self.data_label_selector.grid(row=1, column=1, padx=8, pady=12)

        buttons_frame = tk.Frame(self.window)
        self.go_button = ttk.Button(buttons_frame, text='Go', command=self.execute_analysis)
        self.go_button.grid(row=2, column=0, pady=20, padx=4)

        self.reset_zoom_button = ttk.Button(buttons_frame, text="Reset zoom", command=self.reset_zoom)
        self.reset_zoom_button.grid(row=2, column=1, padx=4)

        self.save_plot_button = ttk.Button(buttons_frame, text="Save figure", command=self.save_plot)
        self.save_plot_button.grid(row=2, column=2, columnspan=2, padx=4)

        data_selection_frame.grid(row=0, column=0)
        buttons_frame.grid(row=1, column=0)

        # Bind the first selector to update the second selector when an option is selected
        self.analysis_selector.bind('<<ComboboxSelected>>', self.update_data_label_selector)

        # Set up the matplotlib figure and canvas
        self.figure = plt.Figure(figsize=(fig_width_in, fig_height_in), dpi=system_dpi)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.window)
        self.canvas.get_tk_widget().grid(row=5, column=0, padx=10, pady=10)

    def update_data_label_selector(self, event):
        selected_analysis = self.analysis_selector.get()
        csv_file_path = self.output_files.get(selected_analysis)
        if csv_file_path:
            self.df = pd.read_csv(csv_file_path)
            label_columns = [col for col in self.df.columns if 'data_label' in col]
            unique_values = set()
            self.label_to_dataset = {}  # Reinitialize to avoid holding outdated entries
            for col in label_columns:
                dataset_num = int(col.split('-', 1)[0])  # Assuming the first character is the dataset number
                values = self.df[col].dropna().unique()
                for value in values:
                    unique_values.add(value)
                    self.label_to_dataset[value] = dataset_num  # Map each unique value to its dataset number
            self.data_label_selector.delete(0, tk.END)  # Clear the current listbox entries
            for value in sorted(unique_values):
                self.data_label_selector.insert(tk.END, value)
        else:
            self.data_label_selector.delete(0, tk.END)  # Clear the listbox if no csv file is selected
            self.data_label_selector.insert(tk.END, 'Choose from first selector')

    def execute_analysis(self):
        selected_indices = self.data_label_selector.curselection()
        selected_labels = [self.data_label_selector.get(i) for i in selected_indices]
        selected_analysis = self.analysis_selector.get()
        analysis_func = self.function_map.get(selected_analysis)
        if not selected_indices:
            error_popup("No data labels selected.")
            return
        # Handle multiple results
        num_datasets = len(selected_labels)
        print(selected_labels)
        self.ax.clear()
        for selected_label in selected_labels:
            which_dataset = self.label_to_dataset.get(selected_label, 0)  # Retrieve dataset number from label
            if analysis_func and which_dataset:
                result = analysis_func(self.user_units, self.df, False, which_dataset)
                if result:
                    x, y, plot_args, _ = result
                    print(x, y, plot_args, num_datasets)
                    self.plot_data(x[0], y[0], plot_args, num_datasets)

    def plot_data(self, x, y, plot_args, num_datasets=1):
        with open("plot_opts/plot_customizations.json", 'r') as plot_customs_file:
            plot_customs = json.load(plot_customs_file)
        font = plot_customs['font']

        self.ax.plot(x, y, 'o', markersize=1, label=plot_args['data_label'][0])

        # adding legend depending on plot args
        if plot_args['has_legend']:
            if num_datasets <= 3:
                legend = self.ax.legend(loc='best', fontsize=plot_customs['legend_text_size'], prop={'family': font}, framealpha=0.3)
            else: # put legend outside plot if more than 3 datasets for readability
                legend = self.ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=plot_customs['legend_text_size'], prop={'family': font}, framealpha=0.3)
        else:
            legend = None

        plt.xticks(fontsize=plot_customs['value_text_size'], fontfamily=font)
        plt.yticks(fontsize=plot_customs['value_text_size'], fontfamily=font) 
        self.ax.set_xlabel(plot_args['x_label'], fontsize=plot_customs['label_text_size'], fontfamily=font)
        self.ax.set_ylabel(plot_args['y_label'], fontsize=plot_customs['label_text_size'], fontfamily=font)
        plt.tick_params(axis='both', direction=plot_customs['tick_dir'])
        self.ax.set_title(plot_args['title'], fontsize=plot_customs['title_text_size'], fontfamily=font)
        y_lower_bound = None if plot_customs['y_lower_bound'] == 'auto' else float(plot_customs['y_lower_bound'])
        y_upper_bound = None if plot_customs['y_upper_bound'] == 'auto' else float(plot_customs['y_upper_bound'])    
        plt.ylim(y_lower_bound, y_upper_bound)
        self.canvas.draw()
        plt.tight_layout()

        # Set up the span selector
        self.span = matplotlib.widgets.SpanSelector(self.ax, self.onselect, 'horizontal', useblit=True,
                                        props=dict(alpha=0.25, facecolor='blue'))
        
    def plot_poissons_from_csv(self, user_units, df, will_save_plot, which_dataset):
        pass
        '''plot_args = {
            'title': r"Poisson's Ratio - $\mathit{\nu(t)}$",
            'x_label': time_label,
            'y_label': r"Poisson's ratio, $\mathit{\nu}$",
            'data_label': data_labels,
            'has_legend': True
        }
        
        
        return times, radial_strains, plot_args, n_datasets

        '''
    def onselect(self, xmin, xmax):
        self.ax.set_xlim(xmin, xmax)
        self.canvas.draw_idle()

    def reset_zoom(self):
        self.ax.autoscale(True)
        self.canvas.draw_idle()

    def save_plot(self):
        # Save the current view of the figure
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 initialdir="figures/",
                                                 filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("TIFF files", "*.tiff")])
        if file_path:
            self.figure.savefig(file_path)

if __name__ == '__main__':
    root = tk.Tk()
    window = TrackingUI(root)
    root.mainloop()
