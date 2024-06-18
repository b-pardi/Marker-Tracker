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
import cProfile
import pstats
import os
import sys
import platform

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

        self.preprocess_vals = None

        self.setup_styles()

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
        file_selection_frame = tk.Frame(self.section1)
        self.video_path = ""
        file_btn = ttk.Button(file_selection_frame, text="Browse for video file", command=self.get_file, style='Regular.TButton')
        file_btn.grid(row=1, column=0, padx=32, pady=24)

        # file name label
        self.data_label_var = tk.StringVar()
        self.data_label_var.set("File not selected")
        data_label = ttk.Label(file_selection_frame, textvariable=self.data_label_var, style='Regular.TLabel')
        data_label.grid(row=2, column=0, pady=(0,8))

        file_selection_frame.grid(row=2, column=0)

        preprocess_frame = tk.Frame(self.section1)

        # video cropping and compression tools window
        compression_window_btn = ttk.Button(preprocess_frame, text="Crop/Compress video", command=self.video_compression, style='Regular.TButton')
        compression_window_btn.grid(row=0, column=0, pady=(12,0))

        # frame selection button
        self.frame_start = -1
        self.frame_end = -1
        self.child = None
        self.frame_label_var = tk.StringVar()
        self.frame_label_var.set("Frame range: FULL")
        self.frame_label = ttk.Label(preprocess_frame, textvariable=self.frame_label_var, style='Regular.TLabel')
        self.frame_selector_btn = ttk.Button(preprocess_frame, text="Select start/end frames", command=self.select_frames, style='Regular.TButton')
        self.frame_selector_btn.grid(row=0, column=1, pady=(12,0))
        self.frame_label.grid(row=1, column=1, pady=(0,8))

        self.frame_preprocessing_opts_btn = ttk.Button(preprocess_frame, text="Frame Preprocessing", command=self.frame_preprocessor, style='Regular.TButton', )
        self.frame_preprocessing_opts_btn.grid(row=0, column=2, pady=(12,0))

        preprocess_frame.grid(row=4, column=0)

        # frame interval input
        self.frame_interval = 0
        self.time_units = TimeUnits.SECONDS
        self.is_timelapse_var = tk.IntVar()
        is_timelapse_check = ttk.Checkbutton(self.section1, text="Is this video a timelapse? ", variable=self.is_timelapse_var, onvalue=1, offvalue=0, command=self.handle_checkbuttons, style='Regular.TCheckbutton')
        is_timelapse_check.grid(row=5, column=0)
        self.frame_interval_frame = tk.Frame(self.section1)
        time_units_label = ttk.Label(self.frame_interval_frame, text="Units of time:", style='Regular.TLabel')
        time_units_label.grid(row=0, column=0, columnspan=3, pady=6)

        time_units_frame = tk.Frame(self.frame_interval_frame)
        self.time_units_var = tk.StringVar()
        self.time_units_var.set(TimeUnits.UNSELECTED.value)
        time_units_seconds_radio = ttk.Radiobutton(time_units_frame, text='s', variable=self.time_units_var, value=TimeUnits.SECONDS.value, style='Regular.TRadiobutton')
        time_units_seconds_radio.grid(row=0, column=0)
        time_units_minutes_radio = ttk.Radiobutton(time_units_frame, text='min', variable=self.time_units_var, value=TimeUnits.MINUTES.value, style='Regular.TRadiobutton')
        time_units_minutes_radio.grid(row=0, column=1)
        time_units_hours_radio = ttk.Radiobutton(time_units_frame, text='hr', variable=self.time_units_var, value=TimeUnits.HOURS.value, style='Regular.TRadiobutton')
        time_units_hours_radio.grid(row=0, column=2)

        frame_interval_label = ttk.Label(self.frame_interval_frame, text="Frame interval: \n(image is taken every\n'x' unit of time)", style='Regular.TLabel')
        frame_interval_label.grid(row=2, column=0, padx=16, pady=16)
        self.frame_interval_entry = ttk.Entry(self.frame_interval_frame, style='Regular.TEntry')
        self.frame_interval_entry.grid(row=2, column=1)
        time_units_frame.grid(row=1, column=0, columnspan=2)

        # indicate if appending data
        self.append_or_overwrite_frame = tk.Frame(self.section1)
        self.append_or_overwrite_var = tk.IntVar()
        self.append_or_overwrite_var.set(FileMode.UNSELECTED.value)
        append_or_overwrite_label = ttk.Label(self.append_or_overwrite_frame, text="Append or overwrite existing tracking data?\n", style='Regular.TLabel')
        append_or_overwrite_label.grid(row=0, column=0, columnspan=2)
        self.append_radio = ttk.Radiobutton(self.append_or_overwrite_frame, text="Append", variable=self.append_or_overwrite_var, value=FileMode.APPEND.value, style='Regular.TRadiobutton')
        self.append_radio.grid(row=1, column=0)
        self.overwrite_radio = ttk.Radiobutton(self.append_or_overwrite_frame, text="Overwrite", variable=self.append_or_overwrite_var, value=FileMode.OVERWRITE.value, style='Regular.TRadiobutton')
        self.overwrite_radio.grid(row=1, column=1)
        self.append_or_overwrite_frame.grid(row=7, column=0, pady=12)

        # skip frames when tracking option
        skip_frames_frame = tk.Frame(self.section1)
        skip_frames_label_prefix = ttk.Label(skip_frames_frame, text="Track every ", style='Regular.TLabel')
        skip_frames_label_prefix.grid(row=0, column=0)
        self.skip_frames_entry = ttk.Entry(skip_frames_frame, width=4, style='Regular.TEntry')
        self.skip_frames_entry.insert(0, '1')
        self.skip_frames_entry.grid(row=0, column=1)
        skip_frames_label_postfix = ttk.Label(skip_frames_frame, text="frame(s)", style='Regular.TLabel')
        skip_frames_label_postfix.grid(row=0, column=2)
        skip_frames_frame.grid(row=8, column=0, pady=12)

        # optional range identifier
        range_identifier_label = ttk.Label(self.section1, text="Enter an optional identifier label for the tracking data: ", style='Regular.TLabel')
        range_identifier_label.grid(row=9, column=0, pady=(8,4))
        self.range_identifier_entry = ttk.Entry(self.section1, style='Regular.TEntry')
        self.range_identifier_entry.grid(row=10, column=0, pady=(0,8))

        # buttons for selecting operation
        self.operation_intvar = tk.IntVar()
        self.operation_intvar.set(TrackingOperation.UNSELECTED.value)
        operation_frame = tk.Frame(self.section1)
        
        operation_tracking_radio = ttk.Radiobutton(operation_frame, text="Marker tracking", variable=self.operation_intvar, value=TrackingOperation.MARKERS.value, command=self.handle_radios, width=20, style='Outline.TButton')
        operation_tracking_radio.grid(row=0, column=0, padx=4, pady=(16, 4))
        
        operation_necking_radio = ttk.Radiobutton(operation_frame, text="Necking pt detection", variable=self.operation_intvar, value=TrackingOperation.NECKING.value, command=self.handle_radios, width=20, style='Outline.TButton')
        operation_necking_radio.grid(row=0, column=1, padx=(4,1), pady=(16, 4))
       
        operation_area_radio = ttk.Radiobutton(operation_frame, text="Surface area tracking", variable=self.operation_intvar, value=TrackingOperation.AREA.value, command=self.handle_radios, width=20, style='Outline.TButton')
        operation_area_radio.grid(row=0, column=3, padx=4, pady=(16,4))
        
        operation_frame.grid(row=11, column=0)
        self.select_msg = ttk.Label(self.section1, text="Select from above for more customizable parameters", style='Regular.TLabel')
        self.select_msg.grid(row=12, column=0)

        # options for marker tracking
        self.tracking_frame = tk.Frame(self.section1)
        bbox_tracking_size_label = ttk.Label(self.tracking_frame, text="Tracker bounding box size (px)", style='Regular.TLabel')
        bbox_tracking_size_label.grid(row=0, column=0, padx=4, pady=8)
        self.bbox_size_tracking_entry = ttk.Entry(self.tracking_frame, style='Regular.TEntry')
        self.bbox_size_tracking_entry.insert(0, "100")
        self.bbox_size_tracking_entry.grid(row=0, column=1, padx=4, pady=8)

        self.tracker_choice_intvar = tk.IntVar()
        self.tracker_choice_intvar.set(TrackerChoice.UNSELECTED.value)
        tracker_choice_label = ttk.Label(self.tracking_frame, text="Choose tracking algorithm", style='Regular.TLabel')
        tracker_choice_label.grid(row=1, column=0, columnspan=2, padx=4, pady=(12,4))
        tracker_KCF_radio = ttk.Radiobutton(self.tracking_frame, text="KCF tracker\n(best for consistent shape tracking)", variable=self.tracker_choice_intvar, value=TrackerChoice.KCF.value, width=30, style='Outline.TButton')
        tracker_KCF_radio.grid(row=2, column=0, padx=4)
        tracker_CSRT_radio = ttk.Radiobutton(self.tracking_frame, text="CSRT tracker\n(best for deformable shape tracking)", variable=self.tracker_choice_intvar, value=TrackerChoice.CSRT.value, width=30, style='Outline.TButton')
        tracker_CSRT_radio.grid(row=2, column=1, padx=4)
        tracker_klt_radio = ttk.Radiobutton(self.tracking_frame, text="KLT optical flow\n(best for slow objects)", variable=self.tracker_choice_intvar, value=TrackerChoice.KLT.value, width=30, style='Outline.TButton')
        tracker_klt_radio.grid(row=2, column=2, padx=4)
        
        # options for necking point
        self.necking_frame = tk.Frame(self.section1)
        self.necking_method_frame = tk.Frame(self.necking_frame)
        self.necking_method_intvar = tk.IntVar()
        necking_min_distance_radio = ttk.Radiobutton(self.necking_method_frame, text="Minimum distance", variable=self.necking_method_intvar, value=NeckingPointMethod.MINIMUM_DISTANCE.value, command=self.handle_necking_method_frames, width=20, style='Outline.TButton')
        necking_min_distance_radio.grid(row=1, column=0, pady=4)
        necking_midpt_radio = ttk.Radiobutton(self.necking_method_frame, text="Midpoint", variable=self.necking_method_intvar, value=NeckingPointMethod.MIDPOINT.value, command=self.handle_necking_method_frames, width=20, style='Outline.TButton')
        necking_midpt_radio.grid(row=1, column=1, pady=4)
        necking_step_radio = ttk.Radiobutton(self.necking_method_frame, text="Step approximation", variable=self.necking_method_intvar, value=NeckingPointMethod.STEP_APPROXIMATION.value, command=self.handle_necking_method_frames, width=20, style='Outline.TButton')
        necking_step_radio.grid(row=1, column=2, pady=4)
        self.necking_method_frame.grid(row=0, column=0, columnspan=3)

        binarize_intensity_thresh_label = ttk.Label(self.necking_frame, text="pixel intensity value for frame binarization (0-255)", style='Regular.TLabel')
        binarize_intensity_thresh_label.grid(row=2, column=0, columnspan=2, padx=4, pady=4, sticky='w')
        self.binarize_intensity_thresh_entry = ttk.Entry(self.necking_frame, style='Regular.TEntry')
        self.binarize_intensity_thresh_entry.insert(0, "120")
        self.binarize_intensity_thresh_entry.grid(row=2, column=2, padx=4, pady=4)

        # minimum distance method args
        self.necking_min_dist_frame = tk.Frame(self.necking_frame)
        percent_crop_label = ttk.Label(self.necking_min_dist_frame, text="% of video width to exclude outter edges of\n(minimum distance and step approximation methods only)", style='Regular.TLabel')
        percent_crop_label.grid(row=3, column=0, rowspan=2, padx=4, pady=4)
        percent_crop_left_label = ttk.Label(self.necking_min_dist_frame, text="left edge", style='Regular.TLabel') 
        percent_crop_left_label.grid(row=3, column=1)    
        self.percent_crop_left_entry = ttk.Entry(self.necking_min_dist_frame, style='Regular.TEntry')
        self.percent_crop_left_entry.insert(0, "0")
        self.percent_crop_left_entry.grid(row=4, column=1, padx=4, pady=4)

        percent_crop_right_label = ttk.Label(self.necking_min_dist_frame, text="right edge", style='Regular.TLabel')     
        percent_crop_right_label.grid(row=3, column=2)    
        self.percent_crop_right_entry = ttk.Entry(self.necking_min_dist_frame, style='Regular.TEntry')
        self.percent_crop_right_entry.insert(0, "0")
        self.percent_crop_right_entry.grid(row=4, column=2, padx=4, pady=4)

        # midpt specific args
        self.necking_midpt_frame = tk.Frame(self.necking_frame)
        bbox_tracking_size_label = ttk.Label(self.necking_midpt_frame, text="Tracker bounding box size (px) (midpt method only)", style='Regular.TLabel')
        bbox_tracking_size_label.grid(row=5, column=0, columnspan=2, padx=4, pady=4, sticky='w')
        self.bbox_size_necking_midpt_entry = ttk.Entry(self.necking_midpt_frame, style='Regular.TEntry')
        self.bbox_size_necking_midpt_entry.insert(0, "100")
        self.bbox_size_necking_midpt_entry.grid(row=5, column=2, padx=4, pady=4)

        # step function approximation args
        self.necking_step_frame = tk.Frame(self.necking_frame)
        step_num_label = ttk.Label(self.necking_step_frame, text="Number of steps (step approximation method only)", style='Regular.TLabel')
        step_num_label.grid(row=6, column=0, columnspan=1, padx=4, pady=4, sticky='w')
        self.step_num_entry = ttk.Entry(self.necking_step_frame, style='Regular.TEntry')
        self.step_num_entry.insert(0, "10")
        self.step_num_entry.grid(row=6, column=1, columnspan=2, padx=4, pady=4)

        step_percent_crop_label = ttk.Label(self.necking_step_frame, text="% of video width to exclude outter edges of\n(minimum distance and step approximation methods only)", style='Regular.TLabel')
        step_percent_crop_label.grid(row=3, column=0, rowspan=2, padx=4, pady=4)
        step_percent_crop_left_label = ttk.Label(self.necking_step_frame, text="left edge", style='Regular.TLabel') 
        step_percent_crop_left_label.grid(row=3, column=1)    
        self.step_percent_crop_left_entry = ttk.Entry(self.necking_step_frame, style='Regular.TEntry')
        self.step_percent_crop_left_entry.insert(0, "0")
        self.step_percent_crop_left_entry.grid(row=4, column=1, padx=4, pady=4)

        step_percent_crop_right_label = ttk.Label(self.necking_step_frame, text="right edge", style='Regular.TLabel')     
        step_percent_crop_right_label.grid(row=3, column=2)    
        self.step_percent_crop_right_entry = ttk.Entry(self.necking_step_frame, style='Regular.TEntry')
        self.step_percent_crop_right_entry.insert(0, "0")
        self.step_percent_crop_right_entry.grid(row=4, column=2, padx=4, pady=4)

        # options for surface area tracking
        self.area_frame = tk.Frame(self.section1)
        bbox_area_size_label = ttk.Label(self.area_frame, text="Tracker bounding box size (px)", style='Regular.TLabel')
        bbox_area_size_label.grid(row=0, column=0, padx=4, pady=4)
        self.bbox_size_area_entry = ttk.Entry(self.area_frame, style='Regular.TEntry')
        self.bbox_size_area_entry.insert(0, "100")
        self.bbox_size_area_entry.grid(row=0, column=1, padx=4, pady=4)
        distance_from_marker_thresh_label = ttk.Label(self.area_frame, text="Max distance from marker to find contours (px)", style='Regular.TLabel')
        distance_from_marker_thresh_label.grid(row=1, column=0, padx=4, pady=4)
        self.distance_from_marker_thresh_entry = ttk.Entry(self.area_frame, style='Regular.TEntry')
        self.distance_from_marker_thresh_entry.insert(0, "150")
        self.distance_from_marker_thresh_entry.grid(row=1, column=1, padx=4, pady=4)
        
        self.preprocessing_choice_var = tk.IntVar()
        self.preprocessing_choice_var.set(PreprocessingIssue.NONE.value)
        preprocessing_label = ttk.Label(self.area_frame, text="Indicate the potential issue with your video that needs preprocessing", style='Regular.TLabel')
        preprocessing_label.grid(row=3, column=0, columnspan=2)
        noisy_bg_radio = ttk.Radiobutton(self.area_frame, text="Noisy background\n(grainy or speckled artifacts)", variable=self.preprocessing_choice_var, value=PreprocessingIssue.NOISY_BG.value, style='Regular.TRadiobutton')        
        noisy_bg_radio.grid(row=4, column=0, pady=4)
        noisy_bg_radio = ttk.Radiobutton(self.area_frame, text="Harsh gradients\n(distinct darker edges or rings in video,\nwith lighter tracked subjects)", variable=self.preprocessing_choice_var, value=PreprocessingIssue.HARSH_GRADIENT.value, style='Regular.TRadiobutton')        
        noisy_bg_radio.grid(row=4, column=1, pady=4)
        salt_pepper_radio = ttk.Radiobutton(self.area_frame, text="Salt+Pepper noise", variable=self.preprocessing_choice_var, value=PreprocessingIssue.SALT_PEPPER.value, style='Regular.TRadiobutton')        
        salt_pepper_radio.grid(row=5, column=0, pady=(0,4))
        custom_radio = ttk.Radiobutton(self.area_frame, text="Custom", variable=self.preprocessing_choice_var, value=PreprocessingIssue.CUSTOM.value, style='Regular.TRadiobutton')        
        custom_radio.grid(row=5, column=1, pady=(0,4))
        no_processing_radio = ttk.Radiobutton(self.area_frame, text="No preprocessing", variable=self.preprocessing_choice_var, value=PreprocessingIssue.NONE.value, style='Regular.TRadiobutton')        
        no_processing_radio.grid(row=6, column=0, columnspan=2, pady=(0,4))

        # tracking operation buttons
        track_record_frame = tk.Frame(self.section1)
        self.multithread_intvar = tk.IntVar()
        multithread_check = ttk.Checkbutton(track_record_frame, text="Use multi-threading? (unstable but faaast)", variable=self.multithread_intvar, offvalue=0, onvalue=1, style='Regular.TCheckbutton')
        #multithread_check.grid(row=0, column=0, columnspan=2, pady=(24,0))
        track_btn = ttk.Button(track_record_frame, text="Begin tracking", command=self.on_submit_tracking, style='Regular.TButton')
        track_btn.grid(row=2, column=0, columnspan=2, padx=32, pady=(24,4))
        
        undo_buttons_frame = tk.Frame(track_record_frame)
        undo_label = ttk.Label(undo_buttons_frame, text="Undo previous appended tracking data recording.", style='Regular.TLabel')
        undo_label.grid(row=0, column=0, columnspan=3)
        undo_marker_tracking_append_btn = ttk.Button(undo_buttons_frame, text="Marker tracking", command=lambda: self.undo_last_tracking_append("output/Tracking_Output.csv"), style='Regular.TButton')
        undo_marker_tracking_append_btn.grid(row=1, column=0)        
        undo_necking_point_append_btn = ttk.Button(undo_buttons_frame, text="Necking point", command=lambda: self.undo_last_tracking_append("output/Necking_Point_Output.csv"), style='Regular.TButton')
        undo_necking_point_append_btn.grid(row=1, column=1)    
        undo_surface_tracking_append_btn = ttk.Button(undo_buttons_frame, text="Surface area", command=lambda: self.undo_last_tracking_append("output/Surface_Area_Output.csv"), style='Regular.TButton')
        undo_surface_tracking_append_btn.grid(row=1, column=2)
        undo_buttons_frame.grid(row=4, column=0, columnspan=2, pady=(8,20))
        track_record_frame.grid(row=18, column=0, pady=(8,20))
        self.section1.grid(row=0, column=0)

        # section 2 pertains to data analysis
        section2 = tk.Frame(self.scrollable_frame)
        sect2_header = ttk.Label(section2, text="Data Analysis", font=('TkDefaultFont', 14, 'bold'))
        sect2_header.grid(row=0, column=0)

        # some data dependent graph options (more in plot_opts.json)
        graph_opts_frame = tk.Frame(section2)
        graph_opts_label = ttk.Label(graph_opts_frame, text="Data dependent graph options", style='Regular.TLabel')
        graph_opts_label.grid(row=0, column=0, columnspan=2, pady=(0,12))
        conversion_factor_label = ttk.Label(graph_opts_frame, text="Enter the conversion factor\nto convert pixels to\nyour desired units: ", style='Regular.TLabel')
        conversion_factor_label.grid(row=1, column=0)
        self.conversion_factor_entry = ttk.Entry(graph_opts_frame, style='Regular.TEntry')
        self.conversion_factor_entry.insert(0, "1.0")
        self.conversion_factor_entry.grid(row=1, column=1, padx=32)
        conversion_units_label = ttk.Label(graph_opts_frame, text="Enter the units that result\nfrom this conversion\n(mm, µm, nm, etc.): ", style='Regular.TLabel')
        conversion_units_label.grid(row=2, column=0, pady=(0,16))
        self.conversion_units_entry = ttk.Entry(graph_opts_frame, style='Regular.TEntry')
        self.conversion_units_entry.insert(0, "pixels")
        self.conversion_units_entry.grid(row=2, column=1, padx=32)

        axis_limits_label = ttk.Label(graph_opts_frame, text="Set 'y' axis limits for plots in units specified above\n(Leave blank for default)", style='Regular.TLabel')
        axis_limits_label.grid(row=3, column=0, pady=(0, 6), columnspan=2)
        lower_limit_label = ttk.Label(graph_opts_frame, text="Lower", style='Regular.TLabel')
        lower_limit_label.grid(row=4, column=0)
        upper_limit_label = ttk.Label(graph_opts_frame, text="Upper", style='Regular.TLabel')
        upper_limit_label.grid(row=4, column=1)
        self.lower_limit_entry = ttk.Entry(graph_opts_frame, style='Regular.TEntry')
        self.lower_limit_entry.grid(row=5, column=0, pady=(0, 16))
        self.upper_limit_entry = ttk.Entry(graph_opts_frame, style='Regular.TEntry')
        self.upper_limit_entry.grid(row=5, column=1, pady=(0, 16))
        set_limits_btn = ttk.Button(graph_opts_frame, text="Set limits", command=self.set_axis_limits, style='Regular.TButton')
        set_limits_btn.grid(row=6, column=0, columnspan=2)
        graph_opts_frame.grid(row=1, column=0, pady=16)

        # submission fields/buttons
        submission_frame = tk.Frame(section2)
        marker_deltas_btn = ttk.Button(submission_frame, text="Marker deltas analysis", command=lambda: self.call_analysis(AnalysisType.MARKER_DELTAS), style='Regular.TButton')
        marker_deltas_btn.grid(row=6, column=0, padx=4, pady=4)
        necking_pt_btn = ttk.Button(submission_frame, text="Necking point analysis", command=lambda: self.call_analysis(AnalysisType.NECKING_POINT), style='Regular.TButton')
        necking_pt_btn.grid(row=7, column=0, padx=4, pady=4)
        poissons_ratio_calc_btn = ttk.Button(submission_frame, text="Poisson's ratio\n(calculate)", command=lambda: self.call_analysis(AnalysisType.POISSONS_RATIO), style='LessYPadding.TButton')
        poissons_ratio_calc_btn.grid(row=8, column=0, padx=4, pady=4)
        poissons_ratio_csv_btn = ttk.Button(submission_frame, text="Poisson's ratio\n(from csv)", command=lambda: self.call_analysis(AnalysisType.POISSONS_RATIO_CSV), style='LessYPadding.TButton')
        poissons_ratio_csv_btn.grid(row=9, column=0, padx=4, pady=4)

        locator_choice_label = ttk.Label(submission_frame, text="Locator choice for:\ndiplacement, distance, and velocity", style='Regular.TLabel')
        locator_choice_label.grid(row=4, column=1, padx=4, pady=(8,2))
        self.locator_choice_var = tk.StringVar()
        self.locator_choice_var.set(LocatorType.BBOX.value)
        locator_marker_choice_radio = ttk.Radiobutton(submission_frame, text='Marker (use marker tracker)', variable=self.locator_choice_var, value=LocatorType.BBOX.value, style='Regular.TRadiobutton')
        locator_marker_choice_radio.grid(row=5, column=1)
        locator_choice_centroid_radio = ttk.Radiobutton(submission_frame, text='Centroid (use surface area)', variable=self.locator_choice_var, value=LocatorType.CENTROID.value, style='Regular.TRadiobutton')
        locator_choice_centroid_radio.grid(row=6, column=1)
        
        cell_distance_btn = ttk.Button(submission_frame, text="Marker distance", command=lambda: self.call_analysis(AnalysisType.DISTANCE), style='Regular.TButton')
        cell_distance_btn.grid(row=7, column=1, padx=4, pady=4)
        cell_displacement_btn = ttk.Button(submission_frame, text="Marker displacement", command=lambda: self.call_analysis(AnalysisType.DISPLACEMENT), style='Regular.TButton')
        cell_displacement_btn.grid(row=8, column=1, padx=4, pady=4)
        cell_spread_btn = ttk.Button(submission_frame, text="Marker spread", command=lambda: self.call_analysis(AnalysisType.SURFACE_AREA), style='Regular.TButton')
        cell_spread_btn.grid(row=10, column=1, padx=4, pady=4)
        cell_velocity_btn = ttk.Button(submission_frame, text="Marker velocity", command=lambda: self.call_analysis(AnalysisType.VELOCITY), style='Regular.TButton')
        cell_velocity_btn.grid(row=9, column=1, padx=4, pady=4)
    
        remove_outliers_button = ttk.Button(submission_frame, text="Remove outliers", command=self.remove_outliers, style='Regular.TButton')
        remove_outliers_button.grid(row=18, column=0, columnspan=2, padx=32, pady=(24,0))
        
        data_selector_button = ttk.Button(submission_frame, text="Data selector", command=self.data_selector, style='Regular.TButton')
        data_selector_button.grid(row=19, column=0, columnspan=2, pady=(4,0))
        
        box_plotter_button = ttk.Button(submission_frame, text="Box plotter", command=self.box_plotter, style='Regular.TButton')
        box_plotter_button.grid(row=20, column=0, columnspan=2, pady=(4,0))
        
        exit_btn = ttk.Button(submission_frame, text='Exit', command=sys.exit, style='Regular.TButton')
        exit_btn.grid(row=25, column=0, columnspan=2, padx=32, pady=(24,12))
        submission_frame.grid(row=2, column=0)
        section2.grid(row=0, column=1, padx=(64,16),sticky='n')

        self.adjust_window_size()  

        # bind scroll wheel for windows and mac
        self.scrollable_frame.bind("<Enter>", self.bind_scroll_events)
        self.scrollable_frame.bind("<Leave>", self.unbind_scroll_events)   
        '''  
        self.scrollable_frame.bind("<Enter>", lambda e: self.scrollable_frame.bind_all("<MouseWheel>", self.on_mousewheel))
        self.scrollable_frame.bind("<Leave>", lambda e: self.scrollable_frame.unbind_all("<MouseWheel>"))
        '''
    
    def setup_styles(self):
        # ttk widget stylings
        self.radio_btn_style = ttk.Style()
        self.radio_btn_style.configure(
            "Outline.TButton",
            borderwidth=2,
            relief="solid",
            padding=(2, 5),
            foreground="black",
            background="white"
        )
        self.btn_style = ttk.Style()
        self.btn_style.configure(
            "Regular.TButton",
            padding=(10, 5),
            relief="raised",
            width=20
        )
        self.small_text_btn_style = ttk.Style()
        self.small_text_btn_style.configure(
            "LessYPadding.TButton",
            padding=(10, 0),
            relief="raised",
            width=20,
            anchor='center'
        )

        # "Outline.TButton" style map
        self.radio_btn_style.map(
            "Outline.TButton",
            foreground=[('selected', 'blue'), ('!selected', 'black')],
            background=[('selected', 'blue'), ('!selected', 'white')]
        )

        self.label_style = ttk.Style()
        self.label_style.configure(
            "Regular.TLabel",
            padding=(10, 5)
        )

        self.checkbox_style = ttk.Style()
        self.checkbox_style.configure(
            "Regular.TCheckbutton",
            padding=(10, 5)
        )

        self.radio_button_style = ttk.Style()
        self.radio_button_style.configure(
            "Regular.TRadiobutton",
            padding=(10, 5)
        )

        self.entry_style = ttk.Style()
        self.entry_style.configure(
            "Regular.TEntry",
            padding=(10, 5)
        )

        # Platform-specific adjustments
        if platform.system() == 'Darwin' or platform.system() == 'Linux':  # macOS and Linux
            self.radio_btn_style.configure(
                "Outline.TButton",
                font=("Helvetica", 12),
                padding=(4, 8),
                width=15
            )
            self.btn_style.configure(
                "Regular.TButton",
                font=("Helvetica", 12),
                padding=(12, 8),
                width=15
            )
            self.small_text_btn_style.configure(
                "LessYPadding.TButton",
                font=("Helvetica", 12),
                padding=(12, 6),
                width=15
            )
            self.label_style.configure(
                "Regular.TLabel",
                font=("Helvetica", 12),
                padding=(12, 8)
            )
            self.checkbox_style.configure(
                "Regular.TCheckbutton",
                font=("Helvetica", 12),
                padding=(12, 8)
            )
            self.radio_button_style.configure(
                "Regular.TRadiobutton",
                font=("Helvetica", 12),
                padding=(12, 8)
            )
            self.entry_style.configure(
                "Regular.TEntry",
                font=("Helvetica", 12),
                padding=(8, 4)
            )

        elif platform.system() == 'Windows':  # Windows
            self.radio_btn_style.configure(
                "Outline.TButton",
                font=("Segoe UI", 10),
                padding=(2, 5),
                width=20
            )
            self.btn_style.configure(
                "Regular.TButton",
                font=("Segoe UI", 10),
                padding=(10, 5),
                width=20
            )
            self.small_text_btn_style.configure(
                "LessYPadding.TButton",
                font=("Segoe UI", 10),
                padding=(10, 0),
                width=20
            )
            self.label_style.configure(
                "Regular.TLabel",
                font=("Segoe UI", 10),
                padding=(10, 5)
            )
            self.checkbox_style.configure(
                "Regular.TCheckbutton",
                font=("Segoe UI", 10),
                padding=(10, 5)
            )
            self.radio_button_style.configure(
                "Regular.TRadiobutton",
                font=("Segoe UI", 10),
                padding=(10, 5)
            )
            self.entry_style.configure(
                "Regular.TEntry",
                font=("Segoe UI", 10),
                padding=(6, 2)
            )

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

    def frame_preprocessor(self):
        if self.video_path != "":
            FramePreprocessor(self, self.video_path, self.preprocess_vals)
        else:
            msg = "Select a video before opening the video preprocessing tool"
            error_popup(msg)

    def remove_outliers(self):
        OutlierRemoval(self, float(self.conversion_factor_entry.get()), self.conversion_units_entry.get())

    def data_selector(self):
        DataSelector(self, float(self.conversion_factor_entry.get()), self.conversion_units_entry.get())

    def box_plotter(self):
        Boxplotter(self, float(self.conversion_factor_entry.get()), self.conversion_units_entry.get()), 

    def video_compression(self):
        if self.video_path != "":
            CropAndCompressVideo(self, self.data_label_var)
        else:
            msg = "Select a video before opening the video compression tool"
            error_popup(msg)
    
    def bind_scroll_events(self, event):
        if platform.system() == 'Darwin' or platform.system() == 'Linux': # macOS and Linux
            self.scrollable_frame.bind_all("<Button-4>", self.on_mousewheel)
            self.scrollable_frame.bind_all("<Button-5>", self.on_mousewheel)
        else: # windows OS
            self.scrollable_frame.bind_all("<MouseWheel>", self.on_mousewheel)

    def unbind_scroll_events(self, event):
        if platform.system() == 'Darwin' or platform.system() == 'Linux': # macOS and Linux
            self.scrollable_frame.unbind_all("<Button-4>")
            self.scrollable_frame.unbind_all("<Button-5>")
        else: # windows OS
            self.scrollable_frame.unbind_all("<MouseWheel>")

    def on_mousewheel(self, event):
        if platform.system() == 'Darwin' or platform.system() == 'Linux': # macOS and Linux
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")
        else: # windows OS
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def handle_checkbuttons(self):
        if self.is_timelapse_var.get() == 1:
            self.frame_interval_frame.grid(row=6, column=0)
        else:
            self.frame_interval_frame.grid_forget()

    def handle_necking_method_frames(self):
        necking_method = NeckingPointMethod(self.necking_method_intvar.get())
        match necking_method:
            case NeckingPointMethod.MINIMUM_DISTANCE:
                self.necking_min_dist_frame.grid(row=5, column=0, columnspan=3)
                self.necking_midpt_frame.grid_forget()
                self.necking_step_frame.grid_forget()
            case NeckingPointMethod.MIDPOINT:
                self.necking_min_dist_frame.grid_forget()
                self.necking_midpt_frame.grid(row=5, column=0, columnspan=3)
                self.necking_step_frame.grid_forget()
            case NeckingPointMethod.STEP_APPROXIMATION:
                self.necking_min_dist_frame.grid_forget()
                self.necking_midpt_frame.grid_forget()
                self.necking_step_frame.grid(row=5, column=0, columnspan=3)

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

    def call_analysis(self, analysis_type):
        conversion_factor = float(self.conversion_factor_entry.get())
        conversion_units = self.conversion_units_entry.get()
        locator_choice = LocatorType(self.locator_choice_var.get())

        if analysis_type == AnalysisType.MARKER_DELTAS:
            analysis.analyze_marker_deltas(conversion_factor, conversion_units)
        if analysis_type == AnalysisType.NECKING_POINT:
            analysis.analyze_necking_point(conversion_factor, conversion_units)
        if analysis_type == AnalysisType.POISSONS_RATIO:
            analysis.poissons_ratio(conversion_factor, conversion_units)
        if analysis_type == AnalysisType.POISSONS_RATIO_CSV:
            analysis.poissons_ratio_csv()
        
        if analysis_type == AnalysisType.DISPLACEMENT:
            analysis.marker_movement_analysis(analysis_type, conversion_factor, conversion_units, 'output/displacement.csv', f'displacement', locator_type=locator_choice)
        if analysis_type == AnalysisType.DISTANCE:
            analysis.marker_movement_analysis(analysis_type, conversion_factor, conversion_units, 'output/distance.csv', f'distance', locator_type=locator_choice)
        if analysis_type == AnalysisType.VELOCITY:
            analysis.marker_movement_analysis(analysis_type, conversion_factor, conversion_units, 'output/velocity.csv', f'velocity', locator_type=locator_choice)
        if analysis_type == AnalysisType.SURFACE_AREA:
            if locator_choice == LocatorType.BBOX: # marker tracking not viable for surface area
                msg = "Error: Please select Centroid locator type for surface area,\nand ensure surface area tracking was done previously"
                error_popup(msg)
            else:
                analysis.marker_movement_analysis(analysis_type, conversion_factor, conversion_units, 'output/surface_area.csv', f'surface_area', locator_type=locator_choice)

    def on_submit_tracking(self):
        """calls the appropriate functions with user spec'd args when tracking start button clicked"""        
        cap = cv2.VideoCapture(self.video_path) # load video

        print(cap, " \n opened from: ", self.video_path)
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

        # user indication of desire to use multithreaded version of tracking functions
        use_multithread = self.multithread_intvar.get()

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
                    selected_markers, first_frame = tracking.select_markers(cap, bbox_size, self.frame_start, self.preprocess_vals) # prompt to select markers
                    print(f"marker locs: {selected_markers}")
                    if not selected_markers.__contains__((-1,-1)): # select_markers returns list of -1 if selections cancelled
                        if use_multithread:
                            tracking.track_markers_threaded(
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
                        else:
                            if tracker_choice == TrackerChoice.KLT:
                                tracking.track_klt_optical_flow(
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
                            else:
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

                necking_method = NeckingPointMethod(self.necking_method_intvar.get())
                binarize_intensity_thresh = int(self.binarize_intensity_thresh_entry.get())

                # check if range_id already used
                if file_mode == FileMode.APPEND: # only need to check prev ids if appending
                    data_label_err_flag = self.check_data_label('output/Necking_Point_Output.csv', range_id)

                if not data_label_err_flag:
                    print("Beginning Necking Point")
                    if necking_method == NeckingPointMethod.MINIMUM_DISTANCE:
                        percent_crop_right = float(self.percent_crop_right_entry.get())
                        percent_crop_left = float(self.percent_crop_left_entry.get())
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
                    elif necking_method == NeckingPointMethod.MIDPOINT:
                        binarize_intensity_thresh = int(self.binarize_intensity_thresh_entry.get())
                        bbox_size = int(self.bbox_size_necking_midpt_entry.get())
                        selected_markers, first_frame = tracking.select_markers(cap, bbox_size, self.frame_start) # prompt to select markers
                        if selected_markers.__contains__((-1,-1)): # select_markers returns list of -1 if selections cancelled
                            msg = "Error: Found no markers selected"
                            error_popup(msg)
                        elif len(selected_markers) != 2: 
                            msg = "Error: please ensure exactly 2 markers selected for necking point"
                            error_popup(msg)
                        else:
                            print("Beginning Necking Point (midpoint method)")
                            tracking.necking_point_midpoint(
                                cap,
                                selected_markers,
                                first_frame,
                                bbox_size,
                                self.frame_start,
                                self.frame_end,
                                binarize_intensity_thresh,
                                frame_record_interval,
                                self.frame_interval,
                                self.time_units,
                                file_mode,
                                video_name,
                                range_id
                            )

                    elif necking_method == NeckingPointMethod.STEP_APPROXIMATION:
                        percent_crop_right = float(self.step_percent_crop_right_entry.get())
                        percent_crop_left = float(self.step_percent_crop_left_entry.get())
                        step_length = int(self.step_num_entry.get())
                        print("Beginning Necking Point (Step approximation method)")
                        tracking.necking_point_step_approximation(
                            cap,
                            self.frame_start,
                            self.frame_end,
                            percent_crop_left,
                            percent_crop_right,
                            binarize_intensity_thresh,
                            step_length,
                            frame_record_interval,
                            self.frame_interval,
                            self.time_units,
                            file_mode,
                            video_name,
                            range_id
                        )

                    else:
                        msg = "Error: Please select a necking point method."
                        error_popup(msg)

            case TrackingOperation.AREA:

                # check if range_id already used
                if file_mode == FileMode.APPEND: # only need to check prev ids if appending
                    data_label_err_flag = self.check_data_label('output/Surface_Area_Output.csv', range_id)

                if not data_label_err_flag:
                    bbox_size = int(self.bbox_size_area_entry.get())
                    distance_from_marker_thresh = int(self.distance_from_marker_thresh_entry.get())
                    preprocessing_need = PreprocessingIssue(self.preprocessing_choice_var.get())
                    selected_markers, first_frame = tracking.select_markers(cap, bbox_size, self.frame_start, self.preprocess_vals) # prompt to select markers
                    print(f"marker locs: {selected_markers}")
                    if not selected_markers.__contains__((-1,-1)): # select_markers returns list of -1 if selections cancelled
                        if use_multithread:    
                            tracking.track_area_threaded(
                                cap,
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
                        else:
                            tracking.track_area(
                                cap,
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
                                range_id,
                                preprocessing_need,
                                self.preprocess_vals
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
        
        self.frame_select_label = ttk.Label(self.child_window, textvariable=self.child_label_var, style='Regular.TLabel')
        self.frame_select_label.grid(pady=10)
        
        self.confirm_start_button = ttk.Button(self.child_window, text="Confirm start frame", command=self.confirm_start, style='Regular.TButton')
        self.confirm_start_button.grid()
        self.confirm_end_button = ttk.Button(self.child_window, text="Confirm end frame", command=self.confirm_end, style='Regular.TButton')
        self.confirm_end_button.grid()
        
        self.frame_display = ttk.Label(self.child_window, text="test", style='Regular.TLabel')
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
            frame, _ = tracking.scale_frame(frame, 0.5)
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
    def __init__(self, parent, conversion_factor, conversion_units):
        self.parent = parent
        self.root = parent.root
        self.conversion_factor = conversion_factor
        self.conversion_units = conversion_units
        self.window = tk.Toplevel(self.root)
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
            "Poisson's ratio (from csv)": 'output/poissons_ratio.csv',
            'Marker velocity': ('output/Tracking_Output.csv', 'output/Surface_Area_Output.csv'), # movement can use marker or centroid locators
            'Marker distance': ('output/Tracking_Output.csv', 'output/Surface_Area_Output.csv'),
            'Marker displacement': ('output/Tracking_Output.csv', 'output/Surface_Area_Output.csv'),
            'Surface area': 'output/Surface_Area_Output.csv',
        }

        # analysis_function(self.user_units, self.df, False, self.dataset_index)
        self.function_map = {
            'Longitudinal strain': analysis.analyze_marker_deltas,
            'Radial strain':  analysis.analyze_necking_point,
            "Poisson's ratio (from csv)": analysis.poissons_ratio_csv,
            'Marker velocity':  analysis.marker_movement_analysis,
            'Marker distance': analysis.marker_movement_analysis,
            'Marker displacement': analysis.marker_movement_analysis,
            'Surface area':  analysis.marker_movement_analysis
        }

        # dynamically set figure/window size
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        system_dpi = self.window.winfo_fpixels('1i')  # This fetches the DPI in some environments
        fig_width_in = (screen_width * 0.4) / system_dpi
        fig_height_in = (screen_height * 0.5) / system_dpi

        # Setup for file selection and data label listbox
        file_selection_frame = tk.Frame(self.window)
        analysis_selector_label = ttk.Label(file_selection_frame, text="Select file type:", style='Regular.TLabel')
        analysis_selector_label.grid(row=2, column=0)
        self.analysis_selector = ttk.Combobox(file_selection_frame, values=list(self.output_files.keys()))
        self.analysis_selector.set('Select analysis type')
        self.analysis_selector.grid(row=3, column=0, padx=8, pady=12)

        data_label_selector_label = ttk.Label(file_selection_frame, text="Select data labels:", style='Regular.TLabel')
        data_label_selector_label.grid(row=2, column=1)
        self.data_label_selector = tk.Listbox(file_selection_frame, selectmode='single', exportselection=0, height=4)
        self.data_label_selector.grid(row=3, column=1, padx=8, pady=12)
        
        locator_choice_label = ttk.Label(file_selection_frame, text="Locator choice for:\ndiplacement, distance, and velocity", style='Regular.TLabel')
        locator_choice_label.grid(row=0, column=0, columnspan=2, padx=4, pady=(8,2))
        self.locator_choice_var = tk.StringVar()
        self.locator_choice_var.set(LocatorType.BBOX.value)
        locator_marker_choice_radio = ttk.Radiobutton(file_selection_frame, text='Marker (use marker tracker)', variable=self.locator_choice_var, value=LocatorType.BBOX.value, style='Regular.TRadiobutton')
        locator_marker_choice_radio.grid(row=1, column=0)
        locator_choice_centroid_radio = ttk.Radiobutton(file_selection_frame, text='Centroid (use surface area)', variable=self.locator_choice_var, value=LocatorType.CENTROID.value, style='Regular.TRadiobutton')
        locator_choice_centroid_radio.grid(row=1, column=1, pady=12)

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
        self.csv_file_path_item = self.output_files.get(selected_analysis)
        
        self.locator_type = LocatorType(self.locator_choice_var.get())
        if isinstance(self.csv_file_path_item, tuple): # for applicable analysis functions, choose appropriate input df path based on locator type selection
            self.csv_file_path = self.csv_file_path_item[0] if self.locator_type == LocatorType.BBOX else self.csv_file_path_item[1]
        else: # ensure that centroid not selected for analysis options that can't use centroid locator
            self.csv_file_path = self.csv_file_path_item
            if self.locator_type == LocatorType.CENTROID and selected_analysis != 'Surface area':
                msg = f"Warning: Centroid locator not viable for {selected_analysis},\ndefaulting to data from {self.csv_file_path}"
                warning_popup(msg)
        print(self.csv_file_path)

        # ensure centroid is selected for surface area
        if self.locator_type != LocatorType.CENTROID and selected_analysis == 'Surface area':
            msg = "Warning: Must use centroid locator type for surface area analysis.\nDefaulting to surface area input file"
            warning_popup(msg)

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

    def set_analysis_function_args(self):
        self.function_args_map = {
            'Longitudinal strain': {
                'conversion_factor': self.conversion_factor,
                'conversion_units': self.conversion_units,
                'df': self.df,
                'will_save_figures': False,
                'chosen_video_data': self.dataset_index
            },
            'Radial strain': {
                'conversion_factor': self.conversion_factor,
                'conversion_units': self.conversion_units,
                'df': self.df,
                'will_save_figures': False,
                'chosen_video_data': self.dataset_index
            },
            "Poisson's ratio (from csv)": {
                'df': self.df,
                'will_save_figures': False,
                'chosen_video_data': self.dataset_index
            },
            'Marker velocity': {
                'analysis_type': AnalysisType.VELOCITY,
                'conversion_factor': self.conversion_factor,
                'conversion_units': self.conversion_units,
                'output_df_path': 'output/velocity.csv',
                'output_y_col_name':  f'velocity ({self.conversion_units})',
                'df': self.df,
                'will_save_figures': False,
                'chosen_video_data': self.dataset_index,
                'locator_type': self.locator_type
            },
            'Marker distance': {
                'analysis_type': AnalysisType.DISTANCE,
                'conversion_factor': self.conversion_factor,
                'conversion_units': self.conversion_units,
                'output_df_path': 'output/distance.csv',
                'output_y_col_name':  f'distance ({self.conversion_units})',
                'df': self.df,
                'will_save_figures': False,
                'chosen_video_data': self.dataset_index,
                'locator_type': self.locator_type
            },
            'Marker displacement': {
                'analysis_type': AnalysisType.DISPLACEMENT,
                'conversion_factor': self.conversion_factor,
                'conversion_units': self.conversion_units,
                'output_df_path': 'output/displacement.csv',
                'output_y_col_name':  f'displacement ({self.conversion_units})',
                'df': self.df,
                'will_save_figures': False,
                'chosen_video_data': self.dataset_index,
                'locator_type': self.locator_type
            },
            'Surface area': {
                'analysis_type': AnalysisType.SURFACE_AREA,
                'conversion_factor': self.conversion_factor,
                'conversion_units': self.conversion_units,
                'output_df_path': 'output/surface_area.csv',
                'output_y_col_name':  f'surface_area ({self.conversion_units})',
                'df': self.df,
                'will_save_figures': False,
                'chosen_video_data': self.dataset_index,
                'locator_type': self.locator_type
            }
        }

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

        # define analysis function args
        self.set_analysis_function_args()
        args = self.function_args_map.get(self.analysis_choice)
        self.x_data, self.y_data, self.plot_args, _ = analysis_function(**args)

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
        try:
            indices_to_remove = self.df[(self.df[time_col] == time_point)].index.tolist()
        except KeyError as ke:
            print(ke)
            msg = "Error: No data label selected"
            error_popup(msg)

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
        self.df = pd.read_csv(self.csv_file_path) # reload original dataframe
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
    def __init__(self, parent, conversion_factor, conversion_units):
        self.parent = parent
        self.root = parent.root
        self.conversion_factor = conversion_factor
        self.conversion_units = conversion_units
        self.window = tk.Toplevel(self.root)
        self.window.title("Select a region of a plot")
        self.window.iconphoto(False, tk.PhotoImage(file="ico/m3b_comp.png"))

        self.label_to_dataset = {}

        # file options
        self.output_files = {
            'Longitudinal strain': 'output/Tracking_Output.csv',
            'Radial strain': 'output/Necking_Point_Output.csv',
            "Poisson's ratio (from csv)": 'output/poissons_ratio.csv',
            'Marker velocity': ('output/Tracking_Output.csv', 'output/Surface_Area_Output.csv'), # movement can use marker or centroid locators
            'Marker distance': ('output/Tracking_Output.csv', 'output/Surface_Area_Output.csv'),
            'Marker displacement': ('output/Tracking_Output.csv', 'output/Surface_Area_Output.csv'),
            'Surface area': 'output/Surface_Area_Output.csv',
        }

        # analysis_function(self.user_units, self.df, False, self.dataset_index)
        self.function_map = {
            'Longitudinal strain': analysis.analyze_marker_deltas,
            'Radial strain':  analysis.analyze_necking_point,
            "Poisson's ratio (from csv)": analysis.poissons_ratio_csv,
            'Marker velocity':  analysis.marker_movement_analysis,
            'Marker distance': analysis.marker_movement_analysis,
            'Marker displacement': analysis.marker_movement_analysis,
            'Surface area':  analysis.marker_movement_analysis
        }
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        system_dpi = self.window.winfo_fpixels('1i')  # This fetches the DPI in some environments
        fig_width_in = (screen_width * 0.4) / system_dpi
        fig_height_in = (screen_height * 0.5) / system_dpi

        # Create a combobox widget for user units selection
        data_selection_frame = tk.Frame(self.window)

        locator_choice_label = ttk.Label(data_selection_frame, text="Locator choice for:\ndiplacement, distance, and velocity", style='Regular.TLabel')
        locator_choice_label.grid(row=0, column=0, columnspan=2, padx=4, pady=(8,2))
        self.locator_choice_var = tk.StringVar()
        self.locator_choice_var.set(LocatorType.BBOX.value)
        locator_marker_choice_radio = ttk.Radiobutton(data_selection_frame, text='Marker (use marker tracker)', variable=self.locator_choice_var, value=LocatorType.BBOX.value, style='Regular.TRadiobutton')
        locator_marker_choice_radio.grid(row=1, column=0)
        locator_choice_centroid_radio = ttk.Radiobutton(data_selection_frame, text='Centroid (use surface area)', variable=self.locator_choice_var, value=LocatorType.CENTROID.value, style='Regular.TRadiobutton')
        locator_choice_centroid_radio.grid(row=1, column=1, pady=12)

        analysis_selector_label = ttk.Label(data_selection_frame, text="Select analysis option", style='Regular.TLabel')
        analysis_selector_label.grid(row=2, column=0)
        self.analysis_selector = ttk.Combobox(data_selection_frame, values=list(self.output_files.keys()))
        self.analysis_selector.set('Select analysis option')  # Set default placeholder text
        self.analysis_selector.grid(row=3, column=0, padx=8, pady=12)  # Pack the combobox into the window

        # Create a second combobox for data labels, initially empty
        data_label_selector_label = ttk.Label(data_selection_frame, text="Select data label(s)", style='Regular.TLabel')
        data_label_selector_label.grid(row=2, column=1)
        self.data_label_selector = tk.Listbox(data_selection_frame, selectmode='multiple', exportselection=0, height=5)
        self.data_label_selector.grid(row=3, column=1, padx=8, pady=12)

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
        self.csv_file_path_item = self.output_files.get(selected_analysis)
        
        self.locator_type = LocatorType(self.locator_choice_var.get())
        if isinstance(self.csv_file_path_item, tuple): # for applicable analysis functions, choose appropriate input df path based on locator type selection
            self.csv_file_path = self.csv_file_path_item[0] if self.locator_type == LocatorType.BBOX else self.csv_file_path_item[1]
        else: # ensure that centroid not selected for analysis options that can't use centroid locator
            self.csv_file_path = self.csv_file_path_item
            if self.locator_type == LocatorType.CENTROID and selected_analysis != 'Surface area':
                msg = f"Warning: Centroid locator not viable for {selected_analysis},\ndefaulting to data from {self.csv_file_path}"
                warning_popup(msg)

        if self.csv_file_path:
            self.df = pd.read_csv(self.csv_file_path)
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

    def set_analysis_function_args(self, dataset_index):
        self.function_args_map = {
            'Longitudinal strain': {
                'conversion_factor': self.conversion_factor,
                'conversion_units': self.conversion_units,
                'df': self.df,
                'will_save_figures': False,
                'chosen_video_data': dataset_index
            },
            'Radial strain': {
                'conversion_factor': self.conversion_factor,
                'conversion_units': self.conversion_units,
                'df': self.df,
                'will_save_figures': False,
                'chosen_video_data': dataset_index
            },
            "Poisson's ratio (from csv)": {
                'df': self.df,
                'will_save_figures': False,
                'chosen_video_data': dataset_index
            },
            'Marker velocity': {
                'analysis_type': AnalysisType.VELOCITY,
                'conversion_factor': self.conversion_factor,
                'conversion_units': self.conversion_units,
                'output_df_path': 'output/velocity.csv',
                'output_y_col_name':  f'velocity ({self.conversion_units})',
                'df': self.df,
                'will_save_figures': False,
                'chosen_video_data': dataset_index,
                'locator_type': self.locator_type
            },
            'Marker distance': {
                'analysis_type': AnalysisType.DISTANCE,
                'conversion_factor': self.conversion_factor,
                'conversion_units': self.conversion_units,
                'output_df_path': 'output/distance.csv',
                'output_y_col_name':  f'distance ({self.conversion_units})',
                'df': self.df,
                'will_save_figures': False,
                'chosen_video_data': dataset_index,
                'locator_type': self.locator_type
            },
            'Marker displacement': {
                'analysis_type': AnalysisType.DISPLACEMENT,
                'conversion_factor': self.conversion_factor,
                'conversion_units': self.conversion_units,
                'output_df_path': 'output/displacement.csv',
                'output_y_col_name':  f'displacement ({self.conversion_units})',
                'df': self.df,
                'will_save_figures': False,
                'chosen_video_data': dataset_index,
                'locator_type': self.locator_type
            },
            'Surface area': {
                'analysis_type': AnalysisType.SURFACE_AREA,
                'conversion_factor': self.conversion_factor,
                'conversion_units': self.conversion_units,
                'output_df_path': 'output/surface_area.csv',
                'output_y_col_name':  f'surface_area ({self.conversion_units})',
                'df': self.df,
                'will_save_figures': False,
                'chosen_video_data': dataset_index,
                'locator_type': self.locator_type
            }
        }

    def execute_analysis(self):
        self.analysis_choice = self.analysis_selector.get()
        self.selected_indices = self.data_label_selector.curselection()
        self.selected_labels = [self.data_label_selector.get(i) for i in self.selected_indices]
        self.selected_analysis = self.analysis_selector.get()
        self.analysis_func = self.function_map.get(self.selected_analysis)
        if not self.selected_indices:
            error_popup("No data labels selected.")
            return
        # Handle multiple results
        num_datasets = len(self.selected_labels)
        print(self.selected_labels)
        self.ax.clear()
        self.all_x = []
        self.all_y = []
        for selected_label in self.selected_labels:
            which_dataset = self.label_to_dataset.get(selected_label, 0)  # Retrieve dataset number from label
            if self.analysis_func and which_dataset:
                self.set_analysis_function_args(which_dataset)
                args = self.function_args_map.get(self.analysis_choice)
                result = self.analysis_func(**args)
                if result:
                    x, y, plot_args, _ = result
                    self.x, self.y = x[0], y[0]
                    self.all_x.append(self.x)
                    self.all_y.append(self.y)
                    print(self.x, self.y, plot_args, num_datasets)
                    self.plot_data(plot_args, num_datasets)

    def plot_data(self, plot_args, num_datasets=1):
        with open("plot_opts/plot_customizations.json", 'r') as plot_customs_file:
            plot_customs = json.load(plot_customs_file)
        font = plot_customs['font']

        self.ax.plot(self.x, self.y, 'o', markersize=1, label=plot_args['data_label'][0])

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
        

    def onselect(self, xmin, xmax):
        self.ax.set_xlim(xmin, xmax)
        
        averages = []
        stddevs = []
        for i, label in enumerate(self.selected_labels):
            print(self.all_x[i], '\n*****', xmin, xmax)
            x, y = self.all_x[i], self.all_y[i]
            idxs_in_range = (x >= xmin) & (x <= xmax)  
            print(y, idxs_in_range)  
            y_in_range = np.asarray(y)[idxs_in_range]
            averages.append(np.mean(y_in_range))
            stddevs.append(np.std(y_in_range))
        #print(np.concatenate(self.all_y), '\n', np.asarray(self.all_y).shape, '\n', self.all_y)
        df = pd.DataFrame({'Averages': averages, 'StdDev': stddevs}, index=self.selected_labels)
        global_series = pd.Series({'Averages': np.mean(np.concatenate(self.all_y).flatten()), 'StdDev': np.std(np.concatenate(self.all_y).flatten())}, name='Global')
        df = pd.concat([df, global_series], axis=1)

        df.to_csv(f"output/data_selector/data_selector_stats_{self.selected_analysis}.csv")

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


class Boxplotter:
    def __init__(self, parent, conversion_factor, conversion_units):
        self.parent = parent
        self.root = parent.root
        self.conversion_factor = conversion_factor
        self.conversion_units = conversion_units
        self.window = tk.Toplevel(self.root)
        self.window.title("Select a region of a plot")
        self.window.iconphoto(False, tk.PhotoImage(file="ico/m3b_comp.png"))

        self.label_to_dataset = {}
        self.output_files = {
            "Poisson's ratio (from csv)": 'output/poissons_ratio.csv',
            'Marker velocity': 'output/velocity.csv',
            'Marker displacement': 'output/displacement.csv',
            'Marker distance': 'output/distance.csv',
            'Surface area': 'output/surface_area.csv'
        }

        warning_label = ttk.Label(self.window, text="Please ensure you have ran the preliminary analysis in the main UI before using this tool", font=('TkDefaultFont', 12, 'bold'))
        warning_label.grid(row=0, column=0, padx=8, pady=(12,24))

        # combo box to choose analysis/file type
        selection_frame = tk.Frame(self.window)
        analysis_selector_label = ttk.Label(selection_frame, text="Select analysis option", style='Regular.TLabel')
        analysis_selector_label.grid(row=0, column=0)
        self.analysis_selector = ttk.Combobox(selection_frame, values=list(self.output_files.keys()))
        self.analysis_selector.set('Select analysis option')  # Set default placeholder text
        self.analysis_selector.grid(row=1, column=0, padx=8, pady=12)  # Pack the combobox into the window

        # Create a listbox for data labels, initially empty
        data_label_selector_label = ttk.Label(selection_frame, text="Select data label(s)", style='Regular.TLabel')
        data_label_selector_label.grid(row=0, column=1)
        self.data_label_selector = tk.Listbox(selection_frame, selectmode='multiple', exportselection=0, height=5)
        self.data_label_selector.grid(row=1, column=1, padx=8, pady=12)

        selection_frame.grid(row=1, column=0)
        self.analysis_selector.bind('<<ComboboxSelected>>', self.update_data_label_selector)

        # Add Radio Buttons for Selection Method
        self.grouping_var = tk.StringVar()
        grouping_frame = tk.Frame(self.window)
        grouping_label = ttk.Label(grouping_frame, text="Select grouping method:", style='Regular.TLabel')
        grouping_label.grid(row=0, column=0, padx=8, pady=8)

        conditions_radio = ttk.Radiobutton(grouping_frame, text="By Conditions", variable=self.grouping_var, value="conditions", style='Regular.TRadiobutton')
        conditions_radio.grid(row=1, column=0, padx=8)
        time_radio = ttk.Radiobutton(grouping_frame, text="By Time Points", variable=self.grouping_var, value="time_points", style='Regular.TRadiobutton')
        time_radio.grid(row=2, column=0, padx=8)

        grouping_frame.grid(row=2, column=0, padx=8, pady=12)
        self.grouping_var.trace_add("write", self.update_grouping_method_ui)

        self.time_ranges = []
        self.condition_to_label = {}

        # Create the "Go" button to execute analysis
        self.go_button = ttk.Button(self.window, text="Go", command=self.execute_analysis)
        self.go_button.grid(row=20, column=0, padx=8, pady=20, sticky="ew")

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

    def setup_conditions_ui(self):
        self.condition_frame = tk.Frame(self.window)
        condition_label = ttk.Label(self.condition_frame, text="Enter Condition Name:", style='Regular.TLabel')
        condition_label.grid(row=0, column=0, padx=8)
        condition_name_entry = ttk.Entry(self.condition_frame, style='Regular.TEntry')
        condition_name_entry.grid(row=0, column=1, padx=8)
        condition_instr_label = ttk.Label(self.condition_frame, text="Select from data labels above that are associated with this condition\nThese labels will be analyzed and grouped into a box in the boxplot", style='Regular.TLabel')
        condition_instr_label.grid(row=1, column=0, columnspan=2)

        add_condition_button = ttk.Button(self.condition_frame, text="Add Condition",
                                        command=lambda: self.add_condition(condition_name_entry.get(), self.data_label_selector.curselection()))
        add_condition_button.grid(row=2, column=0, columnspan=2, pady=8)

        remove_condition_button = ttk.Button(self.condition_frame, text="Remove Selected",
                                        command=self.remove_selected_condition)
        remove_condition_button.grid(row=3, column=0, columnspan=2, pady=8)

        label_instr = ttk.Label(self.condition_frame, text="Each condition will be its own box in the plot, and each label will become a point in that box\nIf you want to group by data time ranges, use 'By Time Points'", style='Regular.TLabel')
        label_instr.grid(row=5, column=0, columnspan=2)

        self.condition_listbox = tk.Listbox(self.condition_frame, height=5)
        self.condition_listbox.grid(row=4, column=0, columnspan=2, padx=8, pady=12)
        self.condition_frame.grid(row=4, column=0)

    def add_condition(self, condition_name, selected_indices):
        selected_labels = [self.data_label_selector.get(i) for i in selected_indices]
        self.condition_to_label[condition_name] = selected_labels
        self.condition_listbox.insert(tk.END, f"{condition_name}: {', '.join(selected_labels)}")

    def remove_selected_condition(self):
        selected_index = self.condition_listbox.curselection()
        if selected_index:
            # Remove the selected condition from the listbox
            condition_name = self.condition_listbox.get(selected_index)
            self.condition_listbox.delete(selected_index)
            
            # Remove the corresponding condition from the dictionary
            for condition, labels in list(self.condition_to_label.items()):
                if f"{condition}: {', '.join(labels)}" == condition_name:
                    del self.condition_to_label[condition]

    def setup_time_ranges_ui(self):
        self.time_frame = tk.Frame(self.window)
        _, _, self.time_units = analysis.get_time_labels(self.df)
        entry_instr_label = ttk.Label(self.time_frame, text=f"Indicate time range of each box\nCurrent units specified in main UI: ({self.time_units})", style='Regular.TLabel')
        entry_instr_label.grid(row=0, column=0, columnspan=2)
        t0_label = ttk.Label(self.time_frame, text="Start Time (t0):", style='Regular.TLabel')
        t0_label.grid(row=2, column=0, padx=8)
        t0_entry = ttk.Entry(self.time_frame, style='Regular.TEntry')
        t0_entry.grid(row=2, column=1, padx=8)

        tf_label = ttk.Label(self.time_frame, text="End Time (tf):", style='Regular.TLabel')
        tf_label.grid(row=3, column=0, padx=8)
        tf_entry = ttk.Entry(self.time_frame, style='Regular.TEntry')
        tf_entry.grid(row=3, column=1, padx=8)

        add_time_button = ttk.Button(self.time_frame, text="Add Time Range",
                                    command=lambda: self.add_time_range(t0_entry.get(), tf_entry.get()))
        add_time_button.grid(row=4, column=0, columnspan=2, pady=8)

        remove_time_button = ttk.Button(self.time_frame, text="Remove Selected", command=self.remove_selected_time_range)
        remove_time_button.grid(row=5, column=0, columnspan=2, pady=8)

        time_range_label_instr = ttk.Label(self.time_frame, text="All labels selected will be analyzed and averaged for each time range\nIf you want to group by data labels, use 'By Conditions'", style='Regular.TLabel')
        time_range_label_instr.grid(row=6, column=0, columnspan=2)

        self.time_listbox = tk.Listbox(self.time_frame, height=5)
        self.time_listbox.grid(row=7, column=0, columnspan=2, padx=8, pady=12)
        self.time_frame.grid(row=5, column=0)

    def add_time_range(self, t0, tf):
        self.time_ranges.append((t0, tf))
        self.time_listbox.insert(tk.END, f"({t0}, {tf})")

    def remove_selected_time_range(self):
        selected_index = self.time_listbox.curselection()
        if selected_index:
            self.time_listbox.delete(selected_index)
            selected_time_range = self.time_ranges[selected_index[0]]
            self.time_ranges.remove(selected_time_range)

    def update_grouping_method_ui(self, *args):
        # check if dataframe specified first
        if not hasattr(self, 'df'):
            msg = "Error: please specify analysis option first"
            error_popup(msg)
            return 
        
        # Forget any existing frames first to avoid overlap
        if hasattr(self, 'condition_frame'):
            self.condition_frame.grid_forget()
        if hasattr(self, 'time_frame'):
            self.time_frame.grid_forget()

        # Check the value of grouping_var and update the UI accordingly
        if self.grouping_var.get() == "conditions":
            self.setup_conditions_ui()
        elif self.grouping_var.get() == "time_points":
            self.setup_time_ranges_ui()

    def execute_analysis(self):
        # Check which grouping method is selected and call the corresponding function
        if self.grouping_var.get() == "conditions":
            analysis.boxplot_conditions(self.df, self.condition_to_label, self.conversion_units)
        elif self.grouping_var.get() == "time_points":
            selected_labels = [self.data_label_selector.get(i) for i in self.data_label_selector.curselection()]
            if not selected_labels:
                msg = "Warning: No data labels are selected for analysis"
                warning_popup(msg)
                return
            analysis.boxplot_time_ranges(self.df, self.time_ranges, selected_labels, self.conversion_units)
        else:
            print("No valid analysis type selected")


class CropAndCompressVideo:
    def __init__(self, parent, file_label_var):
        self.root = parent.root
        self.parent = parent
        self.file_label_var = file_label_var
        self.video_path = self.parent.video_path
        self.roi = None
        self.new_video_path = None
        self.roi = None
        self.new_video_path = None
        
        # Create the main window
        self.window = tk.Toplevel(self.root)
        self.window.title("Compress Video")
        self.window.iconphoto(False, tk.PhotoImage(file="ico/m3b_comp.png"))
        
        # Frame for crop button and labels
        self.crop_frame = ttk.Frame(self.window)
        self.crop_frame.grid(row=0, column=0, sticky="we", padx=10, pady=(10, 0))

        # Crop labels
        crop_label = ttk.Label(self.crop_frame, text="Opens first frame to select region of interest", style='Regular.TLabel')
        crop_label.grid(row=0, column=0, sticky="w")

        crop_label_details = ttk.Label(self.crop_frame, text="Press enter to confirm crop dimensions or ESC to cancel", style='Regular.TLabel')
        crop_label_details.grid(row=1, column=0, sticky="w")

        # Button to crop the video
        self.crop_button = ttk.Button(self.crop_frame, text="Crop Video", command=self.crop_video)
        self.crop_button.grid(row=2, column=0, pady=16)

        # dimensions labels
        self.original_label = ttk.Label(self.crop_frame, text="Original Dimensions: ", style='Regular.TLabel')
        self.original_label.grid(row=3, column=0, sticky="w")
        self.new_label = ttk.Label(self.crop_frame, text="New Dimensions: ", style='Regular.TLabel')
        self.new_label.grid(row=4, column=0, sticky="w", pady=(0, 10))

        # compression options
        self.compression_frame = ttk.Frame(self.window)
        self.compress_var = tk.BooleanVar(value=False)  # Default value is False
        self.compress_checkbox = ttk.Checkbutton(self.window, text="Compress Video", variable=self.compress_var, command=self.toggle_compress_sliders, style='Regular.TCheckbutton')
        self.compress_checkbox.grid(row=2, column=0, pady=(16,0))

        # Quality slider
        self.quality_label = ttk.Label(self.compression_frame, text="Quality:", style='Regular.TLabel')
        self.quality_label.grid(row=1, column=0, sticky="w")
        self.quality_value_label = ttk.Label(self.compression_frame, text="1.0", style='Regular.TLabel')
        self.quality_scale = ttk.Scale(self.compression_frame, from_=0, to=1, orient="horizontal", command=self.update_quality_label)
        self.quality_scale.set(1.0)  # Default value
        self.quality_scale.grid(row=1, column=1, sticky="we", padx=(0, 10))
        self.quality_value_label.grid(row=1, column=2, sticky="w")

        # Resolution scale factor slider
        self.resolution_label = ttk.Label(self.compression_frame, text="Resolution Scale Factor:", style='Regular.TLabel')
        self.resolution_label.grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.resolution_value_label = ttk.Label(self.compression_frame, text="1.0", style='Regular.TLabel')
        self.resolution_scale = ttk.Scale(self.compression_frame, from_=0, to=1, orient="horizontal", command=self.update_resolution_label)
        self.resolution_scale.set(1.0)  # Default value
        self.resolution_scale.grid(row=2, column=1, sticky="we", padx=(0, 10), pady=(10, 0))
        self.resolution_value_label.grid(row=2, column=2, sticky="w", pady=(10, 0))

        # Button to save cropped video
        self.save_button = ttk.Button(self.window, text="Save Video", command=self.save_video)
        self.save_button.grid(row=5, column=0, columnspan=2, pady=(20, 10))

        # Progress bar
        self.progress_bar = ttk.Progressbar(self.window, length=200, mode='determinate')
        self.progress_bar.grid(row=6, column=0, columnspan=2, pady=10, padx=20)

        # Label to indicate completion of saving process
        self.complete_label = ttk.Label(self.window, text="Video Saved!", style='Regular.TLabel')
        self.complete_label.grid(row=7, column=0, columnspan=2, pady=(10, 20))

    def toggle_compress_sliders(self):
        if self.compress_var.get():
            self.compression_frame.grid(row=3, column=0, columnspan=2, sticky="we", padx=10, pady=(10, 0))
        else:
            self.compression_frame.grid_forget()

    def update_quality_label(self, value):
        self.quality_value_label.config(text=f"{float(value):.2f}")
        
    def update_resolution_label(self, value):
        self.resolution_value_label.config(text=f"{float(value):.2f}")

    def crop_video(self):
        # Open the video file
        cap = cv2.VideoCapture(self.video_path)
        
        # Read the first frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read video")
            return
        
        # Display the frame
        cv2.imshow("Select Region", frame)
        
        # Select region using mouse click
        self.roi = cv2.selectROI("Select Region", frame, fromCenter=False, showCrosshair=True)
        
        # Destroy OpenCV window and release video capture
        cv2.destroyAllWindows()
        cap.release()
        
        # Save the entire video with the cropped selection
        original_dims, new_dims = self.get_cropped_dimensions()
        
        # Update labels with dimensions
        self.original_label.config(text=f"Original Dimensions: {original_dims[0]}x{original_dims[1]}")
        self.new_label.config(text=f"New Dimensions: {new_dims[0]}x{new_dims[1]}")
    
    def get_cropped_dimensions(self):
        # Open the video file again
        cap = cv2.VideoCapture(self.video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # New dimensions after cropping
        if self.roi is None:
            new_width, new_height = width, height
        else:
            new_width = int(self.roi[2]) if int(self.roi[2]) != 0 else width
            new_height = int(self.roi[3]) if int(self.roi[3]) != 0 else height
            
        # Return original and new dimensions
        return (width, height), (new_width, new_height)
    
    def save_video(self):
        quality = float(self.quality_scale.get())
        resolution_factor = float(self.resolution_scale.get())
        
        # Check if compression is enabled
        compress_video = self.compress_var.get()
        
        # Save the entire video with the cropped selection
        _, new_dims = self.get_cropped_dimensions()
        
        # Open the video file again
        cap = cv2.VideoCapture(self.video_path)
        
        # Get the directory and filename of the original video
        dir_name, file_name = os.path.split(self.video_path)
        
        # Create a new filename for the cropped video with "CROPPED_" prepended
        cropped_file_name = f"CROP-COMP_{file_name}"
        # Change file extension to .mkv
        cropped_file_name = os.path.splitext(cropped_file_name)[0] + '.mkv'
        self.new_video_path = os.path.join(dir_name, cropped_file_name)
        
        # Determine new resolution based on scale factor but maintaining original aspect ratio
        w, h = new_dims
        aspect_ratio = w / h

        if aspect_ratio >= 1:  # Landscape orientation
            new_width = int(w * resolution_factor)
            new_height = int(new_width / aspect_ratio)
        else:  # Portrait orientation
            new_height = int(h * resolution_factor)
            new_width = int(new_height * aspect_ratio)

        # Initialize video writer
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(self.new_video_path, fourcc, fps, (new_width, new_height))
        
        # Read frames, crop, compress, and save
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Crop frame
            if self.roi:
                x, y, w, h = self.roi
                cropped_frame = frame[y:y+h, x:x+w]
            else:
                cropped_frame = frame
            
            if compress_video:
                # Compress frame with quality parameter
                encoded_frame = cv2.imencode('.jpg', cropped_frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality * 100)])[1]
                decoded_frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
            else:
                decoded_frame = cropped_frame

            # Resize frame to new dimensions
            resized_frame = cv2.resize(decoded_frame, (new_width, new_height))
            
            # Write frame
            out.write(resized_frame)

            # Update progress bar
            frame_count += 1
            progress = int(frame_count * 100 / cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.progress_bar["value"] = progress
            self.progress_bar.update()
            
        # Release video capture and writer objects
        cap.release()
        out.release()
        
        print(f"Video saved: {self.new_video_path}")
        
        # Set the parent class's label variable with the file path of the new video
        self.file_label_var.set(os.path.basename(self.new_video_path))
        self.parent.video_path = self.new_video_path
        print(self.parent.video_path, self.new_video_path)
        
        # Show completion message
        self.complete_label.grid(row=50, column=0, pady=8)
        self.window.after(5000, lambda: self.complete_label.grid_forget())

class FramePreprocessor:
    """
    Details:
        - Initializes a window where the user can adjust preprocessing parameters.
        - Provides checkboxes and sliders for adjusting sharpness, contrast, and brightness.
        - Allows the user to toggle each preprocessing option and adjust its strength using sliders.
        - Loads live preview in-window to show changes

    Args:
        - self: 
        - parent: 
        - video_path: contains the path to the selected video
        - prev_preprocess_vals: preprocessing values the parent had beforehand

    returns:
        - A dictionary with these values:
            - sharpness: -100 to 100. If 0, no change made
            - contrast: 1 to 100. If 0, no change made
            - brightness: -100 to 100. If 0, no change made
    """

    def __init__(self, parent, video_path, prev_preprocess_vals=None):
        self.root = parent.root
        self.parent = parent
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)

        # read first frame
        self.ret, self.first_frame = self.cap.read()
        if not self.ret:
            raise ValueError("Failed to read the video file.")
        
        # shrink and greyscale first frame, then copy for modded frame
        self.first_frame, _ = tracking.scale_frame(cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2GRAY), .9)
        self.modded_frame = self.first_frame.copy()

        # create main window
        self.window = tk.Toplevel(self.root)
        self.window.title("Preprocess Video")
        self.window.iconphoto(False, tk.PhotoImage(file="ico/m3b_comp.png"))

        # Create the basic options frame
        self.basic_options_frame = ttk.Frame(self.window)
        self.basic_options_frame.grid(row=1, column=0, columnspan=10, sticky="w", padx=10, pady=5)

        # Create the advanced options frame
        self.advanced_options_frame = ttk.Frame(self.window)
        self.advanced_options_frame.grid(row=2, column=0, columnspan=10, sticky="w", padx=10, pady=5)
        self.advanced_options_frame.grid_remove()  # Hide the advanced options frame initially
        self.advanced_options_visible = False

        msg = ("In this window you can select various processing techniques and the strength at which they are used\n\n"
               "Warning: Surface area tracking already performs several fine tuned\n"
               "preprocessing techniques that are not displayed to the user when tracking\n"
               "Proceed with caution if adjusting for surface area tracking")

        self.instructions_and_warning_label = ttk.Label(self.window, text=msg, style='Regular.TLabel')
        self.instructions_and_warning_label.grid(row=0, column=0, columnspan=10, padx=10, pady=10)

        # define checkbox variables
        self.sharpness_var = tk.BooleanVar()
        self.contrast_var = tk.BooleanVar()
        self.brightness_var = tk.BooleanVar()
        self.smoothness_var = tk.BooleanVar()
        self.binarize_var = tk.BooleanVar()
        self.denoise_sp_var = tk.BooleanVar()

        # create slider/checkbox pair
        self.sliders = {}
        self.create_checkbox_with_slider("Blur/Sharpness", self.sharpness_var, 1, -100, 100, self.basic_options_frame)
        self.create_checkbox_with_slider("Contrast", self.contrast_var, 2, 1, 100, self.basic_options_frame)
        self.create_checkbox_with_slider("Brightness", self.brightness_var, 3, -100, 100, self.basic_options_frame)

        # Create "Show Advanced" button
        self.show_advanced_button = tk.Button(self.window, text="Show Custom Options - Advanced", command=self.toggle_advanced_options)
        self.show_advanced_button.grid(row=4, column=0, sticky=tk.W, padx=(110, 0))

        # smoothness options
        self.create_checkbox_with_slider("Smoothness", self.smoothness_var, 5, 1, 100, self.advanced_options_frame)

        # create binarize checkbox
        checkbox = ttk.Checkbutton(self.advanced_options_frame, text="Binarize", variable=self.binarize_var, command=self.update_preview)
        checkbox.grid(row=6, column=0, sticky=tk.W, padx=(110, 0))

        # create denoise sp checkbox
        checkbox = ttk.Checkbutton(self.advanced_options_frame, text="Denoise: S&P", variable=self.denoise_sp_var, command=self.update_preview)
        checkbox.grid(row=7, column=0, sticky=tk.W, padx=(110, 0))

        # Add Save and Load buttons
        save_button = ttk.Button(self.window, text="Save Options", command=self.save_options)
        save_button.grid(row=8, column=0, padx=5, pady=5)

        load_button = ttk.Button(self.window, text="Load Options", command=self.load_options)
        load_button.grid(row=8, column=1, padx=5, pady=5)

        # display window
        self.preview_label = ttk.Label(self.window)
        self.preview_label.grid(row=9, column=0, columnspan=6, pady=10)

        # Set values if prev_preprocess_vals is provided
        if prev_preprocess_vals:
            self.set_preprocess_values(prev_preprocess_vals)

        self.update_preview()

        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def save_options(self):
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if filename:
            self.save_preprocess_options(filename)

    def load_options(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if filename:
            self.load_preprocess_options(filename)

    def save_preprocess_options(self, filename):
        preprocess_vals = {
            "Blur/Sharpness": self.sliders["Blur/Sharpness"].get() if self.sharpness_var.get() else 0,
            "Contrast": self.sliders["Contrast"].get() if self.contrast_var.get() else 0,
            "Brightness": self.sliders["Brightness"].get() if self.brightness_var.get() else 0,
            "Smoothness": self.sliders["Smoothness"].get() if self.smoothness_var.get() else 0,
            "Binarize": self.binarize_var.get(),
            "Denoise SP": self.denoise_sp_var.get()
        }
        with open(filename, 'w') as f:
            json.dump(preprocess_vals, f)

    def load_preprocess_options(self, filename):
        with open(filename, 'r') as f:
            preprocess_vals = json.load(f)
        self.set_preprocess_values(preprocess_vals)
        self.update_preview()

    def set_preprocess_values(self, preprocess_vals):
        self.sharpness_var.set(preprocess_vals["Blur/Sharpness"] != 0)
        self.contrast_var.set(preprocess_vals["Contrast"] != 0)
        self.brightness_var.set(preprocess_vals["Brightness"] != 0)
        self.smoothness_var.set(preprocess_vals["Smoothness"] != 0)
        self.binarize_var.set(preprocess_vals["Binarize"])
        self.denoise_sp_var.set(preprocess_vals["Denoise SP"])
        self.sliders["Blur/Sharpness"].set(preprocess_vals["Blur/Sharpness"])
        self.sliders["Contrast"].set(preprocess_vals["Contrast"])
        self.sliders["Brightness"].set(preprocess_vals["Brightness"])
        self.sliders["Smoothness"].set(preprocess_vals["Smoothness"])

    def create_checkbox_with_slider(self, text, variable, row, min_val, max_val, parent_frame):
        checkbox = ttk.Checkbutton(parent_frame, text=text, variable=variable)
        checkbox.grid(row=row, column=0, sticky=tk.W, padx=(110, 0))

        min_label = ttk.Label(parent_frame, text=f"{min_val}")
        min_label.grid(row=row, column=1, sticky="E", padx=(110, 0))

        slider = ttk.Scale(parent_frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL)
        slider.grid(row=row, column=2, padx=10, pady=5, sticky="ew")

        max_label = ttk.Label(parent_frame, text=f"{max_val}")
        max_label.grid(row=row, column=3)

        value_label = ttk.Label(parent_frame, text=f"Value: {int(slider.get())}", width=10)
        value_label.grid(row=row, column=4, padx=(10, 50))

        slider.config(command=lambda value, var=value_label: self.on_slider_change(value, var, variable))

        self.sliders[text] = slider

        slider.grid_remove()
        min_label.grid_remove()
        max_label.grid_remove()
        value_label.grid_remove()

        variable.trace_add("write", lambda *args: self.toggle_slider(slider, variable, min_label, max_label, value_label, parent_frame))

    def on_slider_change(self, value, value_label, variable):
        value_label.config(text=f"Value: {int(float(value))}")
        self.update_preview()

    def toggle_slider(self, slider, variable, min_label, max_label, value_label, parent_frame):
        if variable.get():
            slider.grid()
            min_label.grid()
            max_label.grid()
            value_label.grid()
        else:
            slider.grid_remove()
            min_label.grid_remove()
            max_label.grid_remove()
            value_label.grid_remove()
            
        # Update the layout of the parent frame
        parent_frame.update_idletasks()
        self.update_preview()

    def toggle_advanced_options(self):
        if self.advanced_options_visible:
            self.advanced_options_frame.grid_forget()
            self.advanced_options_visible = False
            self.show_advanced_button.config(text="Show Advanced")
        else:
            self.advanced_options_frame.grid(row=6, column=0, columnspan=6, pady=10, sticky="w")
            self.advanced_options_visible = True
            self.show_advanced_button.config(text="Hide Advanced")

    def update_preview(self):
        if self.modded_frame is not None and self.modded_frame.size > 0:
            self.modded_frame = tracking.preprocess_frame(
                self.first_frame, self.get_preprocess_vals(), True
            )
            frame = cv2.cvtColor(self.modded_frame, cv2.COLOR_GRAY2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.preview_label.imgtk = imgtk
            self.preview_label.configure(image=imgtk)
        
        self.window.lift()

    def get_preprocess_vals(self):
        returnDict = {
            "Blur/Sharpness": self.sliders["Blur/Sharpness"].get() if self.sharpness_var.get() else 0,
            "Contrast": self.sliders["Contrast"].get() if self.contrast_var.get() else 0,
            "Brightness": self.sliders["Brightness"].get() if self.brightness_var.get() else 0,
            "Smoothness": self.sliders["Smoothness"].get() if self.smoothness_var.get() else 0,
            "Binarize": self.binarize_var.get(),
            "Denoise SP": self.denoise_sp_var.get()
        }
        return returnDict

    def on_close(self):
        self.parent.preprocess_vals = self.get_preprocess_vals()
        self.cap.release()
        self.window.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    window = TrackingUI(root)
    root.mainloop()
