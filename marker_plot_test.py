import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import cv2
import sys

class TrackerPlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Distance Between Markers Over Time")
        
        # Create a frame for the plot
        self.frame = ttk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Initialize video capture (example, replace with actual video path if needed)
        self.cap = cv2.VideoCapture(0)  # Using the default camera, replace with your video file

        # Bind the on_close method to the window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Load and plot the data
        self.plot_marker_distance()

    def plot_marker_distance(self):
        # Load the CSV file into a DataFrame
        file_path = 'output/Tracking_Output.csv'
        df = pd.read_csv(file_path)

        # Display the column names to identify any discrepancies
        print("Column names:", df.columns)

        # Clean the column names by stripping leading/trailing whitespace
        df.columns = df.columns.str.strip()

        # Display the column names after cleaning
        print("Column names after cleaning:", df.columns)

        # Extract unique frame times
        frame_times = df['1-Time(s)'].unique()
        print("Frame times:", frame_times)

        # Initialize a figure for plotting
        fig, ax = plt.subplots(figsize=(12, 6))

        # Specify the pairs of trackers to calculate distance between
        tracker_pairs = [(1, 2)]  # Add more pairs if needed

        for pair in tracker_pairs:
            tracker1, tracker2 = pair
            
            # Extract data for each tracker
            tracker1_data = df[df['1-Tracker'] == tracker1].copy()
            tracker2_data = df[df['1-Tracker'] == tracker2].copy()

            # Ensure both trackers have data for the same frames
            common_frames = tracker1_data['1-Frame'].isin(tracker2_data['1-Frame'])
            tracker1_data = tracker1_data[common_frames]
            tracker2_data = tracker2_data[tracker2_data['1-Frame'].isin(tracker1_data['1-Frame'])]

            print(f"Tracker {tracker1} data:\n", tracker1_data.head())
            print(f"Tracker {tracker2} data:\n", tracker2_data.head())

            # Calculate the distance between the two trackers
            distances = np.sqrt(
                (tracker1_data['1-x (px)'].values - tracker2_data['1-x (px)'].values) ** 2 +
                (tracker1_data['1-y (px)'].values - tracker2_data['1-y (px)'].values) ** 2
            )

            print(f"Distances between Tracker {tracker1} and Tracker {tracker2}:", distances)

            # Plot the distance over time
            ax.plot(tracker1_data['1-Time(s)'], distances, marker='o', linestyle='-', label=f'Distance between Tracker {tracker1} and {tracker2}')

        ax.set_title('Distance Between Markers Over Time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance (px)')
        ax.legend()
        ax.grid(True)

        # Create a canvas to display the plot
        canvas = FigureCanvasTkAgg(fig, master=self.frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_close(self):
        # Release the video capture object if it exists
        if self.cap.isOpened():
            self.cap.release()
        # Destroy the Tkinter window
        self.root.destroy()
        # Exit the program
        sys.exit()

# Create the Tkinter application
root = tk.Tk()
app = TrackerPlotApp(root)
root.mainloop()
