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
        self.root.title("Tracker Displacement Over Time")
        
        # Create a frame for the plot
        self.frame = ttk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Bind the on_close method to the window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Load and plot the data
        self.plot_tracker_displacement()

    def plot_tracker_displacement(self):
        # Load the CSV file into a DataFrame
        file_path = 'output/Tracking_Output.csv'
        df = pd.read_csv(file_path)

        # Display the column names to identify any discrepancies
        print("Column names:", df.columns)

        # Clean the column names by stripping leading/trailing whitespace
        df.columns = df.columns.str.strip()

        # Display the column names after cleaning
        print("Column names after cleaning:", df.columns)

        # Extract unique tracker IDs
        trackers = df['1-Tracker'].unique()

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the displacement of each tracker over time
        for tracker in trackers:
            tracker_data = df[df['1-Tracker'] == tracker].copy()  # Use .copy() to avoid SettingWithCopyWarning
            
            # Calculate displacement
            x0, y0 = tracker_data.iloc[0][['1-x (px)', '1-y (px)']]
            tracker_data.loc[:, 'Displacement'] = np.sqrt((tracker_data['1-x (px)'] - x0)**2 + (tracker_data['1-y (px)'] - y0)**2)
            
            ax.plot(tracker_data['1-Time(s)'], tracker_data['Displacement'], marker='o', linestyle='-', label=f'Tracker {tracker}')

        ax.set_title('Tracker Displacement Over Time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Displacement (px)')
        ax.legend()
        ax.grid(True)

        # Create a canvas to display the plot
        canvas = FigureCanvasTkAgg(fig, master=self.frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_close(self):
        # Destroy the Tkinter window
        self.root.destroy()
        sys.exit()

# Create the Tkinter application
root = tk.Tk()
app = TrackerPlotApp(root)
root.mainloop()
