import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

# Load the CSV data
file_path = 'output/Tracking_Output.csv'  # Update the path to your CSV file
data = pd.read_csv(file_path)

# Calculate cumulative displacements for tracker 1
data['1-cum_x_displacement'] = data['1-x (px)'].diff().abs().fillna(0).cumsum()
data['1-cum_y_displacement'] = data['1-y (px)'].diff().abs().fillna(0).cumsum()

# Calculate cumulative displacements for tracker 2
data['2-cum_x_displacement'] = data['2-x (px)'].diff().abs().fillna(0).cumsum()
data['2-cum_y_displacement'] = data['2-y (px)'].diff().abs().fillna(0).cumsum()

# Plotting
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Tracker 1 Cumulative Displacement (px)', color=color)
line1, = ax1.plot(data['1-Time(s)'], data['1-cum_x_displacement'], label='Tracker 1 Cumulative X Displacement', color='red')
line2, = ax1.plot(data['1-Time(s)'], data['1-cum_y_displacement'], label='Tracker 1 Cumulative Y Displacement', color='darkred')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Tracker 2 Cumulative Displacement (px)', color=color)  # we already handled the x-label with ax1
line3, = ax2.plot(data['2-Time(s)'], data['2-cum_x_displacement'], label='Tracker 2 Cumulative X Displacement', color='blue')
line4, = ax2.plot(data['2-Time(s)'], data['2-cum_y_displacement'], label='Tracker 2 Cumulative Y Displacement', color='darkblue')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Tracker Cumulative Displacements Over Time')
fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

# Define the SpanSelector callback
def onselect(xmin, xmax):
    ax1.set_xlim(xmin, xmax)
    ax2.set_xlim(xmin, xmax)
    fig.canvas.draw()

# Create SpanSelector
span = SpanSelector(ax1, onselect, 'horizontal', useblit=True, props=dict(alpha=0.5, facecolor='red'))

# Define the reset function
def reset(event):
    if event.key == 'escape':
        ax1.set_xlim(data['1-Time(s)'].min(), data['1-Time(s)'].max())
        ax2.set_xlim(data['1-Time(s)'].min(), data['1-Time(s)'].max())
        fig.canvas.draw()

# Connect the reset function to the key press event
fig.canvas.mpl_connect('key_press_event', reset)

plt.show()
