import cv2
import numpy as np

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current_marker = [(x, y)]
        marker_positions.append(current_marker)
        cv2.circle(first_frame, current_marker[0], 5, (0, 255, 0), -1)
        cv2.imshow('Select Markers', first_frame)

def select_markers(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Get the first frame
    ret, first_frame = cap.read()

    # Create a window to display the first frame
    cv2.imshow('Select Markers', first_frame)

    # Set the mouse callback function
    cv2.setMouseCallback('Select Markers', mouse_callback)

    # Wait until the user finishes selecting markers
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(marker_positions) == 3:
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()

    # Print the selected marker positions
    print("Selected Marker Positions:")
    for i, marker in enumerate(marker_positions):
        print(f"Marker {i + 1}: {marker}")

    return marker_positions

def main():
    video_path = 'videos/2024-01-23_Test video_DI-01232024120123.avi'
    selected_markers = select_markers(video_path)

    # Use selected_markers for further processing (e.g., tracking)

if __name__ == "__main__":
    marker_positions = []
    main()
