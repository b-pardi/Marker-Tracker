import cv2
import sys

"""TODO
- track the markers
- when selecting markers, ability to right click and cancel previous selection
- wtf do i do with the tracking information
"""

def mouse_callback(event, x, y, flags, params):
    first_frame = params['first_frame']
    
    if event == cv2.EVENT_LBUTTONDOWN: # on left click
        cur_marker = [(x,y)]
        params['marker_positions'].append(cur_marker)
        cv2.circle(first_frame, cur_marker[0], 5, (255, 255, 0), -1) # draw circle where clicked
        cv2.imshow('Select Markers', first_frame)

def select_markers(video_path):
    cap = cv2.VideoCapture(video_path) # load video
    ret, first_frame = cap.read() # get first frame for selection
    cv2.imshow('Select Markers', first_frame) # show first frame

    mouse_params = {"first_frame": first_frame,"marker_positions": []}
    cv2.setMouseCallback('Select Markers', mouse_callback, mouse_params) # set mouse callback function defn above
    
    # inf loop until user hits esc to cancel or enter to confirm selections
    while True:
        key = cv2.waitKey(1) # wait to capture input
        if key == 27: # 27 is ASCII for escape key
            print("SELECTIONS CANCELLED")
            sys.exit()
        elif key == 13: #13 ASCII for Enter key
            break

    print(f"Selected positions: {mouse_params['marker_positions']}")

    # close windows upon hitting select
    cap.release()
    cv2.destroyAllWindows()
    
    return mouse_params['marker_positions']

if __name__ == '__main__':
    video_path = "videos/2024-01-23_Test video_DI-01232024120123.avi"
    selected_markers = select_markers(video_path)

    # get video metadata
    '''width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(5))
    n_frames = int(cap.get(7))
    print(width, height, fps, n_frames)'''

