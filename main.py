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
        cv2.circle(first_frame, cur_marker[0], 10, (255, 255, 0), 2) # draw circle where clicked
        cv2.imshow('Select Markers', first_frame)

def select_markers(cap):
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
            print(f"Selected positions: {mouse_params['marker_positions']}")
            break

    # close windows upon hitting select
    cv2.destroyAllWindows()
    
    return mouse_params['marker_positions'], first_frame


def track_markers(marker_positions, first_frame, cap):
    # create trackers
    trackers = []
    for _ in range(len(marker_positions)):
        trackers.append(cv2.TrackerKCF_create())

    # initialize trackers
    for i, m_pos in enumerate(marker_positions):
        bbox = (m_pos[0][0], m_pos[0][1], 20, 20) # 20x20 bounding box
        trackers[i].init(first_frame, bbox)

    # tracking loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break # break when frame read unsuccessful (end of video or error)
        
        # updating trackers for all markers
        for i, tracker in enumerate(trackers):
            success, bbox = tracker.update(frame)
            if success:
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)

        cv2.imshow("Tracking...", frame) # show updated frame tracking

        if cv2.waitKey(1) == 27: # cut tracking loop short if ESC hit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = "videos/2024-01-23_Test video_DI-01232024120123.avi"
    cap = cv2.VideoCapture(video_path) # load video

    # get video metadata
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(5))
    n_frames = int(cap.get(7))
    print(width, height, fps, n_frames)

    selected_markers, first_frame = select_markers(cap) # prompt to select markers
    track_markers(selected_markers, first_frame, cap)
