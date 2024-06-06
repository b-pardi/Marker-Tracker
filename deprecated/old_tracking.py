def track_area_og(cap, frame_start, frame_end, frame_interval, time_units):
    frame_num = frame_start

    while True:  # read frame by frame until the end of the video
        ret, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start + frame_num)
        frame_num += 1
        #time.sleep(0.25)

        if not ret:
            break

        # frame preprocessing
        scaled_frame, scale_factor = scale_frame(frame)  # scale the frame
        gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)  # convert frame to gray
        kernel_size = (5,5)
        blur_frame = cv2.GaussianBlur(gray_frame, kernel_size, 0) # gaussian blur
        _, binary_frame = cv2.threshold(blur_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # threshold to binarize image

        # segment frame
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find contours
        contour_frame = np.zeros_like(binary_frame) 
        cv2.drawContours(scaled_frame, contours, -1, (255, 0, 0), 2)

        # label and segment contours
        for i, contour in enumerate(contours):
            m = cv2.moments(contour) # calculate moment contours
            if m["m00"] != 0: # if contour has a non-zero zeroth spatial moment (surface area)
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                cv2.putText(scaled_frame, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    

        cv2.imshow('Surface Area Tracking', scaled_frame)
        if cv2.waitKey(1) == 27 or frame_end == frame_num+frame_start:
            break

    cap.release()
    cv2.destroyAllWindows()