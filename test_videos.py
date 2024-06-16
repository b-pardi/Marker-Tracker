import cv2
import tracking
import numpy as np
import time




def nlm_hp_kernel(frame):
    denoised = cv2.fastNlMeansDenoising(frame, None, h=np.std(frame)*1.2, templateWindowSize=5, searchWindowSize=17)
    
    high_pass_kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])

    # Apply the high-pass filter using convolution
    high_pass = cv2.filter2D(denoised, -1, high_pass_kernel)
    return high_pass

def nlm_hp_subtract(frame):
    denoised = cv2.fastNlMeansDenoising(frame, None, h=np.std(frame)*5, templateWindowSize=7, searchWindowSize=21)

    # Subtract the blurred image from the original image to get the high-pass filtered image
    high_pass = cv2.subtract(frame, denoised)

    return high_pass

def hp_kernel(frame):
    # Create a high-pass filter
    high_pass_kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])

    # Apply the high-pass filter using convolution
    high_pass = cv2.filter2D(frame, -1, high_pass_kernel)
    return high_pass

def hp_subtract(frame):
    # Apply a Gaussian blur to create a low-pass filtered image
    blurred_image = cv2.GaussianBlur(frame, (21, 21), 0)

    # Subtract the blurred image from the original image to get the high-pass filtered image
    high_pass = cv2.subtract(frame, blurred_image)

    return high_pass


def improve_binarization(frame):    
    """
    Enhances the binarization of a grayscale image using various image processing techniques. This function applies
    CLAHE for contrast enhancement, background subtraction to highlight foreground objects, morphological operations
    to refine the image, and edge detection to further define object boundaries.

    Steps:
        1. Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to boost the contrast of the image.
        2. Perform background subtraction using a median blur to isolate foreground features.
        3. Apply morphological closing to close small holes within the foreground objects.
        4. Detect edges using the Canny algorithm, and dilate these edges to enhance their visibility.
        5. Optionally adjust the edge thickness with additional morphological operations like dilation or erosion
           depending on specific requirements (commented out in the code but can be adjusted as needed).

    Note:
        - This function is designed to work with grayscale images and expects a single-channel input.
        - Adjustments to parameters like CLAHE limits, kernel sizes for morphological operations, and Canny thresholds
          may be necessary depending on the specific characteristics of the input image.


    Args:
        frame (np.array): A single-channel (grayscale) image on which to perform binarization improvement.

    Returns:
        np.array: The processed image with enhanced binarization and clearer object definitions.

    """

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # boosts contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
    equalized = clahe.apply(frame)
    
    # Perform Background Subtraction
    # (Assuming a relatively uniform background)
    background = cv2.medianBlur(equalized, 13)
    subtracted = cv2.subtract(equalized, background)
    
    # Use morphological closing to close small holes inside the foreground
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(subtracted, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Use Canny edge detector to find edges and use it as a mask
    edges = cv2.Canny(closing, 30, 140)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    result = cv2.bitwise_or(closing, edges_dilated)

    # if edges too fine after result
    #result = cv2.dilate(result, kernel, iterations=1)
    # if edges too thick after result
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.erode(result, kernel, iterations=1)

    return result

def improve_smoothing(frame, strength=1):
    """
    Enhances the input frame by applying Non-Local Means (NLM) denoising followed by a high-pass filter.
    
    NLM averages pixel intensity s.t. similar patches of the image (even far apart) contribute more to the average
    

    Args:
    frame (numpy.ndarray): The input image in grayscale.

    Returns:
    numpy.ndarray: The processed image with improved smoothing and enhanced details.
    """
    noise_level = np.std(frame)
    denoised = cv2.fastNlMeansDenoising(frame, None, h=noise_level*strength, templateWindowSize=7, searchWindowSize=25)
    
    # highlights cental pixel and reduces neighboring pixels
    # passes high frequencies and attenuates low frequencies
    # this kernel represents a discrete ver of the laplacian operator, approximating 2nd order derivative of image
    laplacian_kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])

    # Apply the high-pass filter using convolution
    high_pass = cv2.filter2D(denoised, -1, laplacian_kernel)
    return high_pass

def adjust_gamma(frame, gamma=50.0):
    # Apply gamma correction
    gamma=1-gamma/100
    gamma_corrected = np.array(255 * (frame / 255) ** gamma, dtype='uint8')
    return gamma_corrected

def add_text(image, text, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
    return cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

def test(fp):
    cap = cv2.VideoCapture(fp)
    frame_num = 0
    frame_end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    cv2.namedWindow('Tracking...')
    cv2.moveWindow('Tracking...', 0, 0)

    while frame_num < frame_end:
        ret, frame = cap.read()
        if not ret:
            print("can't read frame")
            break

        scaled_frame, _ = tracking.scale_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0.7)
        original = add_text(scaled_frame.copy(), 'original')
        
        #tv_denoise = add_text(total_variation_denoising_gpu(scaled_frame.copy()), 'tv')
        '''nlm_denoise = add_text(non_local_means_denoising(scaled_frame.copy()), 'nlm')
        hp_filter_subtract = add_text(hp_subtract(scaled_frame.copy()), 'hp_subtract')
        hp_filter_kernel = add_text(hp_kernel(scaled_frame.copy()), 'hp_kernel')
        nlm_hp_filter_subtract = add_text(nlm_hp_subtract(scaled_frame.copy()), 'nlm_hp_subtract')
        nlm_hp_filter_kernel = add_text(nlm_hp_kernel(scaled_frame.copy()), 'nlm_hp_kernel')'''

        smoothedifuckinhope = improve_smoothing(scaled_frame)
        #smoothedifuckinhope = cv2.fastNlMeansDenoising(smoothedifuckinhope, None, h=np.std(frame)*0.5, templateWindowSize=7, searchWindowSize=25)

        binarizedifuckinhope = cv2.adaptiveThreshold(smoothedifuckinhope, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        # {"Blur/Sharpness": -73.21428571428572, "Contrast": 60.223214285714285, "Brightness": 65.78571428571428, "Smoothness": 58.455357142857146, "Binarize": True}
        # {"Blur/Sharpness": -18.604651162790702, "Contrast": 100.0, "Brightness": -7.302325581395351, "Smoothness": 100.0, "Binarize": True}
        # {"Blur/Sharpness": -18.604651162790702, "Contrast": 44.74418604651163, "Brightness": -7.302325581395351, "Smoothness": 100.0, "Binarize": true}
        # {"Blur/Sharpness": -60.46511627906977, "Contrast": 68.9186046511628, "Brightness": -6.976744186046517, "Smoothness": 96.54651162790698, "Binarize": true}
        # {"Blur/Sharpness": -46.162790697674424, "Contrast": 43.68604651162791, "Brightness": -46.51162790697675, "Smoothness": 38.2906976744186, "Binarize": true}
        # {"Blur/Sharpness": -46.162790697674424, "Contrast": 43.68604651162791, "Brightness": -46.51162790697675, "Smoothness": 38.2906976744186, "Binarize": False} - not bad with denoising
        # {"Blur/Sharpness": -100.0, "Contrast": 36.68604651162791, "Brightness": -11.627906976744185, "Smoothness": 42.44186046511628, "Binarize": true, "Denoise SP": true}
        # {"Blur/Sharpness": -100.0, "Contrast": 58.55813953488372, "Brightness": -4.6511627906976685, "Smoothness": 60.860465116279066, "Binarize": false, "Denoise SP": false}
        salt_pep_frame = tracking.preprocess_frame( scaled_frame, {"Blur/Sharpness": -100.0, "Contrast": 66.61627906976744, "Brightness": -25.581395348837205, "Smoothness": 52.80232558139535, "Binarize": False, "Denoise SP": False} , True)

        # saltpep_denoised = tracking.denoise_frame_saltpep(scaled_frame)

        thresholded = cv2.adaptiveThreshold(salt_pep_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        #frame_hstack_top = np.hstack((original, smoothedifuckinhope, binarizedifuckinhope))
        #frame_hstack_bottom = np.hstack((hp_filter_kernel, nlm_hp_filter_subtract, nlm_hp_filter_kernel))
        #frame_stack = np.vstack((frame_hstack_top, frame_hstack_bottom))
        frame_hstack = np.hstack((original, thresholded))
        #frame_hstack = np.hstack((original, saltpep_denoised))

        cv2.imshow("Tracking...",frame_hstack)  # show updated frame tracking

        if cv2.waitKey(1) == 27:
            break

        time.sleep(0.25)
        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = r"C:\Users\ipsou\ProgrammingStuff\Github\Marker-Tracker\viscoelastic_video\20240308_20240307_A549 _stiff VE sub._col-I_AS_current_01.vsi - 001 PH-Cell 1.avi"
    # video_path = r"C:\Users\ipsou\ProgrammingStuff\Github\Marker-Tracker\viscoelastic_video\20240308_20240307_A549 _stiff VE sub._col-I_AS_current_02.vsi - 002 PH-Cell 2.avi"
    #video_path = r"C:\Users\ipsou\ProgrammingStuff\Github\Marker-Tracker\data\20240208_2024_07_02_A549.p23_PAHstiff_no beads_migration_exp.3_current_03.vsi - 003 PH.avi"
    test(video_path)