import cv2
import tracking
import numpy as np
import time
import torch
import torch.nn.functional as F


def improve_smoothing(frame):
    denoised = cv2.fastNlMeansDenoising(frame, None, h=30, templateWindowSize=7, searchWindowSize=21)
    
    # Create a high-pass filter
    high_pass_kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])

    # Apply the high-pass filter using convolution
    high_pass = cv2.filter2D(frame, -1, high_pass_kernel)
    return high_pass

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

def non_local_means_denoising(image):
    return cv2.fastNlMeansDenoising(image, None, h=30, templateWindowSize=7, searchWindowSize=21)


def fourier_transform_filtering(image):
    # Ensure the image is in float32 format
    image_float32 = image.astype(np.float32)
    
    # Perform Fourier Transform
    dft = cv2.dft(image_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 30  # Radius of the low-pass filter
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    return cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

def total_variation_denoising_gpu(image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)

    # Define the TV regularization weight and the number of iterations
    weight = 0.01
    num_iters = 50

    # TV denoising
    for _ in range(num_iters):
        grad_x = torch.diff(image_tensor, dim=2, append=image_tensor[:, :, -1:, :])
        grad_y = torch.diff(image_tensor, dim=3, append=image_tensor[:, :, :, -1:])
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        div_x = torch.diff(grad_x / grad_mag, dim=2, prepend=grad_x[:, :, :1, :])
        div_y = torch.diff(grad_y / grad_mag, dim=3, prepend=grad_y[:, :, :, :1])
        image_tensor = image_tensor - weight * (div_x + div_y)
    
    denoised_image = image_tensor.squeeze().cpu().numpy()
    # Normalize the denoised image to the original image's intensity range
    denoised_image = cv2.normalize(denoised_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return denoised_image

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

        scaled_frame, _ = tracking.scale_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0.5)
        original = add_text(scaled_frame.copy(), 'original')
        
        #tv_denoise = add_text(total_variation_denoising_gpu(scaled_frame.copy()), 'tv')
        nlm_denoise = add_text(non_local_means_denoising(scaled_frame.copy()), 'nlm')
        hp_filter_subtract = add_text(hp_subtract(scaled_frame.copy()), 'hp_subtract')
        hp_filter_kernel = add_text(hp_kernel(scaled_frame.copy()), 'hp_kernel')
        nlm_hp_filter_subtract = add_text(nlm_hp_subtract(scaled_frame.copy()), 'nlm_hp_subtract')
        nlm_hp_filter_kernel = add_text(nlm_hp_kernel(scaled_frame.copy()), 'nlm_hp_kernel')

        frame_hstack_top = np.hstack((original, nlm_denoise, hp_filter_subtract))
        frame_hstack_bottom = np.hstack((hp_filter_kernel, nlm_hp_filter_subtract, nlm_hp_filter_kernel))
        frame_stack = np.vstack((frame_hstack_top, frame_hstack_bottom))


        cv2.imshow("Tracking...", frame_stack)  # show updated frame tracking

        if cv2.waitKey(1) == 27:
            break

        time.sleep(0.25)
        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = r"C:\Users\Brandon\Documents\00 School Files 00\University\M3b Software\temp_videos\C2-20240522_20222605_a549_RFP_stiff Elastic_AS_current_02.vsi - 002 PH, TRITC.avi"
    test(video_path)