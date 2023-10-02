import numpy as np
import cv2
import gradio as gr

def create_unnormalized_hist(image):
    hist = np.zeros(256, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_value = image[i, j]
            hist[pixel_value] += 1
    return hist

def normalize_hist(hist):
    total_pixels = np.sum(hist)
    return hist / total_pixels

def create_cdf(nhist):
    cdf = np.cumsum(nhist)
    return cdf.astype(np.float32)

def get_hist_equalize_transform(image, do_stretching, do_cl=False, cl_thresh=0):
    unnormalized_hist = create_unnormalized_hist(image)
    normalized_hist = normalize_hist(unnormalized_hist)
    cdf = create_cdf(normalized_hist)

    if do_stretching:
        cdf_min = np.min(cdf)
        cdf_max = np.max(cdf)
        cdf_stretched = (cdf - cdf_min) / (cdf_max - cdf_min) * 255.0
        int_transform = np.round(cdf_stretched).astype(np.uint8)
    else:
        int_transform = np.round(cdf * 255.0).astype(np.uint8)
    
    return int_transform

def do_histogram_equalize(image, do_stretching):
    output = np.copy(image)
    int_transform = get_hist_equalize_transform(image, do_stretching)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_value = image[i, j]
            new_pixel_value = int_transform[pixel_value]
            output[i, j] = new_pixel_value

    return output

def intensity_callback(input_img, do_stretching):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    output_img = do_histogram_equalize(input_img, do_stretching)
    return output_img

def main():
    demo = gr.Interface(fn=intensity_callback,
                        inputs=["image", "checkbox"],
                        outputs=["image"])
    demo.launch()

if __name__ == "__main__":
    main()
