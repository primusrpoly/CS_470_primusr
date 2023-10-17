import cv2
import numpy as np
import gradio as gr

def read_kernel_file(filepath):
    with open(filepath, 'r') as file:
        line = file.readline()
        tokens = line.split()
        row_count, col_count = int(tokens[0]), int(tokens[1])
        kernel = np.zeros((row_count, col_count), dtype=float)

        index = 2
        for row in range(row_count):
            for col in range(col_count):
                kernel[row, col] = float(tokens[index])
                index += 1

    return kernel

def apply_filter(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
    image = image.astype(np.float64)
    kernel = cv2.flip(kernel, -1)
    kernel_height, kernel_width = kernel.shape

    padding = (kernel_height // 2, kernel_width // 2)
    padded_image = cv2.copyMakeBorder(image, padding[0], padding[0], padding[1], padding[1], cv2.BORDER_CONSTANT, value=0)

    output = np.zeros_like(image, dtype=np.float64)
    image_height, image_width = image.shape

    for row in range(image_height):
        for col in range(image_width):
            sub_image = padded_image[row:row + kernel_height, col:col + kernel_width]
            filter_vals = sub_image * kernel
            value = np.sum(filter_vals)
            output[row, col] = value

    if convert_uint8:
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)

    return output

def filtering_callback(input_img, filter_file, alpha_val, beta_val):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    kernel = read_kernel_file(filter_file.name)
    output_img = apply_filter(input_img, kernel, alpha_val, beta_val)
    return output_img

def main():
    demo = gr.Interface(fn=filtering_callback,
                        inputs=["image", "file", gr.Number(value=0.125), gr.Number(value=127)],
                        outputs=["image"])
    demo.launch()

if __name__ == "__main__":
    main()
