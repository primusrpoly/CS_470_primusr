import cv2
import numpy as np
import gradio as gr

def read_kernel_file(filepath):
    with open(filepath, 'r') as file:
        firstLine = file.readline()
        tokens = firstLine.split()
        rowCnt = int(tokens[0])
        colCnt = int(tokens[1])
        kernel = np.zeros((rowCnt, colCnt), dtype=float)

        index = 2
        for row in range(rowCnt):
            for col in range(colCnt):
                kernel[row, col] = float(tokens[index])
                index += 1

    return kernel

def apply_filter(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)
    kernel = cv2.flip(kernel, -1)
    
    kernelHeight, kernelWidth = kernel.shape

    padding = (kernelHeight // 2, kernelWidth // 2)
    paddedImage = cv2.copyMakeBorder(image, padding[0], padding[0], padding[1], padding[1], cv2.BORDER_CONSTANT, value=0)

    output = np.zeros_like(image, dtype=np.float64)
    imageHeight, imageWidth = image.shape

    for row in range(imageHeight):
        for col in range(imageWidth):
            subImage = paddedImage[row : (row + kernel.shape[0]), col : (col + kernel.shape[1])]
            filtervals = subImage * kernel
            value = np.sum(filtervals)
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
    
