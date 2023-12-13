import cv2
import numpy as np
from General_A04 import *

# step 1: return the correct lbp label
def getOneLBPLabel(subimage, label_type):
    # create 1D array
    center_value = subimage[1, 1]
    binary_string = ""
  
    # get binary representation based on thresholding
    pos = [(0,0), (0,1), (0,2), (1,2), (2,2), (2,1), (2,0), (1,0)]
    for i,j in pos:
        if subimage[i, j] > center_value:
            binary_string += "1"
        else:
            binary_string += "0"

    # check if it's a uniform pattern
    M = 0
    for i in range(1, len(binary_string)):
        if binary_string[i] != binary_string[i-1]:
            M += 1    
    
    if M <= 2:
        # sum up the uniform pattern values
        label = sum_digits(binary_string)
        return label
    else:
        # junk label 9
        return 9 
    
def sum_digits(binary_string):
    return sum(int(x) for x in binary_string)
    
# step 2: return uniform lbp label image
def getLBPImage(image, label_type):
    radius = 1
    neighbors = 8
    padded_image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    lbp_image = np.zeros_like(image, dtype=np.uint8)
    
    for i in range(1, image.shape[0] + 1):
        for j in range(1, image.shape[1] + 1):
            subimage = padded_image[i - radius:i + radius + 1, j - radius:j + radius + 1]
            lbp_label = getOneLBPLabel(subimage, label_type)
            lbp_image[i - 1, j - 1] = lbp_label
    
    return lbp_image

# step 3: return correct lbp histogram
def getOneRegionLBPFeatures(subImage, label_type):
    unique_labels, counts = np.unique(subImage, return_counts=True)
    histogram = np.zeros(10, dtype=np.float32)
    
    for label, count in zip(unique_labels, counts):
        histogram[label] = count / subImage.size
    
    return histogram

# step 4: grab features
def getLBPFeatures(featureImage, regionSideCnt, label_type):
    # subregion height and with
    subregion_width = featureImage.shape[1] // regionSideCnt
    subregion_height = featureImage.shape[0] // regionSideCnt
    all_hists = []
    
    for i in range(regionSideCnt):
        for j in range(regionSideCnt):
            start_row = i * subregion_height
            start_col = j * subregion_width
            end_row = start_row + subregion_height
            end_col = start_col + subregion_width
            
            subimage = featureImage[start_row:end_row, start_col:end_col]
            sub_hist = getOneRegionLBPFeatures(subimage, label_type)
            all_hists.append(sub_hist)
    
    all_hists = np.array(all_hists)
    all_hists = np.reshape(all_hists, (all_hists.shape[0] * all_hists.shape[1],))
    
    return all_hists
