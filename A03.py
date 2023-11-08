import cv2
import numpy as np
from skimage.segmentation import slic
from scipy.spatial.distance import cdist

def find_WBC(image):
    # step 1: get superpixel groups
    segments = slic(image, n_segments=100, compactness=10, start_label=0)

    # step 2: compute mean color per superpixel
    cnt = len(np.unique(segments))
    group_means = np.zeros((cnt, 3), dtype="float32")
    for specific_group in range(cnt):
        mask_image = np.where(segments == specific_group, 255, 0).astype("uint8")
        mask_image = np.expand_dims(mask_image, axis=-1)
        group_means[specific_group] = cv2.mean(image, mask=mask_image)[0:3]

    # step 3: use kmeans on GROUP mean colors, group into 4 color groups
    k = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, bestLabels, centers = cv2.kmeans(group_means, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # step 4: find kmeans group w/ mean closest to wbc/rbc
    target_color = np.array([255, 0, 0], dtype=np.float32)
    distances = cdist(centers, [target_color], metric='euclidean')
    closest_group_idx = np.argmin(distances)

    # step 5: set kmeans group to white and the rest to black
    centers = np.uint8(centers)
    for i in range(k):
        if i == closest_group_idx:
            centers[i] = [255, 255, 255]
        else:
            centers[i] = [0, 0, 0]

    # step 6: determine the new colors for each superpixel group
    colors_per_clump = centers[bestLabels.flatten()]

    # step 7: recolor superpixels w/ new group colors (white/black)
    cell_mask = colors_per_clump[segments]
    cell_mask = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2GRAY)

    # step 8: use cv2.connectedComponents to get disjoint blobs from cell_mask
    retval, labels = cv2.connectedComponents(cell_mask, connectivity=8)

    # step 9: for each blob group (EXCEPT 0), get coords to get bounding box and add to list of boxes
    bounding_boxes = []
    for i in range(1, retval):
        coords = np.where(labels == i)
        if coords[0].size > 0:
            ymin, xmin, ymax, xmax = np.min(coords[0]), np.min(coords[1]), np.max(coords[0]), np.max(coords[1])
            bounding_boxes.append((ymin, xmin, ymax, xmax))

    return bounding_boxes