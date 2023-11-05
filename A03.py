import cv2
import numpy as np

def find_WBC(image):
    # Step 1: Get superpixel groups using SLIC (Superpixel segmentation)
    segments = cv2.ximgproc.createSuperpixelSLIC(image, cv2.ximgproc.SLICO, region_size=10)
    segments.iterate(10)

    # Step 2: Compute mean color per superpixel
    cnt = len(np.unique(segments))
    group_means = np.zeros((cnt, 3), dtype="float32")

    for specific_group in range(cnt):
        mask_image = np.where(segments.getLabels() == specific_group, 255, 0).astype("uint8")
        mask_image = np.expand_dims(mask_image, axis=-1)
        group_means[specific_group] = cv2.mean(image, mask=mask_image)[0:3]

    # Step 3: Use K-means on GROUP mean colors to group them into 4 color groups
    k = 4 
    group_means = np.float32(group_means)
    _, bestLabels, centers = cv2.kmeans(group_means, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0), 10, cv2.KMEANS_RANDOM_CENTERS)

    # Step 4: Find the k-means group with the mean closest to white blood cells (blue)
    target_color = (255, 0, 0)
    closest_group_idx = np.argmin(np.linalg.norm(centers - target_color, axis=1))

    # Step 5: Set the closest k-means group to white, the rest to black
    centers = np.uint8(centers)
    for i in range(k):
        if i != closest_group_idx:
            centers[i] = [0, 0, 0]
        else:
            centers[i] = [255, 255, 255]

    # Step 6: Determine new colors for each superpixel group
    colors_per_clump = centers[bestLabels.flatten()]

    # Step 7: Recolor the superpixels with their new group colors
    cell_mask = colors_per_clump[segments.getLabels()]
    cell_mask = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2GRAY)

    # Step 8: Use cv2.connectedComponents to get disjoint blobs from cell_mask
    num_labels, labels = cv2.connectedComponents(cell_mask, connectivity=4)

    # Step 9: Get bounding boxes for white blood cells (blobs)
    bounding_boxes = []
    for i in range(1, num_labels):
        coords = np.where(labels == i)
        ymin, xmin = np.min(coords, axis=1)
        ymax, xmax = np.max(coords, axis=1)
        bounding_boxes.append((ymin, xmin, ymax, xmax))

    return bounding_boxes
