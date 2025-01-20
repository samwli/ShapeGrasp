import os
import numpy as np
import cv2

def load_height(base_path):
    if os.path.exists(base_path + '.npy'):
        data = np.load(base_path + '.npy')
    elif os.path.exists(base_path + '.png'):
        image = cv2.imread(base_path + '.png')
        data = np.array(image)
    else:
        raise ValueError("File not found or unsupported file format (npy or png)")
    
    if len(data.shape) > 2:
        data = data[:, :, 0]
    return data

def load_confidence(base_path):
    if os.path.exists(base_path + '.npy'):
        confidence_map = np.load(base_path + '.npy')
    elif os.path.exists(base_path + '.png'):
        confidence_map = cv2.imread(base_path + '.png')
        confidence_map = np.array(confidence_map)
    else:
        raise ValueError("File not found or unsupported file format (npy or png)")
    
    if len(confidence_map.shape) > 2: 
        confidence_map = confidence_map[:,:,0]
    return confidence_map

def load_mask(base_path):
    if os.path.exists(base_path + '.npy'):
        mask = np.load(base_path + '.npy')
    elif os.path.exists(base_path + '.png'):
        mask = cv2.imread(base_path + '.png')
        mask = np.array(mask)
    else:
        raise ValueError("File not found or unsupported file format (npy or png)")
    
    if len(mask.shape) > 2: 
        mask = np.any(mask > 0, axis=2).astype(np.uint8)
    mask = np.where(mask > 0, 1, 0).astype(np.uint8)
    kernel = np.ones((4,4), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # top_level_contour_indices = np.where(hierarchy[0,:,3] == -1)[0]
    # largest_contour = max(top_level_contour_indices, key=lambda index: cv2.contourArea(contours[index]))
    # contour_mask = np.zeros_like(mask)
    # cv2.drawContours(contour_mask, contours, largest_contour, color=255, thickness=cv2.FILLED)
    # mask = cv2.bitwise_and(mask, contour_mask)
    return np.where(mask > 0, 1, 0)