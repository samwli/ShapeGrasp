import numpy as np
import os
from trimesh import voxel
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
from scipy import ndimage
from code.load_data import load_mask

def replace_noisy_patches(image):
    # Convert the image to a numpy array if it's not already
    img_array = np.array(image)
    # Identify pixels with value 255
    mask = (img_array > 760.0) | (img_array == 0.0)
    if mask.sum() == 0:
        return img_array
    # Create an array with NaN where the pixels are 255 and same values everywhere else
    img_with_nans = img_array.astype(float)
    img_with_nans[mask] = np.nan

    # Try to interpolate the NaN values
    filled_image = ndimage.generic_filter(img_with_nans, lambda x: np.nanmean(x), size=50)

    # Handle any remaining NaNs (if any) after interpolation
    filled_image = np.nan_to_num(filled_image, nan=np.nanmean(img_with_nans))

    # Replace NaN values with interpolated values
    img_array[mask] = filled_image[mask]
    
    return img_array

def load_height(base_path, mask):
    if os.path.exists(base_path + '.npy'):
        data = np.load(base_path + '.npy')
    elif os.path.exists(base_path + '.png'):
        image = cv2.imread(base_path + '.png')
        data = np.array(image)
    else:
        raise ValueError("File not found or unsupported file format (npy or png)")
    data = data.astype(float)
    if len(data.shape) > 2:
        data = data[:, :, 0]
    # data = cv2.medianBlur(data, 5)
    data = cv2.GaussianBlur(data, (5, 5), 0)
    neg_indices = np.argwhere(mask == 0)
    for i, j in neg_indices:
        data[i, j] = np.nan
    indices = np.argwhere(mask == 1)
    max_height = np.max([data[i, j] for i, j in indices])
    data = max_height-data
    data = replace_noisy_patches(data)
    return data

def create_solid_mesh(obj_data_path, mesh_file, mode):
    mask = load_mask(obj_data_path+'_mask')
    # cv2.imwrite('mask_image_yeti.png', (mask * 255).astype(np.uint8))
    if mode == '3d':
        # # overlay mask on depth image
        # png_image = cv2.cvtColor(cv2.imread(obj_data_path+'_depth.png'), cv2.COLOR_BGR2BGRA)
        # mask_rgba = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGBA)
        # mask_color = [0, 0, 255, 100] 
        # mask_rgba[np.where(mask == 1)] = mask_color
        # overlay_image = cv2.addWeighted(png_image, 1, mask_rgba, 0.4, 0)
        # cv2.imwrite('overlay_image.png', overlay_image)
        indices = np.argwhere(mask == 1)
        height_array = load_height(obj_data_path+'_depth', mask)
        height_array = np.nan_to_num(height_array, nan=0)
        max_height = np.max([height_array[i, j] for i, j in indices])
  
        height_array = 200*(height_array / max_height)
        min_height = np.min([height_array[i, j] for i, j in indices])-1
        height_array = np.round(height_array - min_height).astype(int)
        max_height = np.max([height_array[i, j] for i, j in indices])
        
        voxels = np.zeros((mask.shape[0], mask.shape[1], max_height), dtype=bool)
        for i, j in indices:
            height = int(height_array[i, j])
            voxels[i, j, :height] = True
        solid_mesh = voxel.ops.matrix_to_marching_cubes(voxels, pitch=1)
    elif mode == '2d':
        # process real world mask if needed
        solid_mesh = voxel.ops.matrix_to_marching_cubes([mask], pitch=1)
    solid_mesh.export(mesh_file)
    return solid_mesh

def load_height(base_path, mask):
    if os.path.exists(base_path + '.npy'):
        data = np.load(base_path + '.npy')
    elif os.path.exists(base_path + '.png'):
        image = cv2.imread(base_path + '.png')
        data = np.array(image)
    else:
        raise ValueError("File not found or unsupported file format (npy or png)")
    data = data.astype(float)
    if len(data.shape) > 2:
        data = data[:, :, 0]
    # data = cv2.medianBlur(data, 5)
    # data = cv2.GaussianBlur(data, (5, 5), 0)
    neg_indices = np.argwhere(mask == 0)
    for i, j in neg_indices:
        data[i, j] = np.nan
    indices = np.argwhere(mask == 1)
    max_height = np.max([data[i, j] for i, j in indices])
    data = max_height-data
    data = replace_noisy_patches(data)
    return data
