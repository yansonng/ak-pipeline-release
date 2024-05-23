import cv2
import numpy as np
from PIL import Image
import skimage.segmentation as seg

'''
This file contains all the functions needed to perform the mask building pipeline.
'''

def get_ksize(sigma):
    # opencv calculates ksize from sigma as
    # sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    # then ksize from sigma is
    # ksize = ((sigma - 0.8)/0.15) + 2.0

    return int(((sigma - 0.8)/0.15) + 2.0)

def get_gaussian_blur(img, ksize=0, sigma=5):
    # if ksize == 0, then compute ksize from sigma
    if ksize == 0:
        ksize = get_ksize(sigma)

    # Gaussian 2D-kernel can be separable into 2-orthogonal vectors
    # then compute full kernel by taking outer product or simply mul(V, V.T)
    sep_k = cv2.getGaussianKernel(ksize, sigma)

    # if ksize >= 11, then convolution is computed by applying fourier transform
    return cv2.filter2D(img, -1, np.outer(sep_k, sep_k))

def ssr(img, sigma):
    # Single-scale retinex of an image
    # SSR(x, y) = log(I(x, y)) - log(I(x, y)*F(x, y))
    # F = surrounding function, here Gaussian

    return np.log10(img) - np.log10(get_gaussian_blur(img, ksize=0, sigma=sigma) + 1.0)

def msr(img, sigma_scales=[10, 20, 50],apply_normalization=True):

    # Multi-scale retinex of an image
    # MSR(x,y) = sum(weight[i]*SSR(x,y, scale[i])), i = {1..n} scales

    msr = np.zeros(img.shape)
    # for each sigma scale compute SSR
    for sigma in sigma_scales:
        msr += ssr(img, sigma)

    # divide MSR by weights of each scale
    # here we use equal weights
    msr = msr / len(sigma_scales)

    # computed MSR could be in range [-k, +l], k and l could be any real value
    # so normalize the MSR image values in range [0, 255]
    if apply_normalization:
        msr = cv2.normalize(msr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

    return msr

def find_seed_points(image, min_brightness=225, min_distance=10):
    '''
    Find seed points for the mask building pipeline.

    Parameters:
    image: np.array
        The image to find the seed points in, preferably grayscale.
    min_brightness: int
        The minimum brightness value to consider a point a seed point.
    min_distance: int
        The minimum distance between seed points. Set to 0 to disable distance limitation.

    Returns:
    points_distanceLimited: list[tuple(int, int)]
        A list of seed points.
    '''
    # Find points based on brightness
    points = []
    _, thresholded = cv2.threshold(image, min_brightness, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            points.append((cX, cY))

    # Points distance limitation
    points_distanceLimited = []
    for point in points:
        if all(np.linalg.norm(np.array(point) - np.array(existing_point)) > min_distance for existing_point in points_distanceLimited):
            points_distanceLimited.append(point)
    return points_distanceLimited

def flood_fill(image, seed, lower=80, upper=50):
    '''
    Flood fills one image based on the seed point provided.

    Parameters:
    image: np.array
        The image to flood fill.
    seed: tuple(int, int)
        The seed point to flood fill from.

    Returns:
    mask: np.array
        The mask created by the flood fill. Grayscale. (0-black, 1-white)
    '''
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8) # cv2 flood fill requires a mask with 1 pixel border
    image_copy = image.copy()
    cv2.floodFill(image_copy, mask, seed, (255, 255, 255), (lower, lower, lower), (upper, upper, upper), cv2.FLOODFILL_FIXED_RANGE)
    mask = mask[1:-1, 1:-1] # remove the border
    return mask

def overlay_mask_and_original(original, mask):
    '''
    Overlay the mask on the original image.

    Parameters:
    original: np.array
        Original image used as background. Should be grayscale. (0-black, 255-white)
    mask: np.array
        Mask to overlay on top of the original image. Should be grayscale. (0-black, 1-white)

    Returns:
    composite: np.array
        The original image with the mask overlayed on top. Grayscale. (0-black, 255-white)
    '''
    mask = mask*255
    condition = mask == 255
    composite = np.where(condition, original, mask)
    return composite

def find_centroid(masked_image): #Broken
    '''
    Find the centroid of the masked image.

    Parameters:
    masked_image: np.array
        The masked image to find the centroid of. Should be grayscale.

    Returns:
    xcoord: int
        The x coordinate of the centroid.
    ycoord: int
        The y coordinate of the centroid.
    '''
    im_clear = seg.clear_border(masked_image)
    y, x = np.nonzero(im_clear)
    xcoord = x.mean()
    ycoord = y.mean()
    return xcoord, ycoord

def crop_100x100_from_point(image, point, adaptive_pad_color=False):
    '''
    Crop a 100x100 image from the point provided.

    Parameters:
    image: np.array
        The image to crop from.
    point: tuple(int, int)
        The point to crop from.
    adaptive_pad_color: bool
        Whether to use the image's average color as padding color.

    Returns:
    cropped: np.array
        The cropped image.
    '''
    # Pad 50 pixels on all sides
    if adaptive_pad_color: # Use the image's average color as padding
        pad_color = image.mean(axis=(0, 1))
    else:
        pad_color = [0, 0, 0]
    imagePadded = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=pad_color)
    # Shift crop point by 50 pixels
    x, y = point
    x += 50
    y += 50
    # Crop 100x100
    cropped = imagePadded[int(y)-50:int(y)+50, int(x)-50:int(x)+50]
    return cropped


def label_masks_mean_brightness(masks, mean_threshold=0.015):
    '''
    Label masks based on their mean brightness.

    Parameters:
    masks: list[np.array]
        A list of masks to label. Grayscale.
    mean_threshold: float
        The threshold to determine if a mask is bright or dark.

    Returns:
    labels: list[int]
        A list of labels. Below threshold will be positive, above will be negative.
    '''
    labels = []
    for mask in masks:
        if mask.mean() > mean_threshold:
            labels.append('negative')
        else:
            labels.append('positive')
    return labels
