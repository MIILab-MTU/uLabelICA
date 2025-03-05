import argparse
import numpy as np

smooth = 1.
dropout_rate = 0.5
act = "relu"


from skimage import measure, filters

# Evaluation metric: IoU
def compute_iou(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)

    if img1.shape[0] != img2.shape[0]:
        raise ValueError("Shape mismatch: the number of images mismatch.")
    IoU = np.zeros((img1.shape[0],), dtype=np.float32)
    for i in range(img1.shape[0]):
        im1 = np.squeeze(img1[i] > 0.5)
        im2 = np.squeeze(img2[i] > 0.5)

        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

        # Compute Dice coefficient
        intersection = np.logical_and(im1, im2)

        if im1.sum() + im2.sum() == 0:
            IoU[i] = 100
        else:
            IoU[i] = 2. * intersection.sum() * 100.0 / (im1.sum() + im2.sum())
        # database.display_image_mask_pairs(im1, im2)

    return IoU


# Evaluation metric: Dice
def compute_dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def search_largest_region(image):
    labeling = measure.label(image)
    regions = measure.regionprops(labeling)

    largest_region = None
    area_max = 0.
    for region in regions:
        if region.area > area_max:
            area_max = region.area
            largest_region = region

    return largest_region


def generate_largest_region(image):
    region = search_largest_region(image)
    bin_image = np.zeros_like(image)
    if region != None:
        for coord in region.coords:
            bin_image[coord[0], coord[1]] = 1

    return bin_image


def threshold_by_otsu(pred_vessels, flatten=True):
    # cut by otsu threshold
    try:
        threshold = filters.threshold_otsu(pred_vessels)
        pred_vessels_bin = np.zeros(pred_vessels.shape)
        pred_vessels_bin[pred_vessels >= threshold] = 1
    except:
        pred_vessels_bin = np.zeros(pred_vessels.shape)

    if flatten:
        return pred_vessels_bin.flatten()
    else:
        return pred_vessels_bin


########################################
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


