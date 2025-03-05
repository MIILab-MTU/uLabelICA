import math
import uuid


import PIL.Image
import PIL.ImageDraw
import argparse
import base64
import json
import os
import os.path as osp
import numpy as np
import PIL.Image

from labelme import utils
from labelme.logger import logger


def polygons_to_mask(img_shape, polygons, shape_type=None):
    logger.warning(
        "The 'polygons_to_mask' function is deprecated, "
        "use 'shape_to_mask' instead."
    )
    return shape_to_mask(img_shape, points=polygons, shape_type=shape_type)


def shape_to_mask(
    img_shape, points, shape_type=None, line_width=10, point_size=5
):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins


def labelme_shapes_to_label(img_shape, shapes):
    logger.warn(
        "labelme_shapes_to_label is deprecated, so please use "
        "shapes_to_label."
    )

    label_name_to_value = {"_background_": 0}
    for shape in shapes:
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value

    lbl, _ = shapes_to_label(img_shape, shapes, label_name_to_value)
    return lbl, label_name_to_value


def masks_to_bboxes(masks):
    if masks.ndim != 3:
        raise ValueError(
            "masks.ndim must be 3, but it is {}".format(masks.ndim)
        )
    if masks.dtype != bool:
        raise ValueError(
            "masks.dtype must be bool type, but it is {}".format(masks.dtype)
        )
    bboxes = []
    for mask in masks:
        where = np.argwhere(mask)
        (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
        bboxes.append((y1, x1, y2, x2))
    bboxes = np.asarray(bboxes, dtype=np.float32)
    return bboxes



# def json_to_images(json_file, patient_id):
#     #out_dir = '/home/weihuazhou/Desktop/automated-labelme/labelme/data/predicted/semantic/{}_{}'.format(patient_id,frame_no)
#     out_dir = os.path.join(os.getcwd(),"data/image/{}".format(patient_id))
#     if os.path.splitext(json_file)[1] != '.json':
#         raise TypeError("File must be of .json format")
#
#     else:
#         if not osp.exists(out_dir):
#             os.mkdir(out_dir)
#
#         data = json.load(open(json_file))
#         imageData = data.get("imageData")
#
#         if not imageData:
#             imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
#             with open(imagePath, "rb") as f:
#                 imageData = f.read()
#                 imageData = base64.b64encode(imageData).decode("utf-8")
#         img = utils.img_b64_to_arr(imageData)
#
#         label_name_to_value = {"Binary_Original": 0}
#         for shape in sorted(data["shapes"], key=lambda x: x["label"]):
#             label_name = shape["label"]
#             if label_name in label_name_to_value:
#                 label_value = label_name_to_value[label_name]
#             else:
#                 label_value = len(label_name_to_value)
#                 label_name_to_value[label_name] = label_value
#
#         # Create a directory to save individual label images
#         label_images_dir = out_dir
#         if not osp.exists(label_images_dir):
#             os.mkdir(label_images_dir)
#
#         for label_name, label_value in label_name_to_value.items():
#             lbl, _ = utils.shapes_to_label(
#                 img.shape, [shape for shape in data["shapes"] if shape["label"] == label_name], label_name_to_value
#             )
#
#             # Create a mask for the specific label
#             label_mask = (lbl == label_value).astype(np.uint8) * 255
#
#             # Apply the mask to the original image
#             label_img = img.copy()
#             label_img[label_mask == 0] = 0
#             label_img[np.where(label_img[:, :, 3] == 0)] = (0, 0, 0, 255)  # Set other pixels to black from transparent
#             # Save the image with only the labeled part
#             label_image_path = osp.join(label_images_dir, "{}_{}.png".format(str(patient_id),label_name))
#             PIL.Image.fromarray(label_img).save(label_image_path)
#
#             for filename in os.listdir(label_images_dir):
#                 if filename.endswith('Binary_Original.png'):
#                     file_path = os.path.join(label_images_dir, filename)
#                     try:
#                         os.remove(file_path)
#                         print(f"Deleted: {file_path}")
#                     except OSError as e:
#                         print(f"Error deleting {file_path}: {e}")
#
#     logger.info("Saved to: {}".format(out_dir))
#     return out_dir, True

def json_to_images(json_file, patient_id):
    #out_dir = '/home/weihuazhou/Desktop/automated-labelme/labelme/data/predicted/semantic/{}_{}'.format(patient_id,frame_no)
    out_dir = os.path.join(os.getcwd(),"data/image/{}".format(patient_id))
    if os.path.splitext(json_file)[1] != '.json':
        raise TypeError("File must be of .json format")

    else:
        if not osp.exists(out_dir):
            os.mkdir(out_dir)

        data = json.load(open(json_file))
        imageData = data.get("imageData")

        if not imageData:
            imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
            with open(imagePath, "rb") as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode("utf-8")
        img = utils.img_b64_to_arr(imageData)

        label_name_to_value = {"Binary_Original": 0}
        for shape in sorted(data["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value

        # Create a directory to save individual label images
        label_images_dir = out_dir
        if not osp.exists(label_images_dir):
            os.mkdir(label_images_dir)

        for label_name, label_value in label_name_to_value.items():
            lbl, _ = utils.shapes_to_label(
                img.shape, [shape for shape in data["shapes"] if shape["label"] == label_name], label_name_to_value
            )

            # Create a mask for the specific label
            label_mask = (lbl == label_value).astype(np.uint8) * 255

            # Apply the mask to the original image
            label_img = img.copy()
            label_img[label_mask == 0] = 0
            label_img[np.where(label_img[:, :, 3] == 0)] = (0, 0, 0, 255)
            red_mask = (label_img[:, :, 0] == 255) & (label_img[:, :, 1] == 0) & (label_img[:, :, 2] == 0)
            label_img[red_mask, :] = [255, 255, 255, 255]
            label_img[~red_mask, :] = [0, 0, 0, 255]
            label_image_path = osp.join(label_images_dir, "{}_{}.png".format(str(patient_id),label_name))
            PIL.Image.fromarray(label_img).save(label_image_path)

            for filename in os.listdir(label_images_dir):
                if filename.endswith('Binary_Original.png'):
                    file_path = os.path.join(label_images_dir, filename)
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except OSError as e:
                        print(f"Error deleting {file_path}: {e}")

    logger.info("Saved to: {}".format(out_dir))
    return out_dir, True



