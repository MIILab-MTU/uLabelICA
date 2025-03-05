# import argparse
# import base64
# import json
# import os
# import os.path as osp
#
# import imgviz
# import PIL.Image
#
# from labelme.logger import logger
# from labelme import utils
#
#
# def main():
#     logger.warning(
#         "This script is aimed to demonstrate how to convert the "
#         "JSON file to a single image dataset."
#     )
#     logger.warning(
#         "It won't handle multiple JSON files to generate a "
#         "real-use dataset."
#     )
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("json_file")
#     parser.add_argument("-o", "--out", default=None)
#     args = parser.parse_args()
#
#     json_file = args.json_file
#
#     if args.out is None:
#         out_dir = osp.basename(json_file).replace(".", "_")
#         out_dir = osp.join(osp.dirname(json_file), out_dir)
#     else:
#         out_dir = args.out
#     if not osp.exists(out_dir):
#         os.mkdir(out_dir)
#
#     data = json.load(open(json_file))
#     imageData = data.get("imageData")
#
#     if not imageData:
#         imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
#         with open(imagePath, "rb") as f:
#             imageData = f.read()
#             imageData = base64.b64encode(imageData).decode("utf-8")
#     img = utils.img_b64_to_arr(imageData)
#
#     label_name_to_value = {"_background_": 0}
#     for shape in sorted(data["shapes"], key=lambda x: x["label"]):
#         label_name = shape["label"]
#         if label_name in label_name_to_value:
#             label_value = label_name_to_value[label_name]
#         else:
#             label_value = len(label_name_to_value)
#             label_name_to_value[label_name] = label_value
#     lbl, _ = utils.shapes_to_label(
#         img.shape, data["shapes"], label_name_to_value
#     )
#
#     label_names = [None] * (max(label_name_to_value.values()) + 1)
#     for name, value in label_name_to_value.items():
#         label_names[value] = name
#
#     lbl_viz = imgviz.label2rgb(
#         lbl, imgviz.asgray(img), label_names=label_names, loc="rb"
#     )
#
#     PIL.Image.fromarray(img).save(osp.join(out_dir, "img.png"))
#     utils.lblsave(osp.join(out_dir, "label.png"), lbl)
#     PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, "label_viz.png"))
#
#     with open(osp.join(out_dir, "label_names.txt"), "w") as f:
#         for lbl_name in label_names:
#             f.write(lbl_name + "\n")
#
#     logger.info("Saved to: {}".format(out_dir))
#
#
# if __name__ == "__main__":
#     main()

import argparse
import base64
import json
import os
import os.path as osp
import numpy as np

import imgviz
import PIL.Image

from labelme.logger import logger
from labelme import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()

    json_file = args.json_file

    if args.out is None:
        out_dir = osp.splitext(osp.basename(json_file))[0]
        out_dir = osp.join(osp.dirname(json_file), out_dir)
    else:
        out_dir = args.out
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
    label_images_dir = osp.join(out_dir, "label_images")
    if not osp.exists(label_images_dir):
        os.mkdir(label_images_dir)

    for label_name, label_value in label_name_to_value.items():
        lbl, _ = utils.shapes_to_label(
            img.shape, [shape for shape in data["shapes"] if shape["label"] == label_name], label_name_to_value
        )

        # Create a mask for the specific label
        label_mask = (lbl == label_value).astype(np.uint8) * 255
        print("image",img.shape),

        # Apply the mask to the original image
        label_img = img.copy()
        label_img[label_mask == 0] = 0
        label_img[np.where(label_img[:, :, 3] == 0)] = (0, 0, 0, 255) #Set other pixels to black from transparent
        print("Label", label_img.shape),
        # Save the image with only the labeled part
        label_image_path = osp.join(label_images_dir, f"{label_name}.png")
        PIL.Image.fromarray(label_img).save(label_image_path)

    # Save label names file
    with open(osp.join(out_dir, "label_names.txt"), "w") as f:
        for lbl_name in label_name_to_value.keys():
            f.write(lbl_name + "\n")

    logger.info("Saved to: {}".format(out_dir))


if __name__ == "__main__":
    main()
