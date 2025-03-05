import numpy as np
import os
import cv2
from glob import glob
import argparse

# IS THIS NEEDED??????????????????????

def convert_for_annotation(download_folder_path, semantic_label_path, binary_label_path, binary_image_path, patient_id,frame_index):
    """

    Parameters
    ----------
    download_folder_path(str): Contains the path for predicted binary image a
    semantic_label_path(str):  Contains the path for semantics images. Please use Save semantic images first.
    binary_label_path(str):
    binary_image_path(str):
    patient_id(str):
    frame_index(str):

    Returns:
     type: True if executed sucessfully.
    -------

    """
    if not os.path.isdir(semantic_label_path):
        os.makedirs(semantic_label_path)

    if not os.path.isdir(binary_label_path):
        os.makedirs(binary_label_path)

    if not os.path.isdir(binary_image_path):
        os.makedirs(binary_image_path)

    # get patient id
    patient_id = os.listdir(f"{download_folder_path}/upload")[0]
    # get predicted frame index
    frame_index = glob(f"{download_folder_path}/upload/{patient_id}/predict/*_p.png")[0]
    frame_index = frame_index[frame_index.rfind("/")+1: frame_index.rfind("_p")]
    print("discovered frame index =", frame_index)
    new_patient_id = patient_id
    print("discovered patient id =", patient_id, "will be converted to", new_patient_id)

    if not os.path.isdir(f"{semantic_label_path}/{new_patient_id}"):
        os.makedirs(f"{semantic_label_path}/{new_patient_id}")

    # write grayscale image
    img = cv2.imread(f"{download_folder_path}/upload/{patient_id}/images/{frame_index}.png", cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(f"{binary_image_path}/{new_patient_id}.png", img)

    # write binary image
    img = cv2.imread(glob(f"{download_folder_path}/upload/{patient_id}/predict/*_p.png")[0], cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(f"{binary_label_path}/{new_patient_id}.png", img)

    # save semantic image
    for semantic_image in glob(f"{download_folder_path}/upload/{patient_id}/semantic/*.png"):
        segment_name = semantic_image[semantic_image.rfind("_")+1: semantic_image.rfind(".png")]
        print(segment_name)
        img = cv2.imread(semantic_image, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(f"{semantic_label_path}/{new_patient_id}/{new_patient_id}_{segment_name}.png", img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_folder_path", type=str, default="/home/weihuazhou/Desktop/ICAannotation/static") # FROM ULABLE FLURO
    parser.add_argument("--semantic_label_path", type=str, default="/home/weihuazhou/Desktop/ICAannotation/data/data_musc/image")
    parser.add_argument("--binary_label_path", type=str, default="/home/weihuazhou/Desktop/ICAannotation/data/data_musc/label")
    parser.add_argument("--binary_image_path", type=str, default="/home/weihuazhou/Desktop/ICAannotation/data/data_musc/training")
    parser.add_argument("--patient_id", type=str, default="89667969_22")
    args = parser.parse_args()

    convert_for_annotation(args.download_folder_path, args.semantic_label_path, args.binary_label_path, args.binary_image_path, args.patient_id)
