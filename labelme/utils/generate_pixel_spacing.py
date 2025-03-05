import os
import pydicom as dicom
#import matlab.engine

def generate_pixel_spacing(dicom_file_path, pixel_spacing_csv_path):
    """

    Parameters
    ----------
    dicom_file_path: provide path for the dicom image
    pixel_spacing_csv_path: provide path for the pixel spacing csv to be saved

    Returns
    -------

    """
    if os.path.isfile(pixel_spacing_csv_path):
        pixel_spacing_csv = open(pixel_spacing_csv_path, "a")
    else:
        pixel_spacing_csv = open(pixel_spacing_csv_path, "w")
        pixel_spacing_csv.write("patient,spacing,view_angle,first_angle,second_angle\n")
    if not os.path.exists(dicom_file_path):
        os.makedirs(dicom_file_path)
    dcm_files = os.listdir(dicom_file_path)
    print(dcm_files)
    for dcm_file in dcm_files:
        dcm = dicom.read_file(os.path.join(dicom_file_path, dcm_file))
        for i in range(dcm.pixel_array.shape[0]):
            image = dcm.pixel_array[i]
            first_view = dcm.PositionerPrimaryAngle
            second_view = dcm.PositionerSecondaryAngle
            spacing = float(dcm.ImagerPixelSpacing[0]) # this is the pixel spacing
            view = "LAO" if first_view > 0 else "RAO"
            pixel_spacing_csv.write(f"{dcm_file}_{i + 1},{spacing},{view},{first_view},{second_view}\n")

    pixel_spacing_csv.flush()
    pixel_spacing_csv.close()

    return pixel_spacing_csv_path
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dicom_file_path", type=str, default="/home/weihuazhou/Desktop/ICAannotation/data/data_musc/dicom")
#     parser.add_argument("--pixel_spacing_csv_path", type=str, default="/home/weihuazhou/Desktop/ICAannotation/data/data_musc/pixel_spacing.csv")
#     args = parser.parse_args()
#
#     generate_pixel_spacing(args.dicom_file_path, args.pixel_spacing_csv_path)