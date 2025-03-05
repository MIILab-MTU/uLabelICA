import os
import numpy as np

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segmentation_models import Unet,Nestnet,Xnet,FPN_XNET
from segmentation_models.helper_functions import threshold_by_otsu, generate_largest_region, dice_coef_loss, mean_iou, dice_coef
import matplotlib.pyplot as plt
import cv2




def get_binary_model():
    model = FPN_XNET(
        backbone_name='inceptionresnetv2', encoder_weights='imagenet',
        decoder_block_type='transpose', input_shape=(512, 512, 3))
    model.load_weights(
        os.path.join(
            os.getcwd(),
            "model" + os.sep + "weights.h5"
        )
    )

    return model



def predict_binary(model, img_path, patient_id):
    # Set up save paths
    save_image_path1 = os.path.join(os.getcwd(), "data/predicted/{}".format(str(patient_id)))
    if not os.path.exists(save_image_path1):
        os.mkdir(save_image_path1)

    frame_no = patient_id.split('_')[1]
    save_image_path = os.path.join(save_image_path1, "{}.png".format(str(frame_no)))
    save_image_path2 = os.path.join(save_image_path1, "{}_p.png".format(str(frame_no)))
    save_image_path3 = os.path.join(os.path.join(os.getcwd(), "data/label"), "{}.png".format(str(patient_id)))
    highlighted_image_path = os.path.join(os.path.join(os.getcwd(), "data/label"), "highlight_{}.png".format(str(patient_id)))

    # Read input image
    input_image = cv2.imread(img_path)
    highlighted_image = input_image.copy()

    # Save the original image for training
    upload_image_path = os.path.join(os.getcwd(), "data/training/{}.png".format(str(patient_id)))
    cv2.imwrite(upload_image_path, input_image)

    # Preprocess input image
    image_shape = input_image.shape[:2]
    input_image = cv2.resize(input_image, (512, 512)) / 255

    # Predict using the model
    predicted_image = model.predict(np.expand_dims(input_image, axis=0))

    # Save predicted image
    plt.imsave(fname=save_image_path, arr=np.squeeze(predicted_image), cmap="gray")

    # Threshold and generate largest region
    image_bin = threshold_by_otsu(np.squeeze(predicted_image), flatten=False)
    image_bin_lr = generate_largest_region(np.squeeze(image_bin))
    image_bin_lr = cv2.resize(image_bin_lr, (image_shape[0], image_shape[1]))

    # Save processed images
    plt.imsave(fname=save_image_path2, arr=np.squeeze(image_bin_lr), cmap="gray")
    plt.imsave(fname=save_image_path3, arr=np.squeeze(image_bin_lr), cmap="gray")

    # Highlight arteries in red
    binary_image = cv2.imread(save_image_path2, cv2.IMREAD_GRAYSCALE)
    highlighted_image[binary_image > 0] = [255, 0, 0]

    # Save the final highlighted image
    plt.imsave(fname=highlighted_image_path, arr=np.squeeze(highlighted_image), cmap="gray")

    return highlighted_image_path
