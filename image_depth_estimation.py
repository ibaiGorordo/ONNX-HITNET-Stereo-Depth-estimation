import cv2
import numpy as np
from imread_from_url import imread_from_url

from hitnet import HitNet, ModelType, CameraConfig

# Select model type
# model_type = ModelType.middlebury
# model_type = ModelType.flyingthings
model_type = ModelType.eth3d

if model_type == ModelType.middlebury:
	model_path = "models/middlebury_d400/saved_model_480x640/model_float32.onnx"
elif model_type == ModelType.flyingthings:
	model_path = "models/flyingthings_finalpass_xl/saved_model_480x640/model_float32.onnx"
elif model_type == ModelType.eth3d:
	model_path = "models/eth3d/saved_model_480x640/model_float32.onnx"

# Initialize model
depth_estimator = HitNet(model_path, model_type)

# Load images
left_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png")
right_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im6.png")

# Estimate the depth
disparity_map = depth_estimator(left_img, right_img)

color_disparity = depth_estimator.draw_disparity()
combined_image = np.hstack((left_img, color_disparity))

cv2.imwrite("out.jpg", combined_image)

cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)	
cv2.imshow("Estimated disparity", combined_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
