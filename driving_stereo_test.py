import cv2
import numpy as np
import glob
from hitnet import HitNet, ModelType, CameraConfig

def get_driving_stereo_images(base_path, start_sample=0):

	# Get image list
	left_images = glob.glob(f'{base_path}/left/*.jpg')
	left_images.sort()
	right_images = glob.glob(f'{base_path}/right/*.jpg')
	right_images.sort()
	depth_images = glob.glob(f'{base_path}/depth/*.png')
	depth_images.sort()

	return left_images[start_sample:], right_images[start_sample:], depth_images[start_sample:]

# Select model type
model_type = ModelType.middlebury
# model_type = ModelType.flyingthings
# model_type = ModelType.eth3d

if model_type == ModelType.middlebury:
	model_path = "models/middlebury_d400/saved_model_480x640/model_float32.onnx"
elif model_type == ModelType.flyingthings:
	model_path = "models/flyingthings_finalpass_xl/saved_model_480x640/model_float32.onnx"
elif model_type == ModelType.eth3d:
	model_path = "models/eth3d/saved_model_480x640/model_float32.onnx"

# Initialize model
input_width = 640
camera_config = CameraConfig(0.546, 500/1720*input_width) # rough estimate from the original calibration
max_distance = 10
depth_estimator = HitNet(model_path, model_type, camera_config, max_distance)

# Get the driving stereo samples
driving_stereo_path = "drivingStereo"
start_sample = 700
left_images, right_images, depth_images = get_driving_stereo_images(driving_stereo_path, start_sample)

cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)	
for left_path, right_path, depth_path in zip(left_images, right_images, depth_images):

	# Read frame from the video
	left_img = cv2.imread(left_path)
	right_img = cv2.imread(right_path)
	depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)/1000

	# Estimate the depth
	disparity_map = depth_estimator(left_img, right_img)

	color_depth = depth_estimator.draw_depth()
	color_real_depth = depth_estimator.util_draw_depth(depth_img, (left_img.shape[1], left_img.shape[0]), max_distance)
	combined_image = np.hstack((left_img, color_real_depth, color_depth))

	cv2.imshow("Estimated depth", combined_image)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

cv2.destroyAllWindows()