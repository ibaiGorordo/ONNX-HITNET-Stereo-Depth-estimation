import cv2
import pafy
import numpy as np
import glob
from hitnet import HitNet, ModelType, CameraConfig

# Initialize video
# cap = cv2.VideoCapture("video.mp4")

videoUrl = 'https://youtu.be/Yui48w71SG0'
videoPafy = pafy.new(videoUrl)
print(videoPafy.streams)
cap = cv2.VideoCapture(videoPafy.getbestvideo().url)

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

# Store baseline (m) and focal length (pixel)
input_width = 640
camera_config = CameraConfig(0.1, 0.5*input_width) # 90 deg. FOV

# Initialize model
max_distance = 5
depth_estimator = HitNet(model_path, model_type, camera_config, max_distance)

cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	try:
		# Read frame from the video
		ret, frame = cap.read()
		if not ret:	
			break
	except:
		continue

	# Extract the left and right images
	left_img  = frame[:,:frame.shape[1]//3]
	right_img = frame[:,frame.shape[1]//3:frame.shape[1]*2//3]
	color_real_depth = frame[:,frame.shape[1]*2//3:]

	# Estimate the depth
	disparity_map = depth_estimator(left_img, right_img)
	color_depth = depth_estimator.draw_depth()

	combined_image = np.hstack((left_img, color_real_depth, color_depth))

	cv2.imshow("Estimated depth", combined_image)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()