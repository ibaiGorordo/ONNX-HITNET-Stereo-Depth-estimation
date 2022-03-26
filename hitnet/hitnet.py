from enum import Enum
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime

class ModelType(Enum):
	eth3d = 0
	middlebury = 1
	flyingthings = 2

@dataclass
class CameraConfig:
    baseline: float
    f: float

DEFAULT_CONFIG = CameraConfig(0.546, 120) # rough estimate from the original calibration

class HitNet():

	def __init__(self, model_path, model_type=ModelType.eth3d, camera_config=DEFAULT_CONFIG, max_dist=10):

		# Initialize model
		self.model = self.initialize_model(model_path, model_type, camera_config, max_dist)

	def __call__(self, left_img, right_img):

		return self.update(left_img, right_img)

	def initialize_model(self, model_path, model_type=ModelType.eth3d, camera_config=DEFAULT_CONFIG, max_dist=10):
		
		self.model_type = model_type
		self.camera_config = camera_config
		self.max_dist = max_dist

		# Initialize model session
		self.session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider',
																		   'CPUExecutionProvider'])

		# Get model info
		self.get_input_details()
		self.get_output_details()

	def update(self, left_img, right_img):

		input_tensor = self.prepare_input(left_img, right_img)

		# Perform inference on the image
		if self.model_type == ModelType.flyingthings:
			left_disparity, right_disparity = self.inference(input_tensor)
			self.disparity_map = left_disparity
		else:
			self.disparity_map = self.inference(input_tensor)

		# Estimate depth map from the disparity
		self.depth_map = self.get_depth_from_disparity(self.disparity_map, self.camera_config)

		return self.disparity_map

	def prepare_input(self, left_img, right_img):

		self.img_height, self.img_width = left_img.shape[:2]

		left_img = cv2.resize(left_img, (self.input_width, self.input_height))
		right_img = cv2.resize(right_img, (self.input_width, self.input_height))

		if (self.model_type is ModelType.eth3d):

			# Shape (1, 2, None, None)
			left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
			right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

			left_img = np.expand_dims(left_img,2)
			right_img = np.expand_dims(right_img,2)

			combined_img = np.concatenate((left_img, right_img), axis=-1) / 255.0
		else:
			# Shape (1, 6, None, None)
			left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
			right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

			combined_img = np.concatenate((left_img, right_img), axis=-1) / 255.0

		combined_img = combined_img.transpose(2, 0, 1)

		return np.expand_dims(combined_img, 0).astype(np.float32)

	def inference(self, input_tensor):

		input_name = self.session.get_inputs()[0].name
		left_output_name = self.session.get_outputs()[0].name

		if self.model_type is not ModelType.flyingthings:
			left_disparity = self.session.run([left_output_name], {input_name: input_tensor})
			return np.squeeze(left_disparity)

		right_output_name = self.session.get_outputs()[1].name
		left_disparity, right_disparity = self.session.run([left_output_name, right_output_name], {input_name: input_tensor})
		
		return np.squeeze(left_disparity), np.squeeze(right_disparity)

	@staticmethod
	def get_depth_from_disparity(disparity_map, camera_config):

		return camera_config.f*camera_config.baseline/disparity_map

	def draw_disparity(self):

		disparity_map =  cv2.resize(self.disparity_map,  (self.img_width, self.img_height))
		norm_disparity_map = 255*((disparity_map-np.min(disparity_map))/
								  (np.max(disparity_map)-np.min(disparity_map)))

		return cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map,1), cv2.COLORMAP_MAGMA)

	def draw_depth(self):
		
		return self.util_draw_depth(self.depth_map, (self.img_width, self.img_height), self.max_dist)

	@staticmethod
	def util_draw_depth(depth_map, img_shape, max_dist):

		norm_depth_map = 255*(1-depth_map/max_dist)
		norm_depth_map[norm_depth_map < 0] = 0
		norm_depth_map[norm_depth_map >= 255] = 0

		norm_depth_map =  cv2.resize(norm_depth_map, img_shape)

		return cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map,1), cv2.COLORMAP_MAGMA)

	def get_input_details(self):

		model_inputs = self.session.get_inputs()
		self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

		self.input_shape = model_inputs[0].shape
		self.input_height = self.input_shape[2]
		self.input_width = self.input_shape[3]

	def get_output_details(self):

		model_outputs = self.session.get_outputs()
		self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

		self.output_shape = model_outputs[0].shape

	
if __name__ == '__main__':

	from imread_from_url import imread_from_url
	
	# Select model type
	# model_type = ModelType.middlebury
	# model_type = ModelType.flyingthings
	model_type = ModelType.eth3d

	if model_type == ModelType.middlebury:
		model_path = "../models/middlebury_d400/saved_model_480x640/model_float32.onnx"
	elif model_type == ModelType.flyingthings:
		model_path = "../models/flyingthings_finalpass_xl/saved_model_480x640/model_float32.onnx"
	elif model_type == ModelType.eth3d:
		model_path = "../models/eth3d/saved_model_480x640/model_float32.onnx"

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






