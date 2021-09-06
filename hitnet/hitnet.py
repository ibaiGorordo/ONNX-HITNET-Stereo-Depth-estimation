import time
import cv2
import numpy as np
import onnx
import onnxruntime

from hitnet.utils_hitnet import *

drivingStereo_config = CameraConfig(0.546, 1000)

class HitNet():

	def __init__(self, model_path, model_type=ModelType.eth3d, camera_config=drivingStereo_config):

		self.fps = 0
		self.timeLastPrediction = time.time()
		self.frameCounter = 0

		self.model_type = model_type
		self.camera_config = camera_config

		# Initialize model
		self.model = self.initialize_model(model_path)

	def __call__(self, left_img, right_img):

		return self.estimate_disparity(left_img, right_img)

	def initialize_model(self, model_path):

		self.session = onnxruntime.InferenceSession(model_path)

		# Get model info
		self.getModel_input_details()
		self.getModel_output_details()

	def estimate_disparity(self, left_img, right_img):

		# Update fps calculator
		self.updateFps()

		input_tensor = self.prepare_input(left_img, right_img)

		# Perform inference on the image
		if self.model_type == ModelType.flyingthings:
			left_disparity, right_disparity = self.inference(input_tensor)
			self.disparity_map = left_disparity
		else:
			self.disparity_map = self.inference(input_tensor)

		return self.disparity_map

	def get_depth(self):
		return self.camera_config.f*self.camera_config.baseline/self.disparity_map

	def prepare_input(self, left_img, right_img):

		left_img = cv2.resize(left_img, (self.input_width, self.input_height))
		right_img = cv2.resize(right_img, (self.input_width, self.input_height))

		if (self.model_type == ModelType.eth3d):

			# Shape (1, None, None, 2)
			left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
			right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

			left_img = np.expand_dims(left_img,2)
			right_img = np.expand_dims(right_img,2)

			combined_img = np.concatenate((left_img, right_img), axis=-1) / 255.0
		else:
			# Shape (1, None, None, 6)
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
		output = self.session.run([left_output_name, right_output_name], {input_name: input_tensor})

		return np.squeeze(left_disparity), np.squeeze(right_disparity)

	def updateFps(self):
		updateRate = 1
		self.frameCounter += 1

		# Every updateRate frames calculate the fps based on the ellapsed time
		if self.frameCounter == updateRate:
			timeNow = time.time()
			ellapsedTime = timeNow - self.timeLastPrediction

			self.fps = int(updateRate/ellapsedTime)
			self.frameCounter = 0
			self.timeLastPrediction = timeNow

	def getModel_input_details(self):

		self.input_shape = self.session.get_inputs()[0].shape
		self.channes = self.input_shape[2]
		self.input_height = self.input_shape[2]
		self.input_width = self.input_shape[3]

	def getModel_output_details(self):

		self.output_shape = self.session.get_outputs()[0].shape

	






