import cv2
import depthai as dai
import numpy as np

from hitnet import HitNet, ModelType, CameraConfig

def create_pipeline():

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    rect_left = pipeline.create(dai.node.XLinkOut)
    rect_right = pipeline.create(dai.node.XLinkOut)

    rect_left.setStreamName("rect_left")
    rect_right.setStreamName("rect_right")

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # StereoDepth
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    stereo.rectifiedLeft.link(rect_left.input)
    stereo.rectifiedRight.link(rect_right.input)

    return pipeline


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

# Store baseline (m) and focal length (pixel) for OAK-D Lite
# Ref: https://docs.luxonis.com/en/latest/pages/faq/#how-do-i-calculate-depth-from-disparity
# TODO: Modify values corrsponding with the actual board info
input_width = 640
camera_config = CameraConfig(0.075, 0.5*input_width/0.72) # 71.9 deg. FOV 
max_distance = 5

# Initialize model
hitnet_depth = HitNet(model_path, model_type, camera_config, max_distance)

# Get Depthai pipeline
pipeline = create_pipeline()

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    rectified_left_queue = device.getOutputQueue(name="rect_left", maxSize=4, blocking=False)
    rectified_right_queue = device.getOutputQueue(name="rect_right", maxSize=4, blocking=False)

    while True:
        in_left_rect = rectified_left_queue.get()
        in_right_rect = rectified_right_queue.get()

        left_rect_img = in_left_rect.getCvFrame()
        right_rect_img = in_right_rect.getCvFrame()

        left_rect_img = cv2.cvtColor(left_rect_img, cv2.COLOR_GRAY2BGR)
        right_rect_img = cv2.cvtColor(right_rect_img, cv2.COLOR_GRAY2BGR)

        # Estimate the depth
        disparity_map = hitnet_depth(left_rect_img, right_rect_img)
        color_depth = hitnet_depth.draw_depth()

        combined_image = np.hstack((left_rect_img, color_depth))
        cv2.imwrite("output.jpg", combined_image)

        cv2.imshow("Estimated depth", combined_image)

        if cv2.waitKey(1) == ord('q'):
            break