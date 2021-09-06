# ONNX-HITNET-Stereo-Depth-estimation
Python scripts form performing stereo depth estimation using the HITNET model in ONNX.

![Hitnet stereo depth estimation ONNX](https://github.com/ibaiGorordo/ONNX-HITNET-Stereo-Depth-estimation/blob/main/doc/img/out.jpg)
*Stereo depth estimation on the cones images from the Middlebury dataset (https://vision.middlebury.edu/stereo/data/scenes2003/)*

# Requirements

 * **OpenCV**, **imread-from-url**, **onnx** and **onnxruntime**. Also, **pafy** and **youtube-dl** are required for youtube video inference. 
 
# Installation
```
pip install -r requirements.txt
pip install pafy youtube-dl
```

# ONNX model
The original models were converted to different formats (including .onnx) by [PINTO0309](https://github.com/PINTO0309), download the models from [his repository](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/142_HITNET) and save them into the **[models](https://github.com/ibaiGorordo/ONNX-HITNET-Stereo-Depth-estimation/tree/main/models)** folder. 

# Original Tensorflow model
The Tensorflow pretrained model was taken from the [original repository](https://github.com/google-research/google-research/tree/master/hitnet).
 
# Examples

 * **Image inference**:
 
 ```
 python imageDepthEstimation.py 
 ```
 
  * **Video inference**:
 
 ```
 python videoDepthEstimation.py
 ```
 
 * **DrivingStereo dataset inference**:
 
 ```
 python drivingStereoTest.py
 ```
 

# Pytorch inference
For performing the inference in Tensorflow, check my other repository **[HITNET Stereo Depth estimation](https://github.com/ibaiGorordo/HITNET-Stereo-Depth-estimation)**.

# TFLite inference
For performing the inference in TFLite, check my other repository **[TFLite HITNET Stereo Depth estimation](https://github.com/ibaiGorordo/TFLite-HITNET-Stereo-depth-estimation)**.

# [Inference video Example](https://youtu.be/BRQ_oaCRj3M) 
 ![Hitnet stereo depth estimation ONNX](https://github.com/ibaiGorordo/ONNX-HITNET-Stereo-Depth-estimation/blob/main/doc/img/onnxHitnetDepthEstimation.gif)

# References:
* Hitnet model: https://github.com/google-research/google-research/tree/master/hitnet
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* PINTO0309's model conversion tool: https://github.com/PINTO0309/openvino2tensorflow
* DrivingStereo dataset: https://drivingstereo-dataset.github.io/
* Original paper: https://arxiv.org/abs/2007.12140
 

