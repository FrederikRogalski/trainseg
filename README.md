# trainseg
Semantic Segmentation for detecting train rails.

## Description
This Project is aims at autonomizing a small model train. Therefore we need to know if the rails ahead are blocked. If they are blocked the train has to stop automaticly.
This is accomplished through the combination of a semantic segmentation model and some other clever algorithms.

## Inputs
The only input sensor is the camera at the front of the train, which is a ESP32 CAM module. It outputs a MJPEG video stream of 320x240@25FPS.
The video stream gets harnessed by a RaspberryPI with an Edge TPU (Google Coral).

## Inference
The Raspberry PI uses the Edge TPU to speed up the Inference of the Machine Learning Modell. In our test we also used an Nvidia Jetson Nano and found out that although the Jetson Nano is more versatile the Edge TPU packs much more power inside of it.
We managed to get a x15 improvement in Inference Speed.

## Training
Not every Model can be used with the Edge TPU. Specifically the model has to be quantized and there are only so many OPs available that the Edge TPU Compiler can map to the TPU.
If you use an unsupported OP-Layer it will be executed on the CPU.
