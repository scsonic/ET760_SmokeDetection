# NASA2019 Team ET760
Smoke Detection Model

## Mission
Use the deep learning YOLO framework to complete the smoke detection model.

## Setup
Clone the darknet framework: `git clone https://github.com/pjreddie/darknet`

Then create smoke.cfg, smoke.data, and smoke.names into the local folder


## Data set
We label 373 images in VOC format. Then, we randomly select 90% as training set and 10% as valid set.
![](https://i.imgur.com/ulIzdsi.png)

## Training
Train the model using the command: `./darknet detector train cfg/smoke.data cfg/smoke.cfg darknet53.conv.74`

## Result
Run the model with OpenCV 
```
self.net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
…
outs = self.net.forward(self.getOutputsNames(self.net))
indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
cv2.rectangle(self.frame, (left, top), (right, bottom), paint)
```
![](https://i.imgur.com/IwM4p8c.png)



