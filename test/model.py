import os
import cv2 as cv

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Use the absolute paths to the YOLO files
cfg_path = os.path.join(script_dir, 'yolov3.cfg')
weights_path = os.path.join(script_dir, 'yolov3.weights')

net = cv.dnn.readNetFromDarknet(cfg_path, weights_path)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

ln = net.getLayerNames()

image = cv.imread('DataSet Info/images/1 (1)t.jpg')

blob = cv.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layer_outputs = net.forward(ln)

print(layer_outputs)
