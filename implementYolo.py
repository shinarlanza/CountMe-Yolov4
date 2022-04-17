import cv2
import numpy as np
import matplotlib.pyplot as plt
### loading the model files
net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')


### reading the whole classes can be detected

with open('coco.names','r') as f:
    classes = ([line.strip() for line in f.readlines()])
len(classes) #here 80 classes can be detected using yolo
80
print([i for i in net.getUnconnectedOutLayers()]) # outptut layer  is 77th layer 

# defining the input and output layer in yolo
layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] # this is the output detection layer
output_layers = net.getUnconnectedOutLayersNames()
# these are 2 methods to find the last layers for detection
print(output_layers)
['yolo_82', 'yolo_94', 'yolo_106']
#### Loading the image

img = cv2.imread('office.jpg')
# img = cv2.resize(img,(500,500))
height, width , channel = img.shape
### detecting the objects and making the input to yolo
blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),  swapRB = True , crop = True)

# True refers to converting into rgb format since opencv uses bgr.

net.setInput(blob)  

# Passing blob image to yolo algo in network

outs = net.forward(output_layers)  

# Giving network to output layer for final result.
net

outs[0].shape
(507, 85)
outs[0][0]

class_ids = []
confidences = []
boxes = []

for output in outs:
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > 0.5:
                 # Object detected

            center_x = int(detection[0] * width)

            center_y = int(detection[1] * height)

            w = int(detection[2] * width)

            h = int(detection[3] * height)

            # Rectangle coordinates

            x = int(center_x - w / 2)

            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])

            confidences.append(float(confidence))

            class_ids.append(class_id)
        
img = img.copy()
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) 
colors = np.random.uniform(0, 255, size=(len(classes), 3))
# NMS - non max supression

#print(indexes)

font = cv2.FONT_HERSHEY_PLAIN

for i in range(len(boxes)):

    if i in indexes:

        x, y, w, h = boxes[i]

        label = str(classes[class_ids[i]])

        color = colors[i]

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2) 

        # Draw rectangle around boxes. '2' is the width of box.

        cv2.putText(img, label, (x, y + 30), font, 2, color, 2)

        # Text in Box to label the object
cv2.imshow("Image", img) 

cv2.waitKey(0) 

 # waitkey stops the output

cv2.destroyAllWindows()