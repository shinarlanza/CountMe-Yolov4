### here we are going to use euclidean tracker but its also not necessary 
# centroid tracker is more accurate 
import cv2
import numpy as np
import time
from itertools import combinations

# from centroidtracker import CentroidTracker
from trackPerson import EuclideanDistTracker
tracker = EuclideanDistTracker()
from non_max_suppression import non_max_suppression_fast

# tracker = CentroidTracker(maxDisappeared=20, maxDistance=90)


net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")  
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines() ]

output_layers = net.getUnconnectedOutLayersNames()


cap = cv2.VideoCapture('video.mp4')

# for calcualting the fps 
starting_time = time.time()
frame_num = 0
centroid_dict = dict()
while cap.isOpened():
    centroid_dict = dict()

    rects = []
    ret, frame = cap.read()
    frame_num +=1
    height, width, channel = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  
    net.setInput(blob)  
    outs = net.forward(output_layers)
    #----------------------------------------------
    class_ids = []

    confidences = []

    boxes = []

    
    for out in outs:

        for detection in out:

            scores = detection[5:]

            class_id = np.argmax(scores)

            confidence = scores[class_id]

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
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) 
    frame = frame.copy()

    # NMS - non max supression

    #print(indexes)
    centre  = list()
    rects = list()
    font = cv2.FONT_HERSHEY_PLAIN
    if len(indexes) > 0:

        
        idf = indexes.flatten()
       
        for i in idf:
            (x,y) = (boxes[i][0], boxes[i][1])
            (w,h) = (boxes[i][2], boxes[i][3])
        

            label = str(classes[class_ids[i]])
            if label == 'person':
                centre.append([int(x+w/2),int(y+h/2)])
                # (startX,startY, endX,endY) = (x,y,x+w,y+h)
                rects.append((x,y,w,h))
        boundingboxes = np.array(rects)
        # boundingboxes = boundingboxes.astype(int)
        # rects = non_max_suppression_fast(boundingboxes, 0.3)

        boxes_ids = tracker.update(rects)
        # for (objectId, bbox) in objects.items():
        #     x1, y1, x2, y2 = bbox
        #     x1 = int(x1)
        #     y1 = int(y1)
        #     x2 = int(x2)
        #     y2 = int(y2)
        for box_id in boxes_ids:
            (x,y,w,h,id)  = box_id
            Cx = int((x+x+w)/2)
            Cy = int((y+y+h)/2)
            centroid_dict[id] = (Cx,Cy,x,y,w,h)
        red_zone_dict = dict()    
        for (id1, pt1),(id2,pt2) in combinations(centroid_dict.items(),2):
            dx,dy  = pt1[0]-pt2[0], pt1[1]-pt2[1]
            center_pair = [(pt1[0],pt1[1]),(pt2[0],pt2[1])]
            # now using distance formula 
            distance = np.sqrt(dx**2 + dy**2)
            #### here this threshold values pixel value(70) depends on the camera perscpective 

            if distance < 70:
                if id1 not in red_zone_dict:
                    red_zone_dict[id1] = center_pair
                if id2 not in red_zone_dict:
                    red_zone_dict[id2] = center_pair

        for id, box in centroid_dict.items():
            if id in red_zone_dict:
                center_pair = red_zone_dict[id]
                cv2.rectangle(frame, (box[2], box[3]), (box[4]+box[2],box[5]+box[3]), (0, 0, 255), 2)
                cv2.line(frame,center_pair[0], center_pair[1],(0,0,255),2)

            else:
                cv2.rectangle(frame, (box[2], box[3]), (box[4]+box[2],box[5]+box[3]), (0, 255,0 ), 2)
            # creating a summary transparent window 
            height, width , channel = frame.shape
            sub_img = frame[0:int(height/6),0:int(width)]

            black_rect = np.ones(sub_img.shape, dtype=np.uint8)*0


            res = cv2.addWeighted(sub_img, 0.77, black_rect,0.23, 0)
            FONT = cv2.FONT_HERSHEY_SIMPLEX
            FONT_SCALE = 0.8
            FONT_THICKNESS = 2
            lable_color = (255, 255, 255)
            lable = "Social Distancing Detection - During COVID19 "
            lable_dimension = cv2.getTextSize(lable,FONT ,FONT_SCALE,FONT_THICKNESS)[0]
            textX = int((res.shape[1] - lable_dimension[0]) / 2)
            textY = int((res.shape[0] + lable_dimension[1]) / 2)
            cv2.putText(res, lable, (textX,textY), FONT, FONT_SCALE, lable_color, FONT_THICKNESS)
            cv2.putText(res, "Persons:{}".format(len(centre)), (0,textY+22+5), FONT,0.7, lable_color,2)
            cv2.putText(res, "Violation:{}".format(len(red_zone_dict)), (0,textY+22+10), FONT,0.7, lable_color,2)


            frame[0:int(height/6),0:int(width)] =res      

            # cv2.circle(frame, (Cx,Cy), 5,(0,255,0),-1)

            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # text = "ID: {}".format(id)
            # cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)  

                # cv2.circle(frame, (int(x+w/2),int(y+h/2)), 3,(0,0,255), -1)
                # cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            elapsed_time = time.time() - starting_time
            fps = frame_num/elapsed_time
            cv2.putText(frame,"FPS:"+str(np.round(fps,2)),(0,25),cv2.FONT_HERSHEY_COMPLEX,0.7,lable_color,2)


    cv2.imshow('window', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows() 