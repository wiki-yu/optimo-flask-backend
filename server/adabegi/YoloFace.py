'''
# Author: Ines Alcibar
# Year: 2020
# Project: AI Builder
# Team: Zazu
'''

import datetime
import numpy as np
import cv2
import itertools
import os
# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------

CONF_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

# Default colors
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)


# -------------------------------------------------------------------
# Help functions
# -------------------------------------------------------------------

# Get the names of the output layers
def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected
    # outputs
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def draw_predict(frame, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)

    text = '{:.2f}'.format(conf)

    # Display the label at the top of the bounding box
    label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    top = max(top, label_size[1])
    cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                COLOR_WHITE, 1)

def post_process(frame, outs, conf_threshold, nms_threshold):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        final_boxes.append(box)
        #frame[top:top+height,left:left+width] = cv2.blur(frame[top:top+height,left:left+width] ,(23,23))
        left, top, right, bottom = refined_box(left, top, width, height)
        # draw_predict(frame, confidences[i], left, top, left + width,
        #              top + height)
        
        print(left, top, right, bottom)
        left = max(0,left-15)
        top = max(0,top-15)
        right = max(0,right+15)
        bottom = max(0,bottom+15)
        frame[top:bottom,left:right] = cv2.blur(frame[top:bottom,left:right],(30,30))
        #draw_predict(frame, confidences[i], left, top, right, bottom)
    #     CoGx = (left+right)/2
    #     CoGy = (top+bottom)/2
    #     cnt =0
    #     for j in range(len(hist_CoG)):
    #         #if not hist_CoG[j][-1]==-1:
    #         #print(frame.shape)
    #         #print(hist_CoG[j][4])
    #         if ((CoGx<(hist_CoG[j][4]+frame.shape[1]*0.05))&(CoGx>(hist_CoG[j][4]-frame.shape[1]*0.05))&(CoGy<(hist_CoG[j][5]+frame.shape[0]*0.05))&(CoGy>(hist_CoG[j][5]-frame.shape[0]*0.05))):
    #             #print('CoGx',CoGx)
    #             #print('CoGy',CoGy)
    #             #print('(hist_CoG[j][4]+frame.shape[1]*0.05)',(hist_CoG[j][4]+frame.shape[1]*0.05))
    #             #print('(hist_CoG[j][4]-frame.shape[1]*0.05)',(hist_CoG[j][4]-frame.shape[1]*0.05))
    #             hist_CoG[j]=[int(left), int(top), int(right), int(bottom),int(CoGx),int(CoGy),-1]
    #             #print('-1')
    #             cnt+=1
    #     if cnt==0:
    #         hist_CoG.append([int(left), int(top), int(right), int(bottom),int(CoGx),int(CoGy),-1])
    #         #print("append")
    #     if len(hist_CoG)<1:
    #         hist_CoG.append([int(left), int(top), int(right), int(bottom),int(CoGx),int(CoGy),-1])
    #         print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # to_remove = []
    # for j in range(len(hist_CoG)):
    #     if hist_CoG[j][-1]==-1:
    #         hist_CoG[j][-1]=0
    #     else:
    #         frame[hist_CoG[j][1]:hist_CoG[j][3],hist_CoG[j][0]:hist_CoG[j][2]] = cv2.blur(frame[hist_CoG[j][1]:hist_CoG[j][3],hist_CoG[j][0]:hist_CoG[j][2]],(30,30))
    #         hist_CoG[j][-1]+=1
    #         if hist_CoG[j][-1]>2:
    #             to_remove.extend([j])
    # for index in sorted(to_remove, reverse=True):
    #     del hist_CoG[index]
    #     #print('deleted')
    # #print(hist_CoG)
    # if len(hist_CoG)>1:
    #     hist_CoG.sort()
    #     hist_CoG = list(k for k,_ in itertools.groupby(hist_CoG))
    # #print(hist_CoG)
    # #assert len(hist_CoG)<5
    return final_boxes


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._num_frames = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._num_frames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._num_frames / self.elapsed()

def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height

    original_vert_height = bottom - top
    top = int(top + original_vert_height * 0.15)
    bottom = int(bottom - original_vert_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

    right = right + margin

    return left, top, right, bottom



def function(inputs,options):
	image = inputs["image"]
	# Create a 4D blob from a frame
	blob = cv2.dnn.blobFromImage(image, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),[0, 0, 0], 1, crop=False)

	# Sets the input to the network
	# print("GLOBALS: ", globals())
	# net = globals()[options["network_name"]]
	options["network"].setInput(blob)

    # Runs the forward pass to get output of the output layers
	outs = options["network"].forward(get_outputs_names(options["network"]))
	# detections = options["network"].forward()
	# print(detections.shape)
 #    # loop over the detections
	# for i in np.arange(0, detections.shape[2]):
	# 	# extract the confidence (i.e., probability) associated with
	# 	# the prediction
	# 	confidence = detections[0, 0, i, 2]
	# 	# filter out weak detections by ensuring the `confidence` is
	# 	# greater than the minimum confidence
	# 	if confidence > conf_threshold:
	# 		# extract the index of the class label from the
	# 		# `detections`, then compute the (x, y)-coordinates of
	# 		# the bounding box for the object
	# 		idx = int(detections[0, 0, i, 1])
	# 		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
	# 		(startX, startY, endX, endY) = box.astype("int")
	# 		# draw the prediction on the frame
	# 		cv2.rectangle(frame, (startX, startY), (endX, endY),
	# 			COLORS[idx], 2)

	faces = post_process(image, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        
    # initialize the set of information we'll displaying on the frame
	info = [
        ('number of faces detected', '{}'.format(len(faces)))
    ]

	for (i, (txt, val)) in enumerate(info):
		text = '{}: {}'.format(txt, val)
		cv2.putText(image, text, (10, (i * 20) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
	return {"image":image,"boxes":[100,100,300,300]}

def load_model(model_name,options,path):
	print('----- yolo face info -----')
	print('model name: ',model_name)
	# Give the configuration and weight files for the model and load the network
	# using them.
	cfg_path = os.path.join(path, str(model_name)+".cfg")
	weights_path = os.path.join(path, str(model_name)+".weights")
	net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
	return net