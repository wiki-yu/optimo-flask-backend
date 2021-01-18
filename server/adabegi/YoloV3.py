'''
@ Author: Ines Alcibar
@ Year: 2020
@ Project: AI_builder
@ Version:

@ Description:

'''

import os
import sys
from PIL import Image
import numpy as np


def function(inputs,options):
	frame = inputs["image"]
	frame = Image.fromarray(frame[:, :, ::-1])
	out_pred, _ = options["network"].detect_image(frame, show_stats=False)

	return {"image": np.array(np.asarray(frame)[:, :, ::-1]), "boxes": out_pred}



def load_model(model_name,options,path):
	print('----- Yolo V3 info -----')
	print('model name: ',model_name)
	# Give the configuration and weight files for the model and load the network
	# using them.
	src_path = os.path.join(os.path.dirname(path),"src")
	weights_path = os.path.join(path, str(model_name)+".h5")
	classes_path = os.path.join(path, str(model_name)+".txt")
	anchors_path = os.path.join(src_path,"keras_yolo3","model_data","yolo_anchors.txt")
	try:
		gpu_num = options["gpu_num"]
	except:
		gpu_num = 1
	try:
		score = options["min_score"]
	except:
		score = 0.25
	try:
		plot_boxes = options["plot_boxes"]
	except:
		plot_boxes = True

	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
	sys.path.append(src_path)
	if plot_boxes:
		from keras_yolo3.yolo import YOLO
	else:
		from keras_yolo3.yolo_noReturnedImage import YOLO


	yolo = YOLO(
        **{
            "model_path": weights_path,
            "anchors_path": anchors_path,
            "classes_path": classes_path,
            "score": score,
            "gpu_num": gpu_num,
            "model_image_size": (416, 416),
        }
    )
	class_file = open(classes_path, "r")
	input_labels = [line.rstrip("\n") for line in class_file.readlines()]
	print("Found {} input labels: {} ...".format(len(input_labels), input_labels))

	return yolo