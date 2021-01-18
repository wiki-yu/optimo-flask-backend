'''
@ Author: Ines Alcibar
@ Year: 2020
@ Project: AI_builder
@ Team: Zazu
@ Version:

@ Description:
'''

#####################################################################################################
#####################################################################################################

# Import config file

import yaml

with open("./src/project_4.yaml", 'r') as stream:
    try:
    	config_file = yaml.safe_load(stream)
    	print(config_file)
    except yaml.YAMLError as exc:
        print(exc)

# # Define the execution method type
class execution_method:
	def __init__(self,**kwargs):
		self.type = kwargs.get("type")
		# self.library = kwargs.get("library")
		# self.algorithm = kwargs.get("algorithm")
		self.save_output = kwargs.get("save_output")
		self.output_vid_name = kwargs.get("output_vid_name")

method1 = execution_method()

for att in config_file["method"]:
	setattr(method1, str(att), config_file["method"][str(att)])

source = "C:\\Users\\ines.alcibar\\Desktop\\CaProj\\InferenceYOLO\\test_video.MOV"
source = "C:\\Users\\wiki\Desktop\\Github\\optimo-express-server\\app-aibuilder\\data\\demo.mp4"
source = "C:\\Users\\Xuyong Yu\\Desktop\\Github\\optimo-express-server\\app-aibuilder\\data\\demo0.mp4"

# source = "C:\\Users\\ines.alcibar\\Desktop\\CaProj\\InferenceYOLO\\output_boxes_test_video2.mp4"
# source = "C:\\Users\\ines.alcibar\\Desktop\\CaProj\\yoloface\\samples\\test.MOV"


# Define the order of the blocks
block_order = []
input_name = [config_file["input_elements"]]
output_name = config_file["output_elements"]


while True:
	# Go through all the blocks
	for block in config_file["block_definitions"]:
		# If block not already used
		if not block in block_order:
			# Check that all the inputs have already appeared as an output of annother block
			# print(config_file["block_definitions"][str(block)]["input_elements"].values())
			if all(item in input_name for item in config_file["block_definitions"][str(block)]["input_elements"].values()) :
				# Add the block into the order list
				block_order.append(block)
				# Add the outputs of the current block to the available inputs
				input_name.extend(config_file["block_definitions"][str(block)]["output_elements"].values())
				# print(block)
				# print(input_name)

	# When the final output is achieved, get out of the loop
	if output_name in input_name:
		break

print("Block order: ",block_order)

variable_values = {key: None for key in input_name}
print("Variable names initialized: ",variable_values)


# Define the objects from the config data (in the corresponding order)
# Save all the objects in the same variable: block_chain

class block:
	def __init__(self,**kwargs):
		self.adabegi = kwargs.get("adabegi")
		self.input_elements = kwargs.get("input_elements")
		self.ouput_elements = kwargs.get("ouput_elements")
		self.block_options = kwargs.get("block_options")
		self.model_name = kwargs.get("model_name")
		self.model_options = kwargs.get("model_options")
		self.initialize_variables = kwargs.get("initialize_variables")
		self.finalize = kwargs.get("finalize")

block_chain = []

for i in range(len(block_order)):
	block1 = block()
	for att in config_file["block_definitions"][str(block_order[i])]:
		setattr(block1, str(att), config_file["block_definitions"][str(block_order[i])][str(att)])

	block_chain.append(block1)



import numpy as np
import cv2
import os
import sys
import importlib



def get_parent_dir(n=1):
    """
    returns the n-th parent dicrectory of the current
    working directory
    """

    #Example:
    #n=3

    #This is the file called Detector.py
    #It's full path is: C:\Users\ines.alcibar\Desktop\Project\YOLO\Detector.py

    #current_path = os.path.dirname(os.path.abspath(__file__))
    #--> current_path = C:\\Users\\ines.alcibar\\Desktop\\Project\\YOLO

    #k = 1
    #current_path = os.path.dirname(current_path)
    #--> current_path = C:\\Users\\ines.alcibar\\Desktop\\Project

    #k = 2
    #current_path = os.path.dirname(current_path)
    #--> current_path = C:\\Users\\ines.alcibar\\Desktop

    #k = 3
    #current_path = os.path.dirname(current_path)
    #--> current_path = C:\\Users\\ines.alcibar
    

    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

actual_path = get_parent_dir(1) # Path where this file is saved
adabegi_path = os.path.join(actual_path, "adabegi")
print(adabegi_path)
sys.path.append(adabegi_path)
models_path = os.path.join(actual_path, "trained_models")
print(models_path)
sys.path.append(models_path)


###########################################################################################################
#####################                                                                 #####################
#####################                 Import blocks as modules                        #####################
#####################                                                                 #####################
###########################################################################################################
# Import the modules from adabegi
print("number of chained blocks: ",len(block_chain))
b_names = []
b_count = len(block_chain)

for i in range(b_count):
	bb_name = "bb" +str(i)
	globals()[bb_name]=block_chain[i].adabegi
	# print(globals()[bb_name])
	b_name = "b" +str(i)
	# print(b_name)
	b_names.append(b_name)
	globals()[b_name] = importlib.import_module(globals()[bb_name])
	# print(globals()[b_name])


###########################################################################################################
#####################                                                                 #####################
#####################                 Import necessary models                         #####################
#####################                                                                 #####################
###########################################################################################################
# Import the models from trained_models

for i in range(b_count):
	if block_chain[i].model_name!=None:
		m_name = "m" + str(i)
		globals()[m_name] = globals()[b_names[i]].load_model(block_chain[i].model_name,block_chain[i].model_options,models_path)
		try:
			block_chain[i].block_options["network"]=globals()[m_name]
		except:
			block_chain[i].block_options = {"network":globals()[m_name]}
		
###########################################################################################################
#####################                                                                 #####################
#####################                 Initialize necessary variables                  #####################
#####################                                                                 #####################
###########################################################################################################
# Initialize necessary variables

for i in range(b_count):
    if block_chain[i].initialize_variables!=None:
        new_var = globals()[b_names[i]].initialize_variables(block_chain[i].initialize_variables)
        # print(new_var)
        for key in new_var['new_values'].keys():
            # if not key in variable_values.keys():
                variable_values[str(key)] = new_var['new_values'][str(key)]
        for key in new_var['new_inputs'].keys():
        	block_chain[i].input_elements[str(key)]=new_var['new_inputs'][str(key)]
        for key in new_var['new_outputs'].keys():
        	block_chain[i].output_elements[str(key)]=new_var['new_outputs'][str(key)]

            # else:
                # raise AttributeError("Couldn't initialize variable "+str(key)+" from "
                	# +str(block_chain[i].adabegi)+". Please, modify the stream name with the same name.")

# print("variable_values: ", variable_values)

###########################################################################################################
#####################                                                                 #####################
#####################                            Define Threads                       #####################
#####################                                                                 #####################
###########################################################################################################
# IDefine different Threads
import threading
from threading import Thread
class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0,lock=None):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):    
        Thread(target=self.get, args=(lock,)).start()
        return self

    def get(self,lock):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                # lock.acquire()

    def stop(self):
        self.stopped = True
        self.stream.release()
        cv2.destroyAllWindows()

class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            cv2.imshow("Video", self.frame)
            # lock.release()
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True

from timeit import default_timer as timer
accum_time = 0
curr_fps = 0
fps = "FPS: ??"
prev_time = timer()


cap = cv2.VideoCapture(source)
# lock = threading.Lock()

if not cap.isOpened():
    raise IOError("Couldn't open webcam or video")

if method1.save_output:

	if method1.output_vid_name!=None:
		output_path = os.path.join(actual_path, "outputs", str(method1.output_vid_name)+'.mp4')
	else:
		output_path = os.path.join(actual_path, "outputs", 'output_video.mp4')


	# Get output video ready and obtain characteristics
	video_FourCC = cv2.VideoWriter_fourcc('H','2','6','4') 
	video_fps = cap.get(cv2.CAP_PROP_FPS)
	video_size = (
	    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
	    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
	)

	print("Processing {} with frame size {} at {:.1f} FPS".format(
			os.path.basename(source), video_size, video_fps)    )

	out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

# video_getter = VideoGet(source,lock).start()
# video_shower = VideoShow(video_getter.frame).start()
cnt = 0
while(True):
    # Capture frame-by-frame
	ret, frame = cap.read()
	if not ret:
		break
	cnt+=1
	# print('frame: ', cnt)
	# if video_getter.stopped or video_shower.stopped:
	# 	video_shower.stop()
	# 	video_getter.stop()
	# 	break
	# lock.acquire()
	# frame = video_getter.frame


	# Our operations on the frame come here
	# input_frame = frame.copy()
	# for i in range(b_count):
	# 	input_frame = globals()[b_names[i]].function(input_frame,block_chain[i].block_options)
	variable_values["input_frame"] = frame.copy() 
	for i in range(len(block_chain)):
		inputs = {key: variable_values[str(block_chain[i].input_elements[str(key)])] for key in block_chain[i].input_elements.keys()}
		# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ input.keys: ', inputs.keys())
		# print(inputs)
		outputs = globals()[b_names[i]].function(inputs,block_chain[i].block_options)
		# print(outputs.keys())
		for key in outputs.keys():
			# print('keeeeyyyy: ',key)
			# print(block_chain[i].output_elements)
			# print('var name: ', str(block_chain[i].output_elements[str(key)]))
			variable_values[str(block_chain[i].output_elements[str(key)])] = outputs[str(key)]
	
	# print("variable_values: ", variable_values)
	curr_time = timer()
	exec_time = curr_time - prev_time
	prev_time = curr_time
	accum_time = accum_time + exec_time
	curr_fps = curr_fps + 1
	if accum_time > 1:
		accum_time = accum_time - 1
		fps = "FPS: " + str(curr_fps)
		curr_fps = 0
	# cv2.putText(
	# 	variable_values["output_frame"],
	# 	text=fps,
 #        org=(3, 15),
 #        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
 #        fontScale=0.50,
 #        color=(255, 0, 0),
 #        thickness=2,
 #    )
	# video_shower.frame = variable_values["output_frame"] 
	# lock.release()
    # Display the resulting frame
	# cv2.imshow('frame',variable_values["output_frame"])
	if cv2.waitKey(1)>0:
		break

	if method1.save_output:
		out.write(variable_values["output_frame"])

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
print('HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')

if method1.save_output:
	out.release()

# Save required files
for i in range(b_count):
	if block_chain[i].finalize!=None:
		inputs = {key: variable_values[str(block_chain[i].input_elements[str(key)])] for key in block_chain[i].input_elements.keys()}
		globals()[b_names[i]].finalize(inputs, block_chain[i].finalize)
        
