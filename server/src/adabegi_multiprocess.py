'''
@Author: Ines Alcibar
@Year: 2020
@Project: AI_builder
@Team: Zazu
'''

#####################################################################################################
#####################################################################################################

# Import config file

import yaml

with open("project_4.yaml", 'r') as stream:
    try:
    	config_file = yaml.safe_load(stream)
    	print(config_file)
    except yaml.YAMLError as exc:
        print(exc)

# # Define the execution method type
# class execution_method:
# 	def __init__(self,**kwargs):
# 		self.type = kwargs.get("type")
# 		self.library = kwargs.get("library")
# 		self.algorithm = kwargs.get("algorithm")
# 		self.model_name = kwargs.get("model_name")

# method1


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
			if all(item in input_name for item in config_file["block_definitions"][str(block)]["input_elements"].values()) :
				# Add the block into the order list
				block_order.append(block)
				# Add the outputs of the current block to the available inputs
				input_name.extend(config_file["block_definitions"][str(block)]["output_elements"].values())

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

block_chain = []

for i in range(len(block_order)):
	block1 = block()
	for att in config_file["block_definitions"][str(block_order[i])]:
		setattr(block1, str(att), config_file["block_definitions"][str(block_order[i])][str(att)])

	block_chain.append(block1)


# input_elements = "input_frame"
# output_elements = "output_frame"


# block1 = block(
#         **{
#         "adabegi": "ImageGray",
#         "input_elements": ["input_frame"],
#         "ouput_elements": ["gray_img"],
#     }
# )	

# block_chain.append(block1)

# print(block1.adabegi)
# block2 = block(
#         **{
#         "adabegi": "ImageCanny",
#         "input_elements": ["gray_img"],
#         "ouput_elements": ["output_frame"],
#     }
# )
# block_chain.append(block2)

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
	b_names.append(b_name)
	globals()[b_name] = importlib.import_module(globals()[bb_name])
	# print(globals()[b_name])


###########################################################################################################
#####################                                                                 #####################
#####################                 Import necessary models                         #####################
#####################                                                                 #####################
###########################################################################################################
# Import the modules from trained_models

for i in range(b_count):
	if block_chain[i].model_name!=None:
		m_name = "m" + str(i)
		globals()[m_name] = globals()[b_names[i]].load_model(block_chain[i].model_name,block_chain[i].model_options,models_path)
		try:
			block_chain[i].block_options["network"]=globals()[m_name]
		except:
			block_chain[i].block_options = {"network":globals()[m_name]}
		

from threading import Thread
class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

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
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True

from timeit import default_timer as timer
accum_time = 0
curr_fps = 0
fps = "FPS: ??"
prev_time = timer()


source = "C:\\Users\\ines.alcibar\\Desktop\\CaProj\\InferenceYOLO\\output_boxes_test_video2.mp4"
# source = "C:\\Users\\ines.alcibar\\Desktop\\CaProj\\yoloface\\samples\\test.MOV"
#cap = cv2.VideoCapture(source)

# video_getter = VideoGet(source).start()
# video_shower = VideoShow(video_getter.frame).start()

# while(True):
#     # Capture frame-by-frame
#     # ret, frame = cap.read()

# 	if video_getter.stopped or video_shower.stopped:
# 		video_shower.stop()
# 		video_getter.stop()
# 		break

# 	frame = video_getter.frame

# 	# Our operations on the frame come here
# 	# input_frame = frame.copy()
# 	# for i in range(b_count):
# 	# 	input_frame = globals()[b_names[i]].function(input_frame,block_chain[i].block_options)
# 	variable_values["input_frame"] = frame.copy() 
# 	for i in range(len(block_chain)):
# 		inputs = {key: variable_values[str(block_chain[i].input_elements[str(key)])] for key in block_chain[i].input_elements.keys()}
# 		# print(inputs)
# 		outputs = globals()[b_names[i]].function(inputs,block_chain[i].block_options)
# 		# print(outputs.keys())
# 		for key in outputs.keys():
# 			# print('keeeeyyyy: ',key)
# 			variable_values[str(block_chain[i].output_elements[str(key)])] = outputs[str(key)]

# 	curr_time = timer()
# 	exec_time = curr_time - prev_time
# 	prev_time = curr_time
# 	accum_time = accum_time + exec_time
# 	curr_fps = curr_fps + 1
# 	if accum_time > 1:
# 		accum_time = accum_time - 1
# 		fps = "FPS: " + str(curr_fps)
# 		curr_fps = 0
# 	cv2.putText(
# 		variable_values["output_frame"],
# 		text=fps,
#         org=(3, 15),
#         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#         fontScale=0.50,
#         color=(255, 0, 0),
#         thickness=2,
#     )
# 	video_shower.frame = variable_values["output_frame"] 
#     # Display the resulting frame
#     # cv2.imshow('frame',input_frame)
# 	# if cv2.waitKey(1) & 0xFF == ord('q'):
# 	# 	break

# # When everything done, release the capture
# # cap.release()
# # cv2.destroyAllWindows()

# https://stackoverflow.com/questions/42284122/opencv-python-multi-threading-for-live-facial-recognition

import numpy as np
import cv2

from multiprocessing import Process, Queue
import time

#from common import clock, draw_str, StatValue
#import video

class Canny_Process(Process):
    
    def __init__(self,frame_queue,output_queue):
        Process.__init__(self)
        self.frame_queue = frame_queue
        self.output_queue = output_queue
        self.stop = False
        #Initialize your face detectors here
        

    def get_frame(self):
        if not self.frame_queue.empty():
            return True, self.frame_queue.get()
        else:
            return False, None

    def stopProcess(self):
        self.stop = True
            
    def canny_frame(self,frame):
        # some intensive computation...
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # edges = cv2.Canny(gray, 50, 100)
        
        # #To simulate CPU Time
        # #############################
        # for i in range(1000000):
        #     x = 546*546
        #     res = x/(i+1)
        # #############################
        # 'REPLACE WITH FACE DETECT CODE HERE'

        variable_values["input_frame"] = frame.copy() 
        for i in range(len(block_chain)):
            inputs = {key: variable_values[str(block_chain[i].input_elements[str(key)])] for key in block_chain[i].input_elements.keys()}
            # print(inputs)
            outputs = globals()[b_names[i]].function(inputs,block_chain[i].block_options)
            # print(outputs.keys())
            for key in outputs.keys():
                # print('keeeeyyyy: ',key)
                variable_values[str(block_chain[i].output_elements[str(key)])] = outputs[str(key)]

        # curr_time = timer()
        # exec_time = curr_time - prev_time
        # prev_time = curr_time
        # accum_time = accum_time + exec_time
        # curr_fps = curr_fps + 1
        # if accum_time > 1:
        #     accum_time = accum_time - 1
        #     fps = "FPS: " + str(curr_fps)
        #     curr_fps = 0
        # cv2.putText(
        #     variable_values["output_frame"],
        #     text=fps,
        #     org=(3, 15),
        #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #     fontScale=0.50,
        #     color=(255, 0, 0),
        #     thickness=2,
        # )

        if self.output_queue.full(): 
            self.output_queue.get_nowait()
        self.output_queue.put(variable_values["output_frame"])

    def run(self):
        while not self.stop: 
            ret, frame = self.get_frame()
            if ret: 
                self.canny_frame(frame)


if __name__ == '__main__':

    frame_sum = 0
    init_time = time.time()

    def put_frame(frame):
        if Input_Queue.full(): 
            Input_Queue.get_nowait()
        Input_Queue.put(frame)

    def cap_read(cv2_cap):
        ret, frame = cv2_cap.read()
        if ret: 
            put_frame(frame)
        
    cap = cv2.VideoCapture(source)

    threadn = cv2.getNumberOfCPUs()
    print('Number of threadn: ', threadn)
    threadn = 2
    threaded_mode = True

    process_list = []
    Input_Queue = Queue(maxsize = 5)
    Output_Queue = Queue(maxsize = 5)

    for x in range((threadn -1)):    
        canny_process = Canny_Process(frame_queue = Input_Queue,output_queue = Output_Queue)
        canny_process.daemon = True
        canny_process.start()
        process_list.append(canny_process)

    ch = cv2.waitKey(1)
    cv2.namedWindow('Threaded Video', cv2.WINDOW_NORMAL)
    while True:        
        cap_read(cap)
        # print('Im here')
        if not Output_Queue.empty():
            print('not empty')
            result = Output_Queue.get()
            cv2.imshow('Threaded Video', result)
            ch = cv2.waitKey(5)

        if ch == ord(' '):
            threaded_mode = not threaded_mode
        if ch == 27:
            break
    cv2.destroyAllWindows()