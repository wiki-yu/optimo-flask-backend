'''
@ Author: Ines Alcibar
@ Year: 2020
@ Project: AI_builder
@ Version:

@ Description:

'''

import numpy as np
import cv2


###########################################################################################################
##########################                                                       ##########################
##########################                Object Tracking Main Function          ##########################
##########################                                                       ##########################
###########################################################################################################


def function(inputs,options):

	task_flags = inputs["checked_task_flags"]
	task_history = inputs["task_history"]
	frame = inputs["image"]
	CoG_hist = inputs["CoG"]

	tracked_tasks = options["tasks"]
	task_duration = options["task_duration"]
	x = options["checked_percentage"]
	cond = options["conditions"]
	reset_cond = options["reset_cond"]

	task_history = np.append(task_history[1:],[int(CoG_hist[-1][-1])])



	for i in range(len(task_flags)):

		checking_cond = False

		if cond[i]=='first':
			if sum(task_flags)==0:
				checking_cond=True
		if cond[i]=='plus1':
			if sum(task_flags)>1:
				checking_cond=True
		if cond[i]=='':
			checking_cond=True
	
		if checking_cond:
			if task_flags[i]==False:
				if sum(task_history[-int(task_duration[i]/x):]==tracked_tasks[i])>task_duration[i]:
					task_flags[i]=True

	# Check if we have to reset the task flags
	if reset_cond[2]=='plus1':
		if sum(task_flags)>1:
			if sum(task_history[-int(reset_cond[1]/x):]==reset_cond[0])>reset_cond[1]:
				task_flags = [False]*len(task_flags)

	task_tracking = 'completed: '+ str(task_flags)

	frame = cv2.putText(frame, task_tracking, (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA) 

	return {"image": frame, "checked_task_flags": task_flags, "task_history": task_history}






###########################################################################################################
##########################                                                       ##########################
##########################               Task Tracking Initialization            ##########################
##########################                                                       ##########################
###########################################################################################################


def initialize_variables(options):

	task_history = np.ones(options["max_history"])*(-1)
	checked_task_flags = [False]*options["tracked_task_qty"]



	return {'new_values': {"task_history": task_history, "checked_task_flags": checked_task_flags}, 
		'new_inputs': {"task_history": "task_history","checked_task_flags":"checked_task_flags"},
		'new_outputs': {"task_history": "task_history","checked_task_flags":"checked_task_flags"}}