'''
@ Author: Ines Alcibar
@ Year: 2020
@ Project: AI_builder
@ Version:

@ Description:

'''

import os
from joblib import load
import cv2
import numpy as np
import pandas as pd

###########################################################################################################
##########################                                                       ##########################             
##########################                Task Recognition Utils                 ##########################
##########################                                                       ########################## 
###########################################################################################################
# Task recognition module is set up here, so that we only need to call the function 

# Define the function for the recognition module
# def recognition_module(history_seq, seq):
#     # history_seq: CoG_history[1:]
#     # seq: CoG for the corresponging frame
    
#     # Initialize the sequence
#     sequence = []

#     # Convert 2D array into 1D array, by taking all the elements of the CoG_history excep the last column (task_label column)
#     for hh in range(history_seq.shape[0]):
#         sequence.extend(history_seq[hh][:-1].tolist())

#     # Add data from the last frame
#     sequence.extend(seq)

#     # Convert the data so that it can be fed to the model
#     sequence = np.array(sequence)
#     sequence = sequence.reshape(-1,sequence.shape[0]) # Remove one dimension

#     # Obtain the predicted task from the model
#     label = options["network"].predict(sequence)
#     label = int(label)

#     return class_names[label], label


	# label_description, task_number = recognition_module(CoG_history[:-1],CoG_history[-1])


###########################################################################################################
##########################                                                       ##########################             
##########################                Task Recognition Main Function         ##########################
##########################                                                       ########################## 
###########################################################################################################

def function(inputs,options):

	frame = inputs["image"]
	CoG_history = inputs["CoG"]
	class_names = inputs["task_class_names"]
	output_df = inputs["output_df"]

    
    # Initialize the sequence
	sequence = []

    # Convert 2D array into 1D array, by taking all the elements of the CoG_history excep the last column (task_label column)
	for hh in range(CoG_history[:-1].shape[0]):
		sequence.extend(CoG_history[:-1][hh][:-1].tolist())

    # Add data from the last frame
	sequence.extend(CoG_history[-1][:-1])

    # Convert the data so that it can be fed to the model
	sequence = np.array(sequence)
	sequence = sequence.reshape(-1,sequence.shape[0]) # Remove one dimension

    # Obtain the predicted task from the model
	label = options["network"].predict(sequence)
	label = int(label)

	# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
	# print(label)
	# print(sequence)

	CoG_history[-1][-1] = label

	frame = cv2.putText(frame, class_names[label], (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 225), 2, cv2.LINE_AA)

	if output_df != False:
		# print("output_df: ",output_df)
		# print("CoG_history: ", CoG_history[-1])
		cnt = [output_df[-1][0]+1]
		# print('cnt: ', cnt)
		cnt.extend(CoG_history[-1])
		output_df.append(cnt)
		if not cnt[0]%7000:
			actual_path = os.path.dirname(os.path.abspath(__file__))
			actual_path = os.path.dirname(actual_path)
			pd.DataFrame(output_df).to_csv(os.path.join(actual_path, "outputs",'output_df_'+str(cnt[0])+'.csv'), index = False)
			output_df = [output_df[-1]]
			print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
			print("output_df: ",output_df)


	return {"image": frame, "CoG": CoG_history, "output_df": output_df}


###########################################################################################################
##########################                                                       ##########################             
##########################             Task Recognition Initialization           ##########################
##########################                                                       ########################## 
###########################################################################################################
# Next step is to change it to a text file coming from the training 

def initialize_variables(options):
	task_class_names = dict(options["task_class_names"])
	print('options["output_df"]', options["output_df"])

	# Create output dataframe with CoG and task label if required
	try:
		if options["output_df"][0]:
			output_df = [[0]*(options["output_df"][1]*2+2)]
			print("output_df: ", output_df)
		else:
			output_df = False
	except:
		output_df = False

	return {'new_values': {"task_class_names": task_class_names, "output_df": output_df},
		'new_inputs':{"task_class_names": "task_class_names","output_df": "output_df"},
		'new_outputs':{"output_df": "output_df"}}



###########################################################################################################
##########################                                                       ##########################             
##########################             Load Task Recognition Model               ##########################
##########################                                                       ########################## 
###########################################################################################################

def load_model(model_name,options,path):
	print('----- Task recognition module info -----')
	print('model name: ', model_name)

	model_path = os.path.join(path, str(model_name)+".joblib")
	clf = load(model_path) 

	return clf

###########################################################################################################
##########################                                                       ##########################             
##########################             Task Recognition Finlize                  ##########################
##########################                                                       ########################## 
###########################################################################################################

def finalize(inputs, options, vid_name):
	
	output_df = inputs["output_df"]
	if options["output_df"] != False:
		actual_path = os.path.dirname(os.path.abspath(__file__))
		actual_path = os.path.dirname(actual_path)
		pd.DataFrame(output_df).to_csv(os.path.join(actual_path, "outputs",'CoG_with_task_'+vid_name+'.csv'), index = False)
			