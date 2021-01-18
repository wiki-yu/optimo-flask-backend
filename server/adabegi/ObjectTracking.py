'''
@ Author: Ines Alcibar
@ Year: 2020
@ Project: AI_builder
@ Version:

@ Description:

'''
import os
import numpy as np
import cv2

###########################################################################################################
##########################                                                       ##########################             
##########################                Object Tracking Utils                  ##########################
##########################                                                       ########################## 
###########################################################################################################

# Define different ways and functions for tracking an object

def calculate_CoG(out_pred_c):
    ''' 
        Calculate CoG
        Calling example:
        CoGx, CoGy = calculate_CoG(out_pred_c)
    '''
    CoGx = (out_pred_c[0]+out_pred_c[2])/2
    CoGy = (out_pred_c[1]+out_pred_c[3])/2    
    return CoGx, CoGy

def check_closeness(CoGx,CoGy,CoG_prev,video_size,closeness):
    '''
        Check if the object is close enough to the previously detected object
        Calling example:
        CoGx,CoGy = check_closeness(CoGx,CoGy,CoG_prev,video_size,closeness)
    '''
    if not ((CoGx<(CoG_prev[0]+video_size[0]*closeness))&(CoGx>(CoG_prev[0]-video_size[0]*closeness))&(CoGy<(CoG_prev[1]+video_size[1]*closeness))&(CoGy>(CoG_prev[1]-video_size[1]*closeness))):
            CoGx = CoG_prev[0]
            CoGy = CoG_prev[1]
    return CoGx, CoGy

def tracking_1_object_never(out_pred_c):
    '''
        Don't track the object, but make sure there's only one if this kind
        Calling example:
        CoGx,CoGy = tracking_1_object_never(out_pred_c)
    '''
    # if reference_labels[obj] == 'Tweezer': # If object is tweezer
    if len(out_pred_c)==0: # If that object is not detected: (0,0)
        CoGx,CoGy = 0,0
    else:
        if len(out_pred_c)>1: # Take the one with the highest score
            out_pred_c.sort(key=lambda tup: tup[5], reverse=True) 
            out_pred_c = [out_pred_c[0]]
        CoGx, CoGy = calculate_CoG(out_pred_c[0])

    return CoGx,CoGy


def tracking_1_object_always(out_pred_c,CoG_prev,video_size,check_closeness_on, closeness=0.15):
    # if reference_labels[obj] in ['Left_hand', 'Cotton_stick', 'Right_hand', 'Press', 'Press_bottom', 'Fixture2', 'Trash', 'Conveyor']: 
    '''
        This function tracks an object that is supposed to appear just once in each frame.
        Different scenarios:
            - There's just one object detected: calculate CoG (Center of Gravity)
            - More than 1 object detected:
                - If no previous detection: selects the object with highest score
                - If previously detected: selects the object that is closer to the previously detected

        After the new CoG is detected: check if the object is close enough to the previously detected object (default 15%)

        Calling example: 
        CoGx,CoGy = tracking_1_object_always(out_pred_c,CoG_prev,video_size,check_closeness_on,closeness=0.15)
    '''                    
    if len(out_pred_c)==0: # If that object is not detected, use previous frame
        CoGx,CoGy = CoG_prev
    elif len(out_pred_c)==1:
        CoGx, CoGy = calculate_CoG(out_pred_c[0])        

    else: # We need to check which one is closer to the previous detection
        
        if ((CoG_prev[0]==0)&(CoG_prev[1]==0)): # If not previously found
            out_pred_c.sort(key=lambda tup: tup[5], reverse=True)     
            CoGx, CoGy = calculate_CoG(out_pred_c[0])
            
        else:

            dist_min = 1000000
                
            for hh in range(len(out_pred_c)):
                CoGx1, CoGy1 = calculate_CoG(out_pred_c[hh])

                dist = (abs(CoGx1-CoG_prev[0]) + abs(CoGy1-CoG_prev[1]))**0.5
                if dist<dist_min:
                    CoGx = CoGx1
                    CoGy = CoGy1
                    dist_min = dist

    # Check that the obtained results are coherent
    if check_closeness_on:
        if ((CoG_prev[0]!=0)&(CoG_prev[1]!=0)):
            CoGx,CoGy = check_closeness(CoGx,CoGy,CoG_prev,video_size,closeness)
    
    return CoGx,CoGy
    
def tracking_1_object_sometimes(CoG_prev_cond,out_pred_c,CoG_prev,video_size,closenessCond=0.2,closeness=0.2):
    '''
        Track the object unless it's near the conveyor (a condition given in CoG_prev_cond (CoGx, CoGy)) 
        Calling example:
        CoGx,CoGy = tracking_1_object_sometimes(CoG_prev_cond,out_pred_c,CoG_prev,video_size,closenessCond=0.2,closeness=0.2)
    '''
    #if reference_labels[obj] == 'Phone_1': # If object is Phone (and at least one was detected)
    if len(out_pred_c)==0: # If that object is not detected, use previous frame
        CoGx,CoGy = CoG_prev
    else: 
        CoGx,CoGy = tracking_1_object_always(out_pred_c,CoG_prev,video_size,False,closeness)
        if not ((CoG_prev[0]==0)&(CoG_prev[1]==0)): # If previously detected
            # If not near the conveyor in previous frame, check closeness to track
            if not ((CoG_prev[0]<(CoG_prev_cond[0]+video_size[0]*closenessCond))&(CoG_prev[0]>(CoG_prev_cond[0]-video_size[0]*closenessCond))&(CoG_prev[1]<(CoG_prev_cond[1]+video_size[1]*closenessCond))&(CoG_prev[1]>(CoG_prev_cond[1]-video_size[1]*closenessCond))):   
                CoGx,CoGy = check_closeness(CoGx,CoGy,CoG_prev,video_size,closeness)
        
    return CoGx,CoGy


            
def tracking_2plus_objects_always(num_obj, CoG_obj_cond,out_pred_c,CoG_prev,video_size,closeness=0.15):
    '''
        Track the 2+ objects all the time (if the object is not detected, the previous detection will be used)
        For example 2 hands.
        Conditions must be set to know which object is which in the beginning
        num_obj(quantity of objects to track), CoG_obj_cond(conditions to know which object is which)
        Calling example:
        CoGxy = tracking_2plus_objects_always(num_obj, CoG_obj_cond,out_pred_c,CoG_prev,video_size,closeness=0.15)
    '''
    if len(out_pred_c)==0: # If that object is not detected, use previous frame
        CoGxy = CoG_prev
    else:

        dist_to_cond = [] 
        CoGxy = CoG_prev.copy()

        if CoG_prev.tolist().count(0)==0: # If all the objects previously detected

            for hh in range(len(out_pred_c)):
                CoGx1, CoGy1 = calculate_CoG(out_pred_c[hh])
                dist_to_cond_i = []
                for j in range(num_obj):
                    dist_to_cond_i.extend([(abs(CoGx1-CoG_prev[(j*2)]) + abs(CoGy1-CoG_prev[(j*2+1)]))**0.5])
                assert len(dist_to_cond_i)==num_obj
                dist_to_cond.append(dist_to_cond_i)

        else: # If one/neither objects were detected previously, check conditions

            for hh in range(len(out_pred_c)):
                CoGx1, CoGy1 = calculate_CoG(out_pred_c[hh])
                dist_to_cond_i = []
                for j in range(num_obj):
                    if len(CoG_obj_cond[j])>0:
                        dist_to_cond_i.extend([min([(abs(CoGx1-(item[0]+item[2])/2) + abs(CoGy1-(item[3]+item[1])/2))**0.5 for item in CoG_obj_cond[j]])])
                    else:
                        dist_to_cond_i.extend([100000])
                assert len(dist_to_cond_i)==num_obj
                dist_to_cond.append(dist_to_cond_i)
      
        for i in range(len(out_pred_c)):
            min_dist = [min(x) for x in dist_to_cond]
            min_dist_idx = min_dist.index(min(min_dist))
            x = dist_to_cond[min_dist_idx].index(min(dist_to_cond[min_dist_idx]))
            CoGxy[(x*2):(x*2+2)] = calculate_CoG(out_pred_c[min_dist_idx])
            dist_to_cond[min_dist_idx] = [100000]*num_obj

        # Check that the obtained results are coherent
        for i in range(num_obj):
            if ((CoG_prev[i*2]!=0)&(CoG_prev[i*2+1]!=0)):
                CoGxy[(i*2):(i*2+2)] = check_closeness(CoGxy[(i*2)],CoGxy[(i*2+1)],CoG_prev[(i*2):(i*2+2)],video_size,closeness)
    
    return CoGxy

###########################################################################################################
##########################                                                       ##########################             
##########################                Object Tracking Main Function          ##########################
##########################                                                       ########################## 
###########################################################################################################



def function(inputs,options):

	CoG_history = inputs["CoG"]
	out_pred = inputs["boxes"]
	input_labels = inputs["yolo_input_labels"]
	reference_labels = inputs["yolo_reference_labels"]
	rr = inputs["rr"]
	t2plus = options["t2plus"]['objects']
	t1always = options["t1always"]['objects']
	t1sometimes = options["t1sometimes"]['objects']
	frame = inputs["image"]
	plot_CoG_on = True
	video_size = (frame.shape[1],frame.shape[0])



	CoG = []

	for obj in range(len(reference_labels)): # Object is a number

		obj1 = input_labels.index(reference_labels[obj])

		out_pred_c = [item for item in out_pred if item[4] == obj1] # Take the outputs with that label
        
        # Get the results from the previous frame
		idx_rr = [i for i,x in enumerate(rr) if x==reference_labels[obj]]
		CoG_prev = np.take(CoG_history[-1],idx_rr)

        # Get the new CoG                
		if reference_labels[obj] in t2plus: # If object==hands, we want to have 2,left and right

			num_obj = int(rr.count(reference_labels[obj]) /2) # 2 hands
			CoG_obj_cond = []
			for hh in options["t2plus"]['conditions'][options["t2plus"]['objects'].index(reference_labels[obj])]:
				out_pred_cond = [item for item in out_pred if item[4] == input_labels.index(hh)]

				CoG_obj_cond.append(out_pred_cond)

			CoGxy = tracking_2plus_objects_always(num_obj, CoG_obj_cond,out_pred_c,CoG_prev,video_size,closeness=0.15)



		elif reference_labels[obj] in t1always:                     
			CoGxy = tracking_1_object_always(out_pred_c,CoG_prev,video_size,check_closeness_on=True,closeness=0.15)        

		elif reference_labels[obj] in t1sometimes: # If object is Phone (and at least one was detected)
			idx_rr = [i for i,x in enumerate(rr) if x==options["t1sometimes"]['conditions'][options["t1sometimes"]['objects'].index(reference_labels[obj])]]
			CoG_prev_cond = np.take(CoG_history[-1],idx_rr) # Position of the conveyor in the previous frame
			CoGxy = tracking_1_object_sometimes(CoG_prev_cond,out_pred_c,CoG_prev,video_size,closenessCond=0.2,closeness=0.2)

		else:# reference_labels[obj] in t1never: 
			CoGxy = tracking_1_object_never(out_pred_c)

		# Add to CoG
		CoG.extend(CoGxy)

        # Plot CoG
		if plot_CoG_on:
			for hh in range(int(len(CoGxy)/2)):
				result = cv2.circle(frame, (int(CoGxy[hh*2]), int(CoGxy[hh*2+1])),10,(0,255,0),-1)
	CoG.extend([-1])
	CoG = np.array([CoG])
	CoG_history = CoG_history[1:] # Remove the oldest CoG data
	CoG_history = np.concatenate((CoG_history,CoG)) # Add the new CoG data



	return {"CoG": CoG_history, "image": frame}

###########################################################################################################
##########################                                                       ##########################             
##########################             Object Tracking Initialization            ##########################
##########################                                                       ########################## 
###########################################################################################################


def initialize_variables(options):
	data_columns_per_frame = options["objects_per_frame"]*2+1

	try:
		hist_count = options["history_count"]
	except:
		hist_count = 15

	class_path = os.path.dirname(os.path.abspath(__file__))
	class_path = os.path.join(os.path.dirname(class_path),"trained_models",str(options["yolo_model_name"]+".txt"))
	class_file = open(class_path, "r")
	input_labels = [line.rstrip("\n") for line in class_file.readlines()]
	reference_labels = options["reference_labels"]

	r = options["several_objects_per_frame"]
	rr = []
	for i in reference_labels:
		if i in r["objects"]:
			for j in range((r["qty"][r["objects"].index(i)])*2):
				rr.extend([i])
		else:
			for j in range(2):
				rr.extend([i])

	CoG_name = options["CoG_name"]


	return {'new_values': {str(CoG_name): np.zeros((hist_count,data_columns_per_frame)), "yolo_input_labels": input_labels, "rr": rr, "yolo_reference_labels": reference_labels}, 
		'new_inputs': {'CoG': str(CoG_name), "yolo_input_labels": "yolo_input_labels", "rr": "rr", "yolo_reference_labels": "yolo_reference_labels"},
		'new_outputs': {}}