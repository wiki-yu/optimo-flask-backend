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

import pandas as pd
import os
import numpy as np


actual_path = os.path.dirname(os.path.abspath(__file__))
actual_path = os.path.dirname(actual_path)
actual_path = os.path.join(actual_path, "outputs",'output_df_final.csv')
df = pd.read_csv(actual_path, index_col = False)

# Make blocks for rows with the same task number
df['subgroup'] = (df.iloc[:,-1] != df.iloc[:,-1].shift(1)).cumsum()

# If the block length is less than 9, we'll assume that is a recognition error
added = df['subgroup'].value_counts()[df['subgroup'].value_counts()<9].index.tolist()

# For those with task recognition error, we'll assume that the correct task number is the preious task
for i in added:
    # print('i: ', i)
    prev_task = int(df[df['subgroup']==(i-1)].iloc[0,-2])
    change_idx = df.index[df['subgroup']==i].tolist()
    # print('change_idx: ', change_idx)
    # print('previous_task: ',prev_task)
    df.iloc[change_idx,-2] = prev_task

# Create the blcoks again
df['subgroup'] = (df.iloc[:,-2] != df.iloc[:,-2].shift(1)).cumsum()

# Normal order of the tasks (defined by user)
normal_order = [0,1,2,1,5,6,1,10,11,10,14,10,1,2,1]

# Add a single row for those tasks that are missed
cnt = 0
for i in range(len(df["subgroup"].unique())):
    task_label = int(df[df['subgroup']==(i+1)].iloc[0,-2])
    # print("task_label: ", task_label)
    
    if task_label != normal_order[cnt]:
        while task_label != normal_order[cnt]:
            idx = df.index[df['subgroup']==(i+1)].tolist()[0]
#             new_row = df[df['subgroup']==(i+1)].iloc[-1].tolist()
            new_row = df.iloc[(idx-1)].tolist()
            #print('new_row: ', new_row)
            # print("normal_order[cnt]: ", normal_order[cnt])
            # print('idx: ', idx)
            new_row[-2]=normal_order[cnt] 
#             new_row[-1]=normal_order[-1]
            new_row = np.array(new_row)
    #         print(new_row)
    #         print(df.columns)
            new_row = pd.DataFrame([new_row], columns = df.columns.tolist())
    #         print('new_row: ', new_row)
            df = pd.concat([df[:idx], new_row, df[idx:]],ignore_index=True).reset_index(drop=True)
            # print('len: ', len(df))
            # print('cnt: ', cnt)
            cnt+=1
            if cnt==len(normal_order):
                cnt = 0

    cnt+=1
    if cnt==len(normal_order):
        cnt = 0
actual_path = os.path.dirname(os.path.abspath(__file__))
actual_path = os.path.dirname(actual_path)

df.to_csv(os.path.join(actual_path, "outputs",'output_df_final_modified.csv'), index = False)