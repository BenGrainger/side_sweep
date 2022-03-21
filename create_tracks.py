import os
import pandas as pd
import numpy as np
import math
import cv2

path = 'X:\ibn-vision\DATA\SUBJECTS_AB'  

new_coordinates_list = []

directory_list = os.listdir(path)
for directory in directory_list:
    file_list = os.listdir(path + '/' + directory)
    for file in file_list:
        if '.h5' in file:
            
            # load h5 file into numpy array
            file_path = path + '/' + directory + '/' + file 
            loaded = pd.read_hdf(file_path)
            obstructive_title = loaded.keys()[0][0]
            extracted_tracks = loaded[obstructive_title]
            DLC_array = extracted_tracks.to_numpy()
            
            # load M matrix
            session = file[:15]
            M = np.load(path + '/' + directory + '/' + session + 'M.npy')
            
            # load stim times
            meta = pd.read_csv(path + '/' + directory + '/' + session + 'Meta.csv')
            stims = np.array(meta)[0][4:]
            stims = [i for i in stims if i!= '[]']
            stims = [int(np.round(i)) for i in stims]
            stims = np.array(stims)
            
            # iterate through stimuli to create slice of whole array around stimulus onset
            for stim_n, stim in enumerate(stims):
                stim = stim*60
                escape_array = DLC_array[stim-60:stim+180] # DLC_array[frame-60+600:frame+240+600]
                escape_array = escape_array.T
                escape_array.shape
                columns = []

                for i in range(0,42,3):
                    ##
                    # iterate thorugh columns i.e. [x,y,liklihood,x2,y2,iklihood2...xn,yn,liklihoodn]
                    ##
                    coordinate_slice = escape_array[i:i+2].T
                    liklihood = escape_array[i+2].T
                    liklihood = liklihood.reshape(240,1)
                    ##
                    # slice accordingly and turn 3 dimensional by z = 1
                    ##
                    coordinates = []
                    for coor in coordinate_slice:
                        xy = list(coor)
                        xy.append(1)
                        coordinates.append(xy)
                    coordinate = np.array(coordinates, dtype='float32')

                    ##
                    # apply M matrix to coordinates
                    ##
                    preDiv = np.dot(M, coordinate.T)
                    afterDiv = preDiv/preDiv[2]
                    postLinearTransform = afterDiv[:2].T

                    ##
                    # restack and arrange list of coordinates back
                    ##
                    new_column = np.column_stack((postLinearTransform, liklihood))
                    columns.append(new_column)

                new_coordinates = np.column_stack(columns).T
                new_coordinates_list.append(new_coordinates)
                
                output_path = path + '/' + directory + '/' + session + '_{}escape.npy'.format(stim_n+1)
                np.save(output_path, new_coordinates, allow_pickle=True)