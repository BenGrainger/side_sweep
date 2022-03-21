import os
import pandas as pd
import numpy as np
import math
import scipy.io as sio
import cv2

def loadMATfile(MATfile, corners = []):
    """
    loads the MAT file containing the checkers corners coordinates
    returns the coordinates in a calable form for CV2
    """
    loadedMATfile = sio.loadmat(MATfile)
    CORNERS = loadedMATfile['CALIBDATA'][0][0][3]
    

    for corner in CORNERS:
        xy = list(corner)
        xy.append(1)
        corners.append(xy)

    corners = np.array(corners, dtype='float32')
    
    return corners

############################################################################

def getManuallyAdjustedCorners(numberCorners, boardwidth, boardheight, xmax, ymax):
    """
    returns a set of cooridnates that correspond with the real corners 
    however unsquewed by perspective or barrel distortion
    """
    widthPerc = 1/(boardwidth-1)
    heightPerc = 1/(boardheight-1)

    transform_cornersX = []
    transform_cornersY = []
    transform_corners = []

    for i in range(numberCorners):
        if i == 0:
            width = 0
            height = 0
        else:
            width = i%boardwidth
            height = math.floor(i/boardwidth)
        transform_cornersX.append(width*widthPerc*xmax)
        transform_cornersY.append(height*heightPerc*ymax)

    transform_cornersY.reverse()
    for x,y in zip(transform_cornersX, transform_cornersY):
        transform_corners.append([x,y])
    transform_corners = np.float32(transform_corners)
    return transform_corners


###############################################################################

def PerspectiveDistortionMatrix(calibrationImage, corners, ManualCorners, warpingDims = (650,800)):
    """
    generates linear matrix from the adjusted coordinates
    """
    img = cv2.imread(calibrationImage)
    rows,cols,ch = img.shape
    pts1 = np.array([[list(corners[49][:2])], [list(corners[-1][:2])], [list(corners[0][:2])], [list(corners[6][:2])]], dtype='float32')
    pst2 = np.array([[list(ManualCorners[49]+100)], [list(ManualCorners[-1]+100)], [list(ManualCorners[0]+100)], [list(ManualCorners[6]+100)]], dtype='float32')
    M = cv2.getPerspectiveTransform(pts1,pst2)

    return M

def main():
    path = 'X:\ibn-vision\DATA\SUBJECTS_AB' 
    video_calibrations_path = 'X:\ibn-vision\DATA\RIGS\SOLOMON11\VIDEO_CALIBRATIONS'
    data_base = pd.read_excel(r'C:\Users\ben.grainger\Desktop\notebooks\ABlookup.xlsx', index_col=False)
    file_names = data_base["'FILENAME'"]
    cal_names = data_base["'CALIBRATION_FILE'"]
    stim1 = data_base["'STIM1_TIME'"]
    stim2 = data_base["'STIM2_TIME'"]
    stim3 = data_base["'STIM3_TIME'"]
    stim4 = data_base["'STIM4_TIME'"]
    stim5 = data_base["'STIM5_TIME'"]
    stim6 = data_base["'STIM6_TIME'"]

    for f,cn,s1,s2,s3,s4,s5,s6 in zip(file_names,cal_names,stim1,stim2,stim3,stim4,stim5,stim6):
        new_f = path + '/' + f[1:6] + '/' + f[1:-1]
        MATfile = video_calibrations_path +'/'+ cn[1:-1] + '.mat'
        Corners = loadMATfile(MATfile, corners = [])
        boardwidth, boardheight = sio.loadmat(MATfile)['CALIBDATA'][0][0][6][0][0]-1, sio.loadmat(MATfile)['CALIBDATA'][0][0][6][0][1]-1
        ManualCorners = getManuallyAdjustedCorners(len(Corners), boardwidth, boardheight, boardwidth*55, boardheight*55)
        M = PerspectiveDistortionMatrix(MATfile[:-4], Corners, ManualCorners)
        Matrix_path = new_f+'M.npy'
        np.save(Matrix_path, np.array(M))
        d = pd.DataFrame({'file_names': [new_f + '.avi'],'cal_names': [MATfile], 'M_matrix': Matrix_path, 'stim1': s1,'stim2':s2,'stim3':s3,'stim4':s4,'stim5':s5,'stim6':s6})
        d.to_csv(new_f + 'Meta' + '.csv')
        
if __name__ == '__main__':
    main()