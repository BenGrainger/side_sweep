import deeplabcut as dlc
import os
import pandas as pd

path = '/research/DATA/SUBJECTS_AB'
csv_paths = []
directory_list = os.listdir(path)
for directory in directory_list:
    file_list = os.listdir(path + '/' + directory)
    for file in file_list:
        if 'Meta.csv' in file:
            name = pd.read_csv(path + '/' + directory + '/' + file)['file_names']
            csv_paths.append(path + '/' + directory + '/' + file[:-8] + '.avi')
            
config_path = '/home/beng/Desktop/notebooks/Escape-BenG-2020-12-03/config.yaml'
for i, path in enumerate(csv_paths):
    print(str(i+1), 'out of:', str(len(csv_paths)))
    dlc.analyze_videos(config_path, path)
    
config_path = '/home/beng/Desktop/notebooks/Escape-BenG-2020-12-03/config.yaml'
for i, path in enumerate(csv_paths[:3]):
    print(str(i+1), 'out of:', str(len(csv_paths[:3])))
    dlc.create_labeled_video(config_path, [path], save_frames=True)