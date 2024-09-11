import os
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
import yaml
from pymap3d import *
import pandas as pd 
import argparse
from typing import List, Union
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import open3d as o3d
import os 
from g2o_utils import *

'''
TransformationHandler class is to 1. read & process g2o files. 
dict_all stores g2o information as .json file:

Dict_all: 
{
    "mission_0_name": [
        {
            "node": 0,
            "scan_name": "cloud_2023_08_15_000000001.pcd",
            "scan_time": "2023_08_15 13:12:48.123456",
            "pose": ["1.0", "2.0", "3.0", "0.0", "0.0", "0.0", "1.0"]
            ..add more key here
        },
        {
            "node": 1,
            "scan_name": "cloud_2023_08_15_000000002.pcd",
            "scan_time": "2023_08_15 13:12:50.654321",
            "pose": ["1.1", "2.1", "3.1", "0.0", "0.0", "0.0", "1.0"]
        },
        ...
    ],
    "mission_1_name": [
        {
            "node": 5000,
            "scan_name": "cloud_2023_08_16_000000001.pcd",
            "scan_time": "2023_08_16 14:15:10.789012",
            "pose": ["4.0", "5.0", "6.0", "0.0", "0.0", "0.0", "1.0"]
'''

class TransformationHandler:
    def __init__(self, place_folder: str, mission_folders: List[str], g2o_files: List[str], offset: int = 5000):
        self.place_folder = place_folder
        self.mission_folders = mission_folders
        self.g2o_files = g2o_files  # Now expects a list of G2O files
        self.offset = offset
        self.dict_all = {}
        self.process_multiple_missions() # Read and write g2o
        
    def process_g2o_dict(self, g2o_file, mission_folder, offset):
        g2o_data = {
            'nodes': getNode_g2o(g2o_file),
            'poses': getPoses_g2o(g2o_file),
            'scan_times': getScanTime_g2o(g2o_file),
            'scan_names': getScanNames_g2o(g2o_file)
        }
        mission_data = []
        for i in range(len(g2o_data['nodes'])):
            mission_data.append({
                "node": g2o_data['nodes'][i] + offset,
                "scan_name": g2o_data['scan_names'][i],
                "scan_time": g2o_data['scan_times'][i],
                "pose": g2o_data['poses'][i]
            })
        
        self.dict_all[mission_folder] = mission_data

    # Basic method to read g2o files 
    def process_multiple_missions(self):
        if isinstance(self.g2o_files, list) and len(self.g2o_files) == len(self.mission_folders):
            # Loop through each mission and corresponding G2O file
            offset_list = [self.offset * i for i in range(len(self.mission_folders))]
            for idx, (mission_folder, g2o_file) in enumerate(zip(self.mission_folders, self.g2o_files)):
                print(f"Processing mission {mission_folder} with G2O file {g2o_file}")
                self.process_g2o_dict(g2o_file, mission_folder, offset_list[idx])
        else:
            raise ValueError("g2o_files must match the number of mission folders")

    ## -------------- Extra functionalities between missions ----------------  

    def calculate_timestamps_difference(self, mission_id_1, mission_id_2):
        """
        Calculate the time difference between corresponding nodes of two missions.
        """
        mission_1_data = self.dict_all[self.mission_folders[mission_id_1]]
        mission_2_data = self.dict_all[self.mission_folders[mission_id_2]]
        
        # Ensure they have the same number of nodes for comparison
        if len(mission_1_data) != len(mission_2_data):
            raise ValueError("Missions have different numbers of nodes, cannot compare directly.")
        
        time_differences = []
        for node_1, node_2 in zip(mission_1_data, mission_2_data):
            time_1 = datetime.strptime(node_1['scan_time'], "%Y_%m_%d %H:%M:%S.%f")
            time_2 = datetime.strptime(node_2['scan_time'], "%Y_%m_%d %H:%M:%S.%f")
            diff = abs((time_1 - time_2).total_seconds())
            time_differences.append(diff)
        
        return time_differences
    
    
    def calculate_relative_transformation(self, mission_id_1, mission_id_2):
        """
        Calculate the relative transformation matrix between mission 1 and mission 2 at each node    
        """
        mission_1_data = self.dict_all[self.mission_folders[mission_id_1]]
        mission_2_data = self.dict_all[self.mission_folders[mission_id_2]]
        
        # Ensure they have the same number of nodes for comparison
        if len(mission_1_data) != len(mission_2_data):
            raise ValueError("Missions have different numbers of nodes, cannot compare directly.")
        
        for node_1, node_2 in zip(mission_1_data, mission_2_data):
            pos_1 = np.array([float(x) for x in node_1['pose'][:3]])
            quat_1 = np.array([float(x) for x in node_1['pose'][3:]])
            pos_2 = np.array([float(x) for x in node_2['pose'][:3]])
            quat_2 = np.array([float(x) for x in node_2['pose'][3:]])
            # 4x4 transformation
            T_m1 = pose_to_matrix(pos_1, quat_1)
            T_m2 = pose_to_matrix(pos_2, quat_2)
            
            # get relative transformation
            # T_target_source = T_target_base @ inv(T_source_base)
            # T_m2_m1 = T_m2_base @ inv(T_m1_base)
            T_m2_m1 =  T_m2 @ np.linalg.inv(T_m1)
            
            # Store relative transformation in the dictionary
            node_1['transform_T_m2_m1'] = T_m2_m1
            
        self.dict_all[self.mission_folders[mission_id_1]] = mission_1_data
        

    def calculate_pose_m1_in_m2(self, mission_id_1, mission_id_2):
            """
            Calculate pose of mission_1 in the frame of mission_id_2.
            """
            self.calculate_relative_transformation(mission_id_1, mission_id_2)
            mission_1_data = self.dict_all[self.mission_folders[mission_id_1]]
            
            for node in mission_1_data:
                relative_transform_matrix = node.get('transform_T_m2_m1')
                if relative_transform_matrix is not None:
                    # Get the pose in mission_id_1's frame
                    pose_m1 = node['pose']
                    pose_m1_matrix = pose_to_matrix(pose_m1[:3], pose_m1[3:])
                    
                    # Compute pose_m1_in_m2
                    pose_m1_m2_matrix = relative_transform_matrix @ pose_m1_matrix
                    pose_m1_m2 = pose_m1_m2_matrix[:3, 3:].flatten().tolist() + R.from_matrix(pose_m1_m2_matrix[:3, :3]).as_quat().tolist()
                    node['pose_m1_m2'] = pose_m1_m2
                else:
                    print('relative_transform_matrix missing!!!!')
    
def main():
    parser = argparse.ArgumentParser(description="Process configurations for multiple missions")
    parser.add_argument('--config', default='config.yaml', type=str, help='Path to YAML config file')
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Initialize TransformationHandler with configurations
    handler = TransformationHandler(
        place_folder=config['place_folder'],
        mission_folders=config['mission_folders'],
        g2o_files=config['g2o_files'],
        offset=config['offset']
    )

    # check dict
    result = handler.dict_all
    print(f"Processed {len(result)} missions:")
    for mission, data in result.items():
        print(f"{mission}: {len(data)} nodes")



    # Execute additional actions specified in YAML
    for action in config['actions_transform_handler']:
        if hasattr(handler, action):
            print(f"Executing {action}...")
            getattr(handler, action)(mission_id_1=0, mission_id_2=1)
        else:
            print(f"Action {action} not found in handler")

if __name__ == "__main__":
    main()