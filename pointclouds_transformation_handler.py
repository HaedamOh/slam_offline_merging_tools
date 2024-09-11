import os
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
from pymap3d import *
import pandas as pd 
import argparse
from typing import List, Union
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import open3d as o3d
import yaml
import os 
from tqdm import tqdm 
from g2o_transformation_handler import TransformationHandler 
from g2o_utils import *


class PointCloudsHandler(TransformationHandler):
    def __init__(self, place_folder: str, mission_folders: List[str], g2o_files: List[str], offset: int = 5000):
        super().__init__(place_folder, mission_folders, g2o_files, offset)
        # Additional initialization if needed
    
    @staticmethod
    def load_point_cloud( input_folder,scan_name ):
        # Example loading code (replace with your actual loading mechanism)
        input_file = os.path.join(input_folder,scan_name) 
        point_cloud = o3d.io.read_point_cloud(input_file)
        return point_cloud

    @staticmethod
    def transform_point_cloud( point_cloud, transform_matrix):
        # T_m2_pcd = T_m2_m1 @ T_m1_pcd  (convention)
        point_cloud.transform(transform_matrix)
        return point_cloud

    @staticmethod
    def save_transformed_point_cloud(point_cloud, output_folder, scan_name):
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder,scan_name)
        o3d.io.write_point_cloud(output_file, point_cloud)
        
    # 1) Transform pointclouds in sensor frame -> map frame
    def transform_pointclouds_base_to_map(self):
        for mission, nodes in self.dict_all.items():
            for node_data in tqdm(nodes):
                scan_name = node_data['scan_name']
                pose = node_data['pose']
                pose_matrix= pose_to_matrix(pose[:3], pose[3:])
                
                # Load and process point cloud
                input_folder= os.path.join(self.place_folder,mission,'individual_clouds')
                point_cloud = self.load_point_cloud(input_folder,scan_name)
                transformed_point_cloud = self.transform_point_cloud(point_cloud, pose_matrix)
                
                # Save transformed point cloud 
                output_folder = os.path.join(self.place_folder,mission,'individual_clouds_map')
                self.save_transformed_point_cloud(transformed_point_cloud, output_folder,f"{scan_name}")
                
    # 2) Transform pointclouds in m1 -> m2 frame                
    def transform_pointclouds_m1_to_m2(self,mission_id_1, mission_id_2):
        
        self.calculate_pose_m1_in_m2(mission_id_1, mission_id_2)
        mission_1_data = self.dict_all[self.mission_folders[mission_id_1]]
        
        # Transform point clouds
        for node_data in tqdm(mission_1_data):
            print(node_data)
            scan_name = node_data['scan_name']
            # pose = node_data['pose_m1_m2']
            T_m2_m1 = node_data['transform_T_m2_m1']
            # Load and process point cloud
            input_folder= os.path.join(self.place_folder,self.mission_folders[mission_id_1],'individual_clouds')
            point_cloud = self.load_point_cloud(input_folder,scan_name)
            transformed_point_cloud = self.transform_point_cloud(point_cloud, T_m2_m1)
            
            # Save transformed point cloud
            output_folder = os.path.join(self.place_folder,self.mission_folders[mission_id_1],'transformed_individual_clouds')
            self.save_transformed_point_cloud(transformed_point_cloud, output_folder, scan_name)

    # 3) Transform pointclouds in map frame -> base frame                
    def transform_pointclouds_map_to_base(self):
        for mission, nodes in self.dict_all.items():
            for node_data in tqdm(nodes):
                scan_name = node_data['scan_name']
                pose = node_data['pose']
                pose_matrix= pose_to_matrix(pose[:3], pose[3:])
                
                # Load and process point cloud
                input_folder= os.path.join(self.place_folder,mission,'individual_clouds_map')
                point_cloud = self.load_point_cloud(input_folder,scan_name)
                transformed_point_cloud = self.transform_point_cloud(point_cloud, np.linalg.inv(pose_matrix))
                
                # Save transformed point cloud 
                output_folder = os.path.join(self.place_folder,mission,'individual_clouds_base')
                self.save_transformed_point_cloud(transformed_point_cloud, output_folder,f"{scan_name}") 


    def save_slam_poses_pcd(self):
        """
        Save slam_poses.pcd 
        """
        for mission, nodes in self.dict_all.items():
            poses = []
            for node_data in nodes:
                pose = [float(coord) for coord in node_data['pose'][:3]]
                poses.append(pose)
            poses = np.array(poses)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(poses)
            output_file = os.path.join(self.place_folder, mission, 'slam_poses.pcd')

            o3d.io.write_point_cloud(output_file, pcd)
            print(f"Saved slam_poses.pcd for mission '{mission}' to {output_file}")



def main():
    parser = argparse.ArgumentParser(description="Process configurations for multiple missions")
    parser.add_argument('--config', default='config.yaml', type=str, help='Path to YAML config file')
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Initialize TransformationHandler with configurations
    pointcloud_handler = PointCloudsHandler(
        place_folder=config['place_folder'],
        mission_folders=config['mission_folders'],
        g2o_files=config['g2o_files'],
        offset=config['offset']
    )

    # check dict
    result = pointcloud_handler.dict_all
    print(f"Processed {len(result)} missions:")
    for mission, data in result.items():
        print(f"{mission}: {len(data)} nodes")

 # Execute actions for PointCloudProcessor
    for action in config.get('actions_pointclouds_handler', []):
        if hasattr(pointcloud_handler, action):
            print(f"Executing {action} in PointCloudProcessor...")
            getattr(pointcloud_handler, action)()
        else:
            print(f"Action {action} not found in PointCloudProcessor")

if __name__ == "__main__":
    main()
