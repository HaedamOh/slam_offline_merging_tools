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
        # point_cloud = transform_matrix @ T_pcd 
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
                input_folder= os.path.join(self.place_folder,self.mission_folders[mission],'individual_clouds')
                point_cloud = self.load_point_cloud(input_folder,scan_name)
                transformed_point_cloud = self.transform_point_cloud(point_cloud, pose_matrix)
                
                # Save transformed point cloud (optional)
                output_folder = os.path.join(self.place_folder,self.mission_folders[mission],'transformed_individual_clouds')
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
            T_m1_m2 = node_data['transform_T_m1_m2']
            # Load and process point cloud
            input_folder= os.path.join(self.place_folder,self.mission_folders[mission_id_1],'arial_individual_clouds')
            point_cloud = self.load_point_cloud(input_folder,scan_name)
            transformed_point_cloud = self.transform_point_cloud(point_cloud, T_m1_m2)
            
            # Save transformed point cloud
            output_folder = os.path.join(self.place_folder,self.mission_folders[mission_id_1],'transformed_individual_clouds')
            self.save_transformed_point_cloud(transformed_point_cloud, output_folder, scan_name)

    # 3) Transform pointclouds in map frame -> base frame                
    def transform_pointclouds_map_to_base(self):
        pass 


    def save_slam_poses_pcd(self):
        """
        Save poses for all missions in dict_all to individual .pcd files.
        Each mission will have its own .pcd file containing all poses from that mission.
        """
        for mission, nodes in self.dict_all.items():
            # Extract poses (x, y, z) from each node
            poses = []
            for node_data in nodes:
                pose = [float(coord) for coord in node_data['pose'][:3]]
                poses.append(pose)
            
            # Convert poses to a numpy array
            poses = np.array(poses)
            
            # Create an Open3D PointCloud object and add the points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(poses)
            
            # Define output file path
            output_file = os.path.join(self.place_folder, mission, 'slam_poses.pcd')
            
            # Save the point cloud to the .pcd file
            o3d.io.write_point_cloud(output_file, pcd)
            print(f"Saved poses for mission '{mission}' to {output_file}")



def main():
    # Should be copy and paste from g2o_transformation_handler.py # 
    parser = argparse.ArgumentParser(description="Process point clouds for multiple missions")
    parser.add_argument('--place_folder', default='/media/haedam/T7', type=str, help='Base folder for all missions')
    parser.add_argument('--mission_folders', default=['2023-08-15-13-12-48-exp20-d2', '2023-08-15-13-12-48-exp20-d2-arial'], type=str, nargs='+', help='List of mission folder names')
    parser.add_argument('--g2o_files', default=['/media/haedam/T7/2023-08-15-13-12-48-exp20-d2/slam_pose_graph.g2o', '/media/haedam/T7/2023-08-15-13-12-48-exp20-d2/optimized_pose_graph_aerial_constraints.g2o'], type=str, nargs='+', help='G2O file(s) to process')
    parser.add_argument('--offset', default=0, type=int, help='Node offset for missions (default: 5000)')
    args = parser.parse_args()

    # Create an instance of PointCloudProcessor
    handler = PointCloudsHandler(
        place_folder=args.place_folder,
        mission_folders=args.mission_folders,
        g2o_files=args.g2o_files,
        offset=args.offset
    )

    # Print processed missions and number of nodes
    result = handler.dict_all
    print(f"Processed {len(result)} missions:")
    for mission, data in result.items():
        print(f"  {mission}: {len(data)} nodes")



    # Process point clouds if the flag is set
    print("Processing point clouds...")
    handler.transform_pointclouds_m1_to_m2(0,1)
    print("Point clouds processing complete.")

if __name__ == "__main__":
    main()
