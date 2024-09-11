import csv
import glob
import os
import struct
import json
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import argparse
import logging

# Read Nodes

def getNode_g2o(g2o_file):
    nodes=[]
    with open(g2o_file,'r') as f:
        for line in f:
            if line.startswith("VERTEX_SE3"):
                temp = line.split(" ")[1]
                node = int(temp)
                nodes.append(node)
    return nodes

# Read Poses (x y z qx qy qz qw)

def getPoses_g2o(g2o_file):
    poses=[]
    with open(g2o_file,'r') as f:
        for line in f:
            if line.startswith("VERTEX_SE3"):
                temp = line.split(" ")[2:-2]
                poses.append(temp)
    return poses

# Read Timestamps

def getScanTime_g2o(g2o_file):
    scan_time=[]
    with open(g2o_file,'r') as f:
        for line in f:
            if line.startswith("VERTEX_SE3"):
                temp = line.split(" ")[-2:]
                temp2 = [k.replace("\n","") for k in temp]

                scan_time.append("_".join(temp2))
    return scan_time


# Read Pointcloud name 
def getScanNames_g2o(g2o_file):
    scan_names=[]
    with open(g2o_file,'r') as f:
        for line in f:
            if line.startswith("VERTEX_SE3"):
                temp = line.split(" ")[-2:]
                # First 9 digits
                temp1 = temp[0]
                # Second 9 digits 
                temp2 = temp[-1].split("\n")[0]
                if len(temp2) !=9:
                    missing_zeros = 9 - len(temp2) 
                    for i in range(missing_zeros):
                        temp2 = "0" + temp2
                scan_names.append("cloud" + "_"+ temp1 +"_" +temp2+ ".pcd")
    return scan_names


def get_translation_matrix(translation):
    """
    Convert a translation vector to a 4x4 transformation matrix.
    """
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    return matrix


def get_rotation_matrix(rotation_quat):
    """
    Convert a quaternion to a 3x3 rotation matrix.
    """
    rot = R.from_quat(rotation_quat)
    return rot.as_matrix()


def get_pose_matrix(position, quaternion):
    """
    Convert position and quaternion to a 4x4 transformation matrix.
    """
    translation_matrix = TransformationHandler.get_translation_matrix(position)
    rotation_matrix = TransformationHandler.get_rotation_matrix(quaternion)
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation_matrix
    pose_matrix[:3, 3] = position
    return pose_matrix


def extract_translation(matrix):
    """
    Extract the translation vector from a 4x4 transformation matrix.
    """
    return matrix[:3, 3]


def extract_rotation_quat(matrix):
    """
    Extract the quaternion from a 4x4 transformation matrix.
    """
    rot = R.from_matrix(matrix[:3, :3])
    return rot.as_quat()


def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

 
def pose_to_matrix(position: list, quaternion: list) -> np.ndarray:
    """
    Convert position (translation) and quaternion (rotation) to a 4x4 transformation matrix.
    """
    translation = np.array(position)
    rotation = R.from_quat(quaternion).as_matrix()
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix