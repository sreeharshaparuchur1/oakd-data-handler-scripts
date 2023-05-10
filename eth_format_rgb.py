#!/usr/bin/env python3
import numpy as np
import glob
import os
import argparse
#TODO: Add saving the calibration.txt files.

parser = argparse.ArgumentParser(description="To specify the data dump to parse")
parser.add_argument("--room", required=True,
                     help="Name the room specific dump to generate related scripts")

args = parser.parse_args()

roomNumber  = args.room #"306_0812"
rgb_path = os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "rgb")

files_rgb = sorted(glob.glob(os.path.join(rgb_path, "*")))
print("Number of rgb files found: ", len(files_rgb))

with open(os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "rgb.txt"), 'w') as fi:
    for data in files_rgb:
        to_splat = data.split('/')[-2:]
        fi.write(f"{data.split('/')[-1].split('.')[0]} {os.path.join('./', *to_splat)}\n")
    fi.close

depth_path = os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "depth")

files_depth = sorted(glob.glob(os.path.join(depth_path, "*")))
print("Number of depth files found: ",len(files_depth)) 

with open(os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "depth.txt"), 'w') as fi:
    for data in files_depth: 
        to_splat = data.split('/')[-2:]
        fi.write(f"{data.split('/')[-1].split('.')[0]} {os.path.join('./', *to_splat)}\n")
    fi.close

with open(os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "associated.txt"), 'w') as fi:
    for idx in range(min(len(files_rgb), len(files_depth))): 
        to_splat_rgb = files_rgb[idx].split('/')[-2:]
        to_splat_depth = files_depth[idx].split('/')[-2:]
        fi.write(f"{files_rgb[idx].split('/')[-1].split('.')[-2]} {os.path.join('./', *to_splat_rgb)} {files_depth[idx].split('/')[-1].split('.')[-2]} {os.path.join('./', *to_splat_depth)}\n")
    fi.close
print("The number of data points in assosciated.txt: ", min(len(files_rgb), len(files_depth)))

