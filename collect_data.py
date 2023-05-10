#!/usr/bin/env python3
#https://learnopencv.com/introduction-to-opencv-ai-kit-and-depthai/
import math
import cv2
import numpy as np
import depthai as dai
import time
from datetime import datetime
import glob
import os
import argparse
import multiprocessing


parser = argparse.ArgumentParser(description='Give parameters for data collection.')
parser.add_argument('--extended_disparity', default=0, 
                    help='Closer-in minimum depth, disparity range is doubled (from 95 to 190)')
parser.add_argument('--subpixel', default=0,
                    help='Better accuracy for longer distance, fractional disparity 32-levels')
parser.add_argument('--lrcheck', default=1,
                    help='Better handling for occlusions. This has to be true for center alignment') #Disparity/depth CENTER alignment requires left-right check mode enabled!
parser.add_argument('--fps', default=30,
                    help='frames per second')
parser.add_argument('--median_filter', default=0,
                    help='Set median filter off or at 7X7')
parser.add_argument('--room_number', required=True,
                    help='frames per second')
parser.add_argument('--dot_projector_current', required=False, default = 0,
                    help='Name says all, max value 1200mah')
parser.add_argument('--flood_light_current', required=False, default = 0,
                    help='Name says all, max value 1500mah')
parser.add_argument('--clear_tree', required=False, default=1,
                    help='Clear previous data')

args = parser.parse_args()
### Initialize Variables:

# Remove previous data at the dump locations
clearTree = bool(args.clear_tree)
# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = bool(args.extended_disparity)
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = bool(args.subpixel) #True
# Better handling for occlusions:
lr_check = bool(args.lrcheck)
fps = int(args.fps)
roomNumber = args.room_number #"306_1812" #+str(datetime.now().strftime('%Y%m%d%H%M%S'))
dotProjectorCurrent = int(args.dot_projector_current)
floodLightCurrent = int(args.flood_light_current)

# Optional. If set (True), the ColorCamera is downscaled from 1080p to 720p.
# Otherwise (False), the aligned depth is automatically upscaled to 1080p
downscaleColor = True
# The disparity is computed at this resolution, then upscaled to RGB resolution
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P

# Create pipeline
pipeline = dai.Pipeline()
device = dai.Device()
queueNames = []

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

rgbOut = pipeline.create(dai.node.XLinkOut)
xoutMonoLeft = pipeline.create(dai.node.XLinkOut)
outStereoDepth = pipeline.create(dai.node.XLinkOut)

rgbOut.setStreamName("rgb")
queueNames.append("rgb")
xoutMonoLeft.setStreamName('synced_mono_left')
queueNames.append("synced_mono_left")
outStereoDepth.setStreamName('oakd_stereo_depth')
queueNames.append("oakd_stereo_depth")


#Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setFps(fps)
if downscaleColor: camRgb.setIspScale(2, 3)
# For now, RGB needs fixed focus to properly align with depth.
# This value was used during calibration
try:
    calibData = device.readCalibration2()
    print(calibData) # to know what format the data is and what the defaults are
    lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
    if lensPosition:
        camRgb.initialControl.setManualFocus(lensPosition)
except:
    raise
# print(dir(calibData))
print(f"calib data for the LEFT mono: \n{calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT)} \nCalib data for RGB: \n{calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB)}")
 
left.setResolution(monoResolution)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
left.setFps(fps)

right.setResolution(monoResolution)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
right.setFps(fps)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

stereo.setLeftRightCheck(lr_check)
stereo.setExtendedDisparity(extended_disparity)
stereo.setSubpixel(subpixel)

# Linking
camRgb.isp.link(rgbOut.input)
left.out.link(stereo.left)
right.out.link(stereo.right)
stereo.syncedLeft.link(xoutMonoLeft.input)
stereo.depth.link(outStereoDepth.input)

### Python code to create the data dump heirarchy:

if args.median_filter:
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
else:
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)

os.makedirs(os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "rgb"), exist_ok = True)
os.makedirs(os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "depth"), exist_ok = True)
os.makedirs(os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "mono_left"), exist_ok = True)

if clearTree:
    for removee in ["rgb", "depth", "mono_left"]:
        for toRemove in glob.glob(os.path.join("/home/harsha/Perception/oakd/data", roomNumber, removee, "*")):
            os.remove(toRemove)

def dump_data(rgb_path, rgb_data, depth_path, depth_data, mono_path, mono_data):
    rgb_data = cv2.resize(rgb_data, (640, 480))
    depth_data = cv2.resize(depth_data, (640, 480))
    mono_data = cv2.resize(mono_data, (640, 480))
    cv2.imwrite(rgb_path, rgb_data)
    cv2.imwrite(depth_path, depth_data)
    cv2.imwrite(mono_path, mono_data)

pool = multiprocessing.Pool(8) # #https://gist.github.com/youralien/b4943a221378c6ef064f

coordinates = (100,100)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
# color = (255,0,255)
color = (0,255,0) # It doesn't matter which colour we choose becuase the mono channel is greyscale in nature.
thickness = 2
closest_point = 0.00000

# Connect to device and start pipeline
with device:
    device.startPipeline(pipeline)
    device.setIrLaserDotProjectorBrightness(dotProjectorCurrent) # in mA, 0..1200
    device.setIrFloodLightBrightness(floodLightCurrent)

    frameRgb = None
    frameDepth = None    

    start_time = time.time()
    fps_to_check = 30
    fps_counter=0
    while True:
        latestPacket = {}
        latestPacket["rgb"] = None
        latestPacket["synced_mono_left"] = None
        latestPacket["oakd_stereo_depth"] = None

        
        queueEvents = device.getQueueEvents(("rgb",  "oakd_stereo_depth", "synced_mono_left")) # to add mono, maybe IMU. # pull timestamp from here. a .get method for each packet.
        for queueName in queueEvents:
            packets = device.getOutputQueue(name=queueName, maxSize=1, blocking=True).tryGetAll()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1]

        fps_counter += 1
        if fps_counter == fps_to_check:
            end_time = time.time()
            print(f"Time taken to dump {fps_to_check} frames is: {round(end_time - start_time, 5)} seconds")
            fps_counter = 0
            start_time = end_time
        dt = datetime.now().strftime('%Y%m%d%H%M%S%f')

        if latestPacket["rgb"] is not None:
            #dt = str(latestPacket["rgb"].getSequenceNum())
            frameRgb = latestPacket["rgb"].getCvFrame()
            if latestPacket["rgb"].getSequenceNum() < 30:
                # Introduced to eiminate cold-start abberations
                continue

        if latestPacket["oakd_stereo_depth"] is not None:
            frameDepth = latestPacket["oakd_stereo_depth"].getCvFrame()
            if latestPacket["oakd_stereo_depth"].getSequenceNum() < 30:
                # Introduced to eiminate cold-start abberations
                continue

            # Enable to get the 10th percentile value of depth (in mm) to know if the camera has too much of the scene in its 'blind spot'
            # closest_point = np.percentile(frameDepth[np.nonzero(frameDepth)].flatten(), 10)
            # print(closest_point)

            # print(f"Max value for depth: {np.max(frameDepth)} \nMin value for depth: {np.min(frameDepth)}")
            # cv2.imwrite(os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "depth", dt + ".png"), frameDepth)

        if latestPacket["synced_mono_left"] is not None:
            frameMono = latestPacket["synced_mono_left"].getCvFrame()
            if latestPacket["synced_mono_left"].getSequenceNum() < 30:
                # Introduced to eiminate cold-start abberations
                continue
            # cv2.imwrite(os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "mono_left", dt + ".png"), frameMono)
            # cv2.putText(frameMono, f"{closest_point:.2f}", coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow("global shutter mono", frameMono)

        if frameDepth is not None and frameRgb is not None:
            '''
            try:
                Estimating the latency introduced by the percentile operation on our script
                cp_time = time.time()
                closest_point = np.percentile(frameDepth[np.nonzero(frameDepth)].flatten(), 10)
                closest_point = np.nanmin(frameDepth[np.nonzero(frameDepth)])
                print("cp time = ", time.time()-cp_time)
            except:
                pass
            '''
            pool.apply_async(dump_data, (os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "rgb", dt + ".png"), frameRgb, os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "depth", dt + ".png"), frameDepth,  os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "mono_left", dt + ".png"), frameMono))
        
        if cv2.waitKey(1) == ord("/"):
            break
