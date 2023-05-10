#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import time
from datetime import datetime
import glob
import os
# date = datetime. now(). strftime("%Y_%m_%d-%I:%M:%S_%p")
# print(f"filename_{date}")
# 'filename_2020_08_12-03:29:22_AM'
#https://learnopencv.com/introduction-to-opencv-ai-kit-and-depthai/
# Weights to use when blending rgb/depth/confidence image
rgbWeight = 0.3
depthWeight = 0.3
confWeight = 0.3
# Normalized weights to use when blending rgb/depth/confidence image (should equal 1.0)
rgbWeightNorm = 0.3
depthWeightNorm = 0.3
confWeightNorm = 0.3

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = True
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True


def updateRgbBlendWeights(percent):
    """
    Update the rgb weight used to blend rgb/depth/confidence image

    @param[in] percent The rgb weight expressed as a percentage (0..100)
    """
    global rgbWeight
    rgbWeight = float(percent)/100.0

def updateDepthBlendWeights(percent):
    """
    Update the depth weight used to blend rgb/depth/confidence image

    @param[in] percent The depth weight expressed as a percentage (0..100)
    """
    global depthWeight
    depthWeight = float(percent)/100.0

def updateConfBlendWeights(percent):
    """
    Update the confidence weight used to blend rgb/depth/confidence image

    @param[in] percent The confidence weight expressed as a percentage (0..100)
    """
    global confWeight
    confWeight = float(percent)/100.0

# Optional. If set (True), the ColorCamera is downscaled from 1080p to 720p.
# Otherwise (False), the aligned depth is automatically upscaled to 1080p
downscaleColor = True
fps = 30
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
disparityOut = pipeline.create(dai.node.XLinkOut)
# xoutConfMap = pipeline.create(dai.node.XLinkOut)
# xoutMonoLeft = pipeline.create(dai.node.XLinkOut)
# outStereoDepth = pipeline.create(dai.node.XLinkOut)
# xoutLeft = pipeline.createXLinkOut()
# xoutLeft.setStreamName("left")

rgbOut.setStreamName("rgb")
queueNames.append("rgb")
disparityOut.setStreamName("disp")
queueNames.append("disp")
# xoutConfMap.setStreamName('confidence_map')
# queueNames.append("confidence_map")
# xoutMonoLeft.setStreamName('synced_mono_left')
# queueNames.append("synced_mono_left")
# outStereoDepth.setStreamName('oakd_stereo_depth')
# queueNames.append("oakd_stereo_depth")


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

left.setResolution(monoResolution)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
left.setFps(fps)

right.setResolution(monoResolution)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
right.setFps(fps)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# LR-check is required for depth alignment
# stereo.setLeftRightCheck(True)
# stereo.setSubpixel(True)  # TODO enable for test
# if 0: stereo.setSubpixel(True)  # TODO enable for test
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
#stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(lr_check)
stereo.setExtendedDisparity(extended_disparity)
stereo.setSubpixel(subpixel)

# Linking
camRgb.isp.link(rgbOut.input)
left.out.link(stereo.left)
right.out.link(stereo.right)
stereo.disparity.link(disparityOut.input)
# stereo.confidenceMap.link(xoutConfMap.input)
# stereo.syncedLeft.link(xoutMonoLeft.input)
# stereo.depth.link(outStereoDepth.input)
# left.out.link(xoutLeft.input)
# stereo.disparity.link(disparityOut.input)

### Initialize Variables:
clearTree = True
roomNumber = "306_1712"

### Python code to create the data dump heirarchy:

os.makedirs(os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "rgb"), exist_ok = True)
# os.makedirs(os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "depth"), exist_ok = True)
os.makedirs(os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "disparity"), exist_ok = True)
# os.makedirs(os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "mono_left"), exist_ok = True)


if clearTree:
    for removee in ["rgb", "disparity"]:
        for toRemove in glob.glob(os.path.join("/home/harsha/Perception/oakd/data", roomNumber, removee, "*")):
            os.remove(toRemove)

# Connect to device and start pipeline
with device:
    device.startPipeline(pipeline)

    frameRgb = None
    frameDisp = None    
    frameC = None

    # Configure windows; trackbar adjusts blending ratio of rgb/depth
    rgbWindowName = "rgb"
    depthWindowName = "depth"
    # confWindowName = "conf"
    # blendedWindowName = "rgb-depth-conf"
    # cv2.namedWindow(rgbWindowName)
    # cv2.namedWindow(depthWindowName)
    # cv2.namedWindow(confWindowName)
    # cv2.namedWindow(blendedWindowName)
    # cv2.createTrackbar('RGB Weight %', blendedWindowName, int(rgbWeight*100), 100, updateRgbBlendWeights)
    # cv2.createTrackbar('Depth Weight %', blendedWindowName, int(depthWeight*100), 100, updateDepthBlendWeights)
    # cv2.createTrackbar('Confidence Weight %', blendedWindowName, int(confWeight*100), 100, updateConfBlendWeights)
    
    start_time = time.time()
    fps_to_check = 30
    fps_counter=0
    while True:
        latestPacket = {}
        latestPacket["rgb"] = None
        latestPacket["disp"] = None
        # latestPacket["confidence_map"] = None
        # latestPacket["synced_mono_left"] = None
        # latestPacket["oakd_stereo_depth"] = None

        
#"disp",
        queueEvents = device.getQueueEvents(("rgb",  "disp")) # to add mono, maybe IMU. # pull timestamp from here. a .get method for each packet.
        for queueName in queueEvents:
            # packets = device.getOutputQueue(queueName).tryGetAll()
            packets = device.getOutputQueue(name=queueName, maxSize=1, blocking=False).tryGetAll()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1]

        fps_counter += 1
        if fps_counter == fps_to_check:
            end_time = time.time()
            print(f"Time taken to dump {fps_to_check} frames is: {round(end_time - start_time, 5)} seconds")
            fps_counter = 0
            start_time = end_time
        # print(latestPacket.keys(), fps_to_check)
        dt = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        if latestPacket["rgb"] is not None:
            frameRgb = latestPacket["rgb"].getCvFrame()
            cv2.imwrite(os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "rgb", dt + ".png"), frameRgb)
            # cv2.imshow(rgbWindowName, frameRgb)

        # if latestPacket["oakd_stereo_depth"] is not None:
        #     frameDepth = latestPacket["oakd_stereo_depth"].getCvFrame()
        #     cv2.imwrite(os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "depth", dt + ".png"), frameDepth)

        # if latestPacket["synced_mono_left"] is not None:
        #     frameMono = latestPacket["synced_mono_left"].getCvFrame()
        #     cv2.imwrite(os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "mono_left", dt + ".png"), frameMono)
        #     cv2.imshow(depthWindowName, frameMono)

        # if latestPacket["confidence_map"] is not None:
        #     frameC = latestPacket["confidence_map"].getCvFrame()
            # cv2.imshow(confWindowName, frameC)

        if latestPacket["disp"] is not None:
            frameDisp = latestPacket["disp"].getFrame()
            maxDisparity = stereo.initialConfig.getMaxDisparity()
            frameDispDump = (frameDisp * 65535. / maxDisparity).astype(np.uint16)
            # #Optional, extend range 0..95 -> 0..255, for a better visualisation
            # if 1: frameDisp = (frameDisp * 255. / maxDisparity).astype(np.uint8)
            # # Optional, apply false colorization
            # if 1: frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_HOT)  #display a falsely coloured  disparity
            # frameDisp = np.ascontiguousarray(frameDisp) # convert numpy type to uint16.
            frameDispDump = np.ascontiguousarray(frameDispDump) # convert numpy type to uint16.

            cv2.imwrite(os.path.join("/home/harsha/Perception/oakd/data", roomNumber, "disparity", dt + ".png"), frameDispDump)

        # Blend when all three frames received
        # if frameRgb is not None and frameDisp is not None and frameC is not None:
        #     # Need to have all three frames in BGR format before blending
        #     if len(frameDisp.shape) < 3:
        #         frameDisp = cv2.cvtColor(frameDisp, cv2.COLOR_GRAY2BGR)
        #     if len(frameC.shape) < 3:
        #         frameC = cv2.cvtColor(frameC, cv2.COLOR_GRAY2BGR)
        #     sumWeight = rgbWeight + depthWeight + confWeight
        #     # Normalize the weights so their sum to be <= 1.0
        #     if sumWeight <= 1.0:
        #         rgbWeightNorm = rgbWeight
        #         depthWeightNorm = depthWeight
        #         confWeightNorm = confWeight
        #     else :
        #         rgbWeightNorm = rgbWeight / sumWeight
        #         depthWeightNorm = depthWeight / sumWeight
        #         confWeightNorm = confWeight / sumWeight
        #     blended1 = cv2.addWeighted(frameRgb, rgbWeightNorm, frameDisp, depthWeightNorm, 0)
        #     blended2 = cv2.addWeighted(blended1, rgbWeightNorm + depthWeightNorm, frameC, confWeightNorm, 0)
        #     cv2.imshow(blendedWindowName, blended2)
            # frameRgb = None
            # frameDisp = None
            # frameC = None

        if cv2.waitKey(1) == ord('q'):
            break
