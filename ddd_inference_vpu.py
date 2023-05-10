#!/usr/bin/env python3

from pathlib import Path
import sys
import numpy as np
import cv2
import depthai as dai
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("nnPath", type=str)
args = ap.parse_args()

p = dai.Pipeline()
p.setOpenVINOVersion(dai.OpenVINO.VERSION_2022_1)

# Define sources and outputs
monoRight = p.create(dai.node.MonoCamera)
monoLeft = p.create(dai.node.MonoCamera)
stereo = p.create(dai.node.StereoDepth)
nn_xout = p.create(dai.node.XLinkOut)
mono_xout = p.create(dai.node.XLinkOut)

nn = p.createNeuralNetwork()
nn.setBlobPath(args.nnPath)
nn.setNumInferenceThreads(2)

# Properties
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# Linking
monoRight.out.link(stereo.right)
monoLeft.out.link(stereo.left)
stereo.depth.link(nn.input)
nn.out.link(nn_xout.input)
monoLeft.out.link(mono_xout.input)

nn_xout.setStreamName("nn_depth")
mono_xout.setStreamName("mono")

# Pipeline is defined, now we can connect to the device
with dai.Device(p) as device:
    frameMono = None
    frameNNdepth = None
    
    while True:
        latestPacket = {}
        latestPacket["mono"] = None
        latestPacket["nn_depth"] = None

        queueEvents = device.getQueueEvents(("mono", "nn_depth"))
        for queueName in queueEvents:
            packets = device.getOutputQueue(queueName).tryGetAll()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1]

        if latestPacket["mono"] is not None:
            frameMono = latestPacket["mono"].getCvFrame()
            cv2.imshow("mono", frameMono)

        if latestPacket["nn_depth"] is not None:
            frameNNdepth = latestPacket["nn_depth"].getData()
            #print(frameNNdepth.shape, frameNNdepth.dtype)
            frameNNdepth = np.frombuffer(frameNNdepth.tobytes(), dtype=np.uint16).reshape(360,640)
            print(frameNNdepth.shape, frameNNdepth.dtype)
            print(np.nanmax(frameNNdepth),np.nanmin(frameNNdepth),np.nanmean(frameNNdepth))
            #print(latestPacket["nn_depth"].getAllLayerNames())
            cv2.imshow("nn_depth", frameNNdepth)

        if cv2.waitKey(1) == ord('q'):
            break
