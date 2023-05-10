#!/usr/bin/env python3
import math

width, height = 640, 480
baseline = 75
fov = 71.86

focal = width / (2 * math.tan(fov / 2 / 180 * math.pi))

print(focal)