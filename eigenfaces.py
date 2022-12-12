# import the necessary packages
from imutils import paths
import numpy as np
import cv2
import os
from os import walk



def load_face_dataset(inputPath, minSamples):
	f = []
	for (dirpath, dirnames, filenames) in walk(inputPath):
		f.extend(filenames)
		break

(faces, labels) = load_face_dataset("Samples/", minSamples=20)