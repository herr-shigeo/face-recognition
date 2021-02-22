#!/usr/bin/env python3

import cv2

class TrainingData:
	def __init__(self):
		self._training_data = []
		self._training_labels = []
	
	def set(self, descriptors, label):
		if descriptors is not None:
			self._training_data.extend(descriptors)
			self._training_labels.append(label)

	def get(self):
		return self._training_data, self._training_labels
			
