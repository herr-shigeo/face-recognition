#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import os
import argparse
import logging
import util
import pickle
from matplotlib import pyplot as plt

logger = logging.getLogger()

BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 30
SVM_NUM_TRAINING_SAMPLES_PER_CLASS = 100
BOW_NUM_CLUSTERS = 40

class MyBoW:
	def __init__(self, dextractor, dmatcher, cluster_count):
		self._dextractor = dextractor
		self._trainer = cv2.BOWKMeansTrainer(cluster_count)
		self._extractor = cv2.BOWImgDescriptorExtractor(dextractor, dmatcher)

	def addSample(self, path):
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		keypoints, descriptors = self._dextractor.detectAndCompute(img, None)
		if descriptors is None:
			logger.debug('No descriptor genearted')
		else:
			self._trainer.add(descriptors)

	def createVocAndSave(self, data_file=None):
		voc = self._trainer.cluster()
		self._extractor.setVocabulary(voc)
		if data_file is not None:
			with open(data_file, 'wb') as f:
				pickle.dump(voc, f)	

	def loadVoc(self, data_file):
		if data_file is not None:
			with open(data_file, 'rb') as f:
				voc = pickle.load(f)
			self._extractor.setVocabulary(voc)

	def extractDescriptors(self, img):
		keypoints = self._dextractor.detect(img)
		return self._extractor.compute(img, keypoints)

class MySVM:
	def __init__(self):
		self._svm = cv2.ml.SVM_create()
	
	def train(self, training_data, training_labels, data_file):
		aa = np.array(training_data)
		self._svm.train(np.array(training_data), cv2.ml.ROW_SAMPLE,
						np.array(training_labels))
		if data_file is not None:
			self._svm.save(data_file)

	def config(self, count):
		self._svm.setType(cv2.ml.SVM_C_SVC)
		self._svm.setKernel(cv2.ml.SVM_LINEAR)
		self._svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, count, 1e-6))		

	def load(self, data_file):
		new_svm = self._svm.load(data_file)
		self._svm = new_svm

	def predict(self, descriptros):
		return self._svm.predict(descriptros)

class MyMatcher:
	def createFlann():
		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
		search_params = {}
		return cv2.FlannBasedMatcher(index_params, search_params)

def train(file_manager, voc_file, svm_file):
	# create a sift and a flann
	sift = cv2.SIFT_create()
	flann = MyMatcher.createFlann()

	# create a BoW KMeans Trainer
	bow_trainer = MyBoW(sift, flann, BOW_NUM_CLUSTERS)

	# add samples to the trainer
	for i in range(BOW_NUM_TRAINING_SAMPLES_PER_CLASS):
		pos_path, neg_path = file_manager.getFile(i)
		bow_trainer.addSample(pos_path)
		bow_trainer.addSample(neg_path)

	# create vocabulary
	bow_trainer.createVocAndSave(voc_file)

	# prepare training data with BoW decriptors
	training_data = util.TrainingData()
	for i in range(SVM_NUM_TRAINING_SAMPLES_PER_CLASS):
		pos_path, neg_path = file_manager.getFile(i)	
		pos_img = cv2.imread(pos_path, cv2.IMREAD_GRAYSCALE)
		neg_img = cv2.imread(neg_path, cv2.IMREAD_GRAYSCALE)
		pos_descriptors = bow_trainer.extractDescriptors(pos_img)
		neg_descriptors = bow_trainer.extractDescriptors(neg_img)
		training_data.set(pos_descriptors, 1)
		training_data.set(neg_descriptors, -1)

	# create a SVM and train it
	svm = MySVM()
	#svm.config(100)
	data, labels = training_data.get()
	svm.train(data, labels, svm_file)

def predict(file_manager, voc_file, svm_file):
	# create a sift and a flann
	sift = cv2.SIFT_create()
	flann = MyMatcher.createFlann()

	# create a Bow KMeans Trainer and the load vocaabulary
	bow_trainer = MyBoW(sift, flann, BOW_NUM_CLUSTERS)
	bow_trainer.loadVoc(voc_file)

	# create a SVM and load the data
	svm = MySVM()
	svm.load(svm_file)

	# predict
	for i in range(file_manager.getNumFiles()):
		path = file_manager.getFile(i)
		img = cv2.imread(path)
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		descriptors = bow_trainer.extractDescriptors(gray_img)
		result = svm.predict(descriptors)
		logger.debug('file={0}, result={1}'.format(path, result))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--verbose', action='store_true')
	parser.add_argument('-t', '--train', action='store_true')
	parser.add_argument('-pd', '--pos-input-dir', action='store')
	parser.add_argument('-nd', '--neg-input-dir', action='store')
	parser.add_argument('-td', '--test-input-dir', action='store')
	parser.add_argument('-vf', '--voc-file', action='store')
	parser.add_argument('-sf', '--svm-file', action='store')
	parser.add_argument('-p', '--predict', action='store_true')

	args = parser.parse_args(sys.argv[1:])

	if args.verbose:
		logging.basicConfig(level=logging.DEBUG, format='%(message)s')

	done = False

	if args.train:
		if args.pos_input_dir is None or args.neg_input_dir is None:
			parser.print_help()
			sys.exit(1)	
		file_manager = util.PosNegFileManager(args.pos_input_dir, args.neg_input_dir)
		train(file_manager, args.voc_file, args.svm_file)
		done = True

	if args.predict:
		if args.voc_file is None or args.svm_file is None or args.test_input_dir is None:
			parser.print_help()
			sys.exit(1)	
		file_manager = util.FileManager(args.test_input_dir)
		predict(file_manager, args.voc_file, args.svm_file)
		done = True

	if not done:
		parser.print_help()
		sys.exit(1)	

	sys.exit(0)
		
