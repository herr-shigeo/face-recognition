#!/usr/bin/env python3

import cv2
import dlib
import sys
import os
import argparse
import logging

logger = logging.getLogger()

SAVED_IMAGE_SIZE = (200, 200)

class faceDetector:
	pass

class HaarDetector(faceDetector):
	def __init__(self):
		self.__detector = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
	def run(self, img):
		return self.__detector.detectMultiScale(img, 1.08, 5, minSize=SAVED_IMAGE_SIZE)
	def getVertexes(self, face):
		(x, y, w, h) = face
		return (x, y, w, h)

class HogDetector(faceDetector):
	def __init__(self):
		self.__detector = dlib.get_frontal_face_detector()
	def run(self, img):
		return self.__detector(img, 1)
	def getVertexes(self, face):
		x0 = max(0, face.left())
		y0 = max(0, face.top())
		x1 = max(0, face.right())
		y1 = max(0, face.bottom())
		return (x0, y0, x1-x0+1, y1-y0+1)

def detectAndSaveFace(input_dir, output_dir, useHaar):
	logging.basicConfig(level=logging.INFO, format='%(message)s')

	if useHaar:
		detector = HaarDetector()
	else:
		detector = HogDetector()

	files_processed = 0
	faces_detected = 0

	for filename in os.listdir(input_dir):
		orig_img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE)	
		if orig_img is None:
			logger.warning('ignoring {0}'.format(filename))
			continue

		faces = detector.run(orig_img)
		logger.debug('Detected({0}) in {1}'.format(len(faces), filename))
		files_processed += 1

		# Save the detected bounding boxes
		for face in faces:
			(x, y, w, h) = detector.getVertexes(face)
			face_img = cv2.resize(orig_img[y:y+h, x:x+w], SAVED_IMAGE_SIZE)
			face_filename = '{0}/{1}.pgm'.format(output_dir, faces_detected)
			cv2.imwrite(face_filename, face_img)
			faces_detected += 1

	logger.info('files processed = {0}, faces detected = {1}'.format(files_processed, faces_detected))
	return 0

def reindexFiles(input_dir):
	index = 0
	for filename in os.listdir(input_dir):
		orig_file = os.path.join(input_dir, filename)
		new_file  = os.path.join(input_dir, 'new_{0}.pgm'.format(index))
		os.rename(orig_file, new_file)
		index += 1
	for filename in os.listdir(input_dir):
		orig_file = os.path.join(input_dir, filename)
		new_file = os.path.join(input_dir, filename.split('_')[1])
		os.rename(orig_file, new_file)
	return 0

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--haar', action='store_true', help='use Haar cascades')
	parser.add_argument('-i', '--input-dir', action='store')
	parser.add_argument('-o', '--output-dir', action='store')
	parser.add_argument('-v', '--verbose', action='store_true')
	parser.add_argument('--reindex', action='store_true')

	args = parser.parse_args(sys.argv[1:])

	if args.verbose:
		logging.basicConfig(level=logging.DEBUG, format='%(message)s')

	if args.reindex:
		if args.input_dir is None:
			parser.print_help()
			sys.exit(1)
		sys.exit(reindexFiles(args.input_dir))

	if args.input_dir is None or args.output_dir is None:
		parser.print_help()
		sys.exit(1)
	
	sys.exit(detectAndSaveFace(args.input_dir, args.output_dir, args.haar))
