#!/usr/bin/env python3

import os

class FileManager:
	def __init__(self, input_dir):
		self._input_dir = input_dir
		self._filenames = os.listdir(input_dir)
		self._num_files = len(self._filenames)

	def getFile(self, index):
		if index >= self._num_files:
			return None
		file = os.path.join(self._input_dir, self._filenames[index])
		return file

	def getNumFiles(self):
		return self._num_files

class PosNegFileManager(FileManager):
	def __init__(self, pos_input_dir, neg_input_dir):
		self._pos_input_dir = pos_input_dir
		self._neg_input_dir = neg_input_dir
		self._pos_filenames = os.listdir(pos_input_dir)
		self._neg_filenames = os.listdir(neg_input_dir)
		self._num_pos_files = len(self._pos_filenames)
		self._num_neg_files = len(self._neg_filenames)
		self._num_files = min(self._num_pos_files, self._num_neg_files)

	def getFile(self, index):
		if index >= self._num_files:
			return None, None
		pos_file = os.path.join(self._pos_input_dir, self._pos_filenames[index])
		neg_file = os.path.join(self._neg_input_dir, self._neg_filenames[index])
		return pos_file, neg_file

