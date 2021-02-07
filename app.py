#!/usr/bin/env python3

import sys
import io
import os
import tempfile
import datetime
from bottle import route, run, template, request, static_file

TMPDIR = 'tmp'

@route('/')
def index():
	return static_file('top.html', root='.')

@route('/upload', method='POST')
def upload():
	upload = request.files.get('upload')
	name, ext = os.path.splitext(upload.filename)
	if ext not in ('.png', '.jpg', '.jpeg'):
		return "File extension is not supported!\nPlease upload a png for jpg(jepg) file."

	if not os.path.exists(TMPDIR):
		os.mkdir(TMPDIR)
	timestamp = datetime.datetime.now().strftime("%Y%b%d")
	file_path = "{tmpdir}/{prefix}-{file}".format(tmpdir=TMPDIR, prefix=timestamp, file=upload.filename)
	upload.save(file_path)
	return template('result.tpl', image_path=file_path)

@route('/tmp/<picture>')
def serve_pictures(picture):
	return static_file(picture, root='tmp')
	
run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
