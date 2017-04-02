import csv
import os
import numpy as np
import tensorflow as tf
import cv2

DATASET_DIR = "../datasets/WikiArt/"
IMG_DIR = DATASET_DIR + 'wikiart/'
INFO_DIR = DATASET_DIR + 'train_info_local.csv'
NEW_INFO_DIR = DATASET_DIR + 'train_info_new.csv'
STYLES_DIR = 'style list.txt'

ENTRIES = ['filename', 'artist', 'title', 'style', 'genre', 'date']

# read images and styles
# only read nimages from each style
def read_data(nimages, style_index):
	styles = get_style_names(style_index)
	counts = dict(map(lambda s: [s, 0], styles))
	data = []
	with open(INFO_DIR, 'rb') as ifile:
		reader = csv.reader(ifile, delimiter=',')
		for row in reader:
			style = row[3]
			# file_exists = os.path.exists(IMG_DIR+row[0])
			if (style in styles) and (counts[style] < nimages):
				data.append([row[0], styles.index(style)])
				counts[style] += 1
	return data, styles


def get_style_names(style_index):
	with open(STYLES_DIR, 'r') as sfile:
		styles = sfile.read().split('\n')
		styles = map(lambda n: ' '.join(n.split(' ')[:-1]), styles)
	styles = [styles[i] for i in style_index]
	return styles


def shorten_csv():
	mypath = IMG_DIR
	onlyfiles = [f for f in os.listdir(mypath)]
	allimgs = set(onlyfiles)
	with open(INFO_DIR, 'rb') as file:
		with open(NEW_INFO_DIR, 'wb') as outfile:
			reader = csv.reader(file, delimiter=',')
			writer = csv.writer(outfile, delimiter=',')
			for row in reader:
				name = row[0]
				if name in allimgs and len(row[3]) >= 2:
					writer.writerow(row)
				else:
					print 'removing', name
	print 'shortening done.'


def style_sheet():
	genres = {}
	c = 0
	with open(INFO_DIR, 'rb') as file:
		reader = csv.reader(file, delimiter=',')
		for row in reader:
			g = row[3]
			if g:
				c += 1
				if g in genres:
					genres[g] += 1
				else:
					genres[g] = 1
	sgenres = sorted(genres, key=genres.get, reverse=True)
	for g in sgenres:
		print g, genres[g]


def scan_images():
	with open(INFO_DIR, 'rb') as file:
		reader = csv.reader(file, delimiter=',')
		for row in reader:
			name = IMG_DIR+row[0]
			im_encoded = tf.gfile.GFile(name, 'r').read()
			if im_encoded[:3] != '\xff\xd8\xff':
				print name
				# os.remove(name)
				# im = cv2.imread(name)
				# cv2.imwrite(row[0], im)
	print 'scan_images done.'


def scan_csv():
	c = 0
	with open(INFO_DIR, 'rb') as file:
		reader = csv.reader(file, delimiter=',')
		for row in reader:
			name = row[0]
			if name[-4:] != '.jpg':
				print name
				c += 1
	print c
	print 'checking done.'

# shorten_csv()
# scan_csv()
# scan_images()
# style_sheet()
