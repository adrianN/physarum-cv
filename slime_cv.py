import cv2 as cv

import numpy as np
from scipy.ndimage import label
import sys
import os

import time

def display(img):
	width, height = img.shape[0], img.shape[1]
	tmp = cv.resize(img, (int(height/1.5), int(width/1.5)))
	cv.imshow('w',tmp)
	cv.waitKey(500)

step = 2
block_size = 7
darkness = 20
darkness_zero = 35
component_size = 250

def schange(v):
	global step
	if step < v:
		while step < v:
			step += 1
			update_display()
	else:
		step = v
		update_display()

def bchange(v):
	global block_size
	if v>0:
		block_size = v
		contours()

def dchange(v):
	global darkness
	darkness = v
	contours()

def cchange(v):
	global component_size
	component_size = v
	contours()


def update_display():
	global step
	if step == 0:
		display(img_gray)
	if step == 1:
		contours()
	if step == 2:
		calculate_graph()

def setup():
	try:
		filename = sys.argv[1]
	except IndexError:
		filename = 'physarum.jpg'
	if not os.path.isfile(filename):
		raise IOError("That file does not exist")

	img = cv.imread(filename)
	cv.namedWindow('w')
	cv.createTrackbar('step', 'w', step, 2, schange)
	cv.createTrackbar('block size', 'w', block_size, 40, bchange)
	cv.createTrackbar('darkness', 'w', darkness, 60, dchange )
	cv.createTrackbar('component size', 'w', component_size, 1000, cchange)
	return img

def contours():
	now = time.clock()
	print "Threshold"
	img_bin = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 2*block_size+1, darkness - darkness_zero)
	print "Contours"
	contours, h = cv.findContours(img_bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
	contours = [c for c in contours if cv.contourArea(c) > component_size]
	blank = np.zeros((width,height), np.uint8)
	cv.drawContours(blank, contours, -1, color=255, thickness=2, lineType = 8)
	inv = 255 - blank

	print "Watershed"
	global labels
	lbl, ncc = label(inv, np.ones((3,3)))
	nbild = np.zeros((width, height, 3), np.uint8)
	cv.watershed(nbild, lbl)
	labels = lbl
	nbild[lbl != -1] = (255,255,255)

	global draw_on
	draw_on = nbild

	display(draw_on)
	print time.clock()-now
	if step >=2:
		calculate_graph()

def corners(bild):
	w,h = bild.shape[0], bild.shape[1]
	for x,y in np.argwhere(bild == -1):
		xl = max(0,x-1)
		xr = min(x+2,w)
		yl = max(0,y-1)
		yr = min(y+2,h)
		d = set(bild[xl:xr, yl:yr].flat)
		if len(d)>3: 
			d.remove(-1)
			yield (y,x), d

def calculate_graph():
	now = time.clock()
	print "corner detection"
	ecken = list(corners(labels))
	print time.clock() - now 

	now = time.clock()
	print "bubble detection"
	face_adj = dict()
	for v, faces in ecken:
			for f1 in faces:
				for f2 in faces:
					if f1>=f2:continue

					if not f1 in face_adj: face_adj[f1] = set()
					if not f2 in face_adj: face_adj[f2] = set()
					face_adj[f1].add(f2)
					face_adj[f2].add(f1)

	bubbles = set()
	while True:
		round_bubbles = [(x,n) for x,n in face_adj.iteritems() if len(n)<=2]
		if not round_bubbles: break
		for x,n in round_bubbles:
			bubbles.add(x)
			b = True
			del face_adj[x]
			for v in n:
				face_adj[v].remove(x)

	print len(bubbles), "bubbles detected"
	print time.clock() - now 

	now = time.clock()
	print "edge calculation"
	edge_incidence = dict()
	adj = dict()
	for v, faces in ecken:
		faces -= bubbles
		if len(faces) < 3: continue
		if not v in adj: adj[v] = []
		for f1 in faces:
			for f2 in faces:
				if f1>=f2:continue
				if (f1,f2) in edge_incidence:
					u = edge_incidence[(f1,f2)]
					adj[v].append(u)
					if not u in adj: adj[u] = []
					adj[u].append(v)
				else:
					edge_incidence[(f1,f2)] = v

	print time.clock()-now
	# draw_on = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
	draw_on = img.copy()
	for (x,y), neighbors in adj.iteritems():
		cv.circle(draw_on, (x,y), 4, (0,0,255), -1)
		for (x2,y2) in neighbors:
			cv.line(draw_on, (x,y), (x2, y2), pink, 2, lineType=cv.CV_AA)

	display(draw_on)
	cv.imwrite("./output.jpg", draw_on)

pink = (255,92,205)
img= setup()
print img.shape
width, height, channels = img.shape


img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    

contours()


cv.waitKey()



