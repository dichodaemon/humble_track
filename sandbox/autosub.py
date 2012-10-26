#!/usr/bin/python

import cv
import numpy as np
import random
import time
from scipy.cluster.vq import kmeans, vq

num_texcels = 16
num_hist = 7
window_size = 5

W1 = 0.5
W2 = 1.0
W3 = 0.5


def flatten( features ):
  return features.reshape( [features.shape[0] * features.shape[1], features.shape[2]] )

def compute_features( image ):
  result = np.zeros( image.shape[:2] + (11, ), np.float32 )
  for i in xrange( 1, image.shape[0] - 2 ):
    for j in xrange( 1, image.shape[1] - 2 ):
      result[i, j, 0]  = W1 * image[i, j, 0]
      result[i, j, 1]  = W2 * image[i, j, 1]
      result[i, j, 2]  = W2 * image[i, j, 2]

      l1 = image[i, j, 0] * 1.
      result[i, j, 3]  = W3 * ( l1 - image[i - 1, j - 1, 0] )
      result[i, j, 4]  = W3 * ( l1 - image[i    , j - 1, 0] )
      result[i, j, 5]  = W3 * ( l1 - image[i + 1, j - 1, 0] )
      result[i, j, 6]  = W3 * ( l1 - image[i - 1, j, 0] )
      result[i, j, 7]  = W3 * ( l1 - image[i + 1, j, 0] )
      result[i, j, 8]  = W3 * ( l1 - image[i - 1, j + 1, 0] )
      result[i, j, 9]  = W3 * ( l1 - image[i    , j + 1, 0] )
      result[i, j, 10] = W3 * ( l1 - image[i + 1, j + 1, 0] )
  return result

def compute_texcels( features, k = 16 ):
  centroids, distortion = kmeans( flatten( features ), k )
  return centroids

def integral_image( image ):
  result = image * 0
  for i in xrange( image.shape[0] ):
    for j in xrange( image.shape[1] ):
      r = image[i, j]
      if i > 0:
        r += result[i - 1, j]
      if j > 0:
        r += result[i, j - 1]
      if i > 0 and j > 0:
        r -= result[i - 1, j - 1]
      result[i, j] = r
  return result

def compute_histogram_features( labels, k = 16, window = 8 ):
  integrals = np.zeros( labels.shape[:2] + (k, ), np.float32 )
  for i in xrange( k ):
    integrals[:, :, i] = integral_image( ( labels == i ) * 1.0 )

  result = np.zeros( labels.shape[:2] + (k, ), np.float32 )

  for i in xrange( window, labels.shape[0] - window - 1 ) :
    for j in xrange( window, labels.shape[1] - window - 1 ) :
      result[i, j] = integrals[i + window, j + window] +\
                     integrals[i, j] -\
                     integrals[i + window, j] -\
                     integrals[i, j + window]
  return result

def cluster_histograms( features, k = 16 ):
  centroids, distortion = kmeans( flatten( features ), k )
  return centroids

def create_palette( colors = 16 ):
  samples = np.random.randint( 256, size = ( 1000, 3 ) )
  palette, _ = kmeans( samples, colors )
  return palette

image = cv.LoadImageM( "img/lobby1.jpg" )
cielab = cv.CreateMat( image.rows, image.cols, image.type )
cv.CvtColor( image, cielab, cv.CV_RGB2Lab )


print "Computing features"
t = time.time()
features = compute_features( np.asarray( cielab ) )
t = time.time() - t
print "%f seconds" % t
print "Computing texcels"
t = time.time()
texcels   = compute_texcels( features, num_texcels )
t = time.time() - t
print "%f seconds" % t
print "Labeling"
t = time.time()
labels, _ = vq( flatten( features ), texcels )
labels = labels.reshape( [cielab.rows, cielab.cols] ).astype( np.int8 )
t = time.time() - t
print "%f seconds" % t
print "Computing histograms"
t = time.time()
h_features = compute_histogram_features( labels, num_texcels, window_size )
t = time.time() - t
print "%f seconds" % t
print "Clustering histograms"
t = time.time()
h_codebook = cluster_histograms( h_features, num_hist )
t = time.time() - t
print "%f seconds" % t
print "Labeling"
t = time.time()
h_labels, _ = vq( flatten( h_features ), h_codebook )
h_labels = h_labels.reshape( [cielab.rows, cielab.cols] ).astype( np.int8 )
t = time.time() - t
print "%f seconds" % t

palette = create_palette( num_hist )
output = np.zeros( [cielab.rows, cielab.cols, 3] ).astype( np.int8 )
for k in xrange( num_hist ):
  output[h_labels == k] = palette[k]

label_img = cv.fromarray( output )

cv.NamedWindow( "Original" )
cv.ShowImage( "Original", image )
cv.NamedWindow( "Segmented" )
cv.ShowImage( "Segmented", label_img )
cv.WaitKey( 0 )
