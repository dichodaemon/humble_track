import cv
from util import *
import numpy as np
from scipy.cluster.vq import kmeans, vq

class Detector( object ):

  def __init__( 
    self, 
    num_texcels = 16, num_histograms = 7, window_size = 5, 
    w1 = 0.5, w2 = 1.0, w3 = 0.5
   ) :
    self.num_texcels = num_texcels
    self.num_histograms = num_histograms
    self.window_size = window_size
    self.w1 = w1
    self.w2 = w2
    self.w3 = w3

  def compute_texcels( self, frame ):
    result = np.zeros( frame.shape[:2] + (11, ), np.float32 )
    for i in xrange( 1, frame.shape[0] - 2 ):
      for j in xrange( 1, frame.shape[1] - 2 ):
        result[i, j, 0]  = self.w1 * frame[i, j, 0]
        result[i, j, 1]  = self.w2 * frame[i, j, 1]
        result[i, j, 2]  = self.w2 * frame[i, j, 2]

        l1 = frame[i, j, 0] * 1.
        result[i, j, 3]  = self.w3 * ( l1 - frame[i - 1, j - 1, 0] )
        result[i, j, 4]  = self.w3 * ( l1 - frame[i    , j - 1, 0] )
        result[i, j, 5]  = self.w3 * ( l1 - frame[i + 1, j - 1, 0] )
        result[i, j, 6]  = self.w3 * ( l1 - frame[i - 1, j, 0] )
        result[i, j, 7]  = self.w3 * ( l1 - frame[i + 1, j, 0] )
        result[i, j, 8]  = self.w3 * ( l1 - frame[i - 1, j + 1, 0] )
        result[i, j, 9]  = self.w3 * ( l1 - frame[i    , j + 1, 0] )
        result[i, j, 10] = self.w3 * ( l1 - frame[i + 1, j + 1, 0] )
    return result

  def learn( self, frame ):
    texcels = self.compute_texcels( frame )
    self.tc_codebook = kmeans( flatten( texcels ), self.num_texcels )
    tc_image, _ = vq( flatten( texcels ), self.tc_codebook )
    tc_image = tc_image.reshape( frame.shape[:2] )

    histograms = self.compute_histograms( tc_image )
    self.hg_codebook = kmeans( flatten( histograms ), self.num_histograms )

  def segment( self, frame ):
    texcels = self.compute_texcels( frame )
    tc_image, _ = vq( flatten( texcels ), self.tc_codebook )
    tc_image = tc_image.reshape( frame.shape[:2] )
    
    histograms = self.compute_histograms( tc_image )
    hg_image, _ = vq( flatten( histograms ), self.hg_codebook )
    h_labels = h_labels.reshape( frame.shape[:2] )
    return h_labels

